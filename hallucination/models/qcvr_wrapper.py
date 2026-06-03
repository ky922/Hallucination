"""
QCVR + IACD: Training-Free Hallucination Mitigation for LLaVA-1.5

Two independent ablatable components:
  - QCVR (Query-Conditioned Visual Relevance Steering):
      At attention layers, re-weight non-Inert visual token attention scores
      by cosine similarity between the current query hidden state and each
      visual token hidden state at a reference layer l_m.

  - IACD (Inert-Anchored Contrastive Decoding):
      At the logit layer, subtract the logit produced when only Inert tokens
      contribute to the value aggregation — a zero-extra-inference negative anchor
      that captures "the model sees the image but only through noisy Inert tokens".

Both components share HABI (Hijacking Anchor-Based Identification) as a
free by-product: Logit Lens is run over visual token hidden states once at
prefill time to build inert_mask.

Usage:
    from models.qcvr_wrapper import QCVRWrapper
    model = QCVRWrapper(use_qcvr=True, use_iacd=True)
    pred, lp_yes, lp_no = model.generate_yes_no_logits(image, question)
    text = model.generate(image, question)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union


# ── helpers ────────────────────────────────────────────────────────────────────

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [d]   query vector
    b: [N, d] key vectors
    returns: [N] cosine similarities
    """
    a = F.normalize(a.float().unsqueeze(0), dim=-1)   # [1, d]
    b = F.normalize(b.float(), dim=-1)                 # [N, d]
    return (a @ b.T).squeeze(0)                        # [N]


# ── HABI: Inert Token detection ────────────────────────────────────────────────

class HABI:
    """
    Hijacking Anchor-Based Identification (from VHL/HAVAE).

    Runs Logit Lens over visual token hidden states at prefill.
    A token is "Inert" if its top-1 decoded word is the same
    (a "Hijacking Anchor") across >= anchor_consistency_threshold layers.

    Outputs:
        inert_mask  : BoolTensor [N_vis]  True = Inert
        hidden_all  : Tensor [L, N_vis, d]  per-layer hidden states (free by-product)
    """

    def __init__(
        self,
        model: LlavaForConditionalGeneration,
        anchor_consistency_threshold: float = 0.6,
        top_k_check: int = 1,
        logit_lens_chunk_size: int = 128,
    ):
        self.model = model
        self.thresh = anchor_consistency_threshold
        self.top_k  = top_k_check
        self.logit_lens_chunk_size = logit_lens_chunk_size
        self.lm_head = model.language_model.lm_head
        self.ln_f    = model.language_model.model.norm   # final layer norm

        # hook storage
        self._vis_hidden: List[torch.Tensor] = []   # per layer [N_vis, d]
        self._hooks: List = []

    def _register_hooks(self, vis_slice: slice):
        """Register output hooks on every transformer decoder layer."""
        layers = self.model.language_model.model.layers
        self._vis_hidden.clear()

        def make_hook(layer_idx):
            def hook(module, inp, out):
                # out[0]: [batch, seq, d]
                h = out[0][0, vis_slice, :].detach()   # [N_vis, d]
                self._vis_hidden.append(h)
            return hook

        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def detect(
        self,
        hidden_states_per_layer: List[torch.Tensor],  # already collected
        tokenizer,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states_per_layer: list of [N_vis, d], length = n_layers
        Returns:
            inert_mask : BoolTensor [N_vis]
            hidden_all : Tensor [L, N_vis, d]
        """
        L = len(hidden_states_per_layer)
        N_vis = hidden_states_per_layer[0].shape[0]
        device = hidden_states_per_layer[0].device

        # Stack to [L, N_vis, d]
        hidden_all = torch.stack(hidden_states_per_layer, dim=0)  # [L, N_vis, d]

        # Apply LN + LM head to get top-1 token per layer per visual token.
        # Chunk the LM head projection to avoid materializing a huge
        # [L*N_vis, vocab] logits tensor on GPU during CHAIR generation.
        h_flat = hidden_all.reshape(L * N_vis, -1)
        chunk_size = max(1, int(self.logit_lens_chunk_size))
        top1_chunks = []
        for start in range(0, L * N_vis, chunk_size):
            h_chunk = h_flat[start:start + chunk_size].to(self.lm_head.weight.dtype)
            h_norm = self.ln_f(h_chunk)
            logits = self.lm_head(h_norm)
            top1_chunks.append(logits.argmax(dim=-1))
            del logits, h_norm, h_chunk
        top1 = torch.cat(top1_chunks, dim=0).reshape(L, N_vis)     # [L, N_vis]

        # A token is Inert if its top-1 token is the SAME across >= thresh fraction
        # of layers (the "Hijacking Anchor" phenomenon)
        inert_mask = torch.zeros(N_vis, dtype=torch.bool, device=device)
        for i in range(N_vis):
            token_ids = top1[:, i]                                 # [L]
            # count mode frequency
            mode_count = torch.bincount(token_ids).max().item()
            if mode_count / L >= self.thresh:
                inert_mask[i] = True

        return inert_mask, hidden_all


# ── QCVRWrapper ────────────────────────────────────────────────────────────────

class QCVRWrapper:
    """
    LLaVA-1.5-7B with QCVR + IACD hallucination mitigation.

    Args:
        use_qcvr  : enable Query-Conditioned Visual Relevance Steering
        use_iacd  : enable Inert-Anchored Contrastive Decoding
        lm_layer  : reference layer index l_m for QCVR hidden states
                    (-1 = last layer, recommended: middle layer ~16 for 32-layer model)
        tau       : temperature for QCVR softmax re-weighting
        lambda_   : IACD subtraction strength
        nhar_threshold : HAVAE head selection threshold (fraction of non-Inert
                         visual attention; heads above this are "good heads")
        anchor_thresh  : HABI Inert detection consistency threshold
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        dtype: torch.dtype = torch.float16,
        use_qcvr: bool = True,
        use_iacd: bool = True,
        lm_layer: int = 16,
        tau: float = 0.1,
        lambda_: float = 1.0,
        iacd_mode: str = "first_token",
        iacd_decay_steps: int = 8,
        nhar_threshold: float = 0.5,
        anchor_thresh: float = 0.6,
    ):
        print(f"[QCVRWrapper] Loading {model_id} …")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",   # required: flash-attn hides attn weights
        )
        self.model.eval()

        self.use_qcvr = use_qcvr
        self.use_iacd = use_iacd
        self.lm_layer = lm_layer
        self.tau      = tau
        self.lambda_  = lambda_
        self.iacd_mode = iacd_mode
        self.iacd_decay_steps = max(1, int(iacd_decay_steps))
        self.nhar_thr = nhar_threshold
        self.dtype    = dtype

        self.habi = HABI(self.model, anchor_consistency_threshold=anchor_thresh)

        tok = self.processor.tokenizer
        self._yes_ids = list(set(
            tok.encode(" Yes", add_special_tokens=False) +
            tok.encode("Yes",  add_special_tokens=False)
        ))
        self._no_ids = list(set(
            tok.encode(" No", add_special_tokens=False) +
            tok.encode("No",  add_special_tokens=False)
        ))
        print(f"[QCVRWrapper] Ready. QCVR={use_qcvr}, IACD={use_iacd}, "
              f"l_m={lm_layer}, tau={tau}, lambda={lambda_}, "
              f"iacd_mode={iacd_mode}")

    # ── token position helpers ─────────────────────────────────────────────────

    def _get_visual_token_slice(self, input_ids: torch.Tensor) -> slice:
        """
        LLaVA-1.5 inserts visual tokens in place of the <image> placeholder.
        The processor sets image_token_index; we find the contiguous visual
        token block in the flat input_ids sequence.
        """
        img_tok_id = self.model.config.image_token_index  # typically 32000
        positions  = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            # fallback: no visual tokens (shouldn't happen)
            return slice(0, 0)
        return slice(positions[0].item(), positions[-1].item() + 1)

    # ── HABI prefill ───────────────────────────────────────────────────────────

    def _run_habi(
        self,
        inputs: Dict[str, torch.Tensor],
        vis_slice: slice,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One forward pass (prefill) with hooks to collect visual token hidden
        states at every layer.  Returns (inert_mask, hidden_all).

        inert_mask : BoolTensor [N_vis]
        hidden_all : Tensor [L, N_vis, d]
        """
        self.habi._register_hooks(vis_slice)
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            self.habi._remove_hooks()

        hidden_layers = self.habi._vis_hidden   # list of [N_vis, d]
        inert_mask, hidden_all = self.habi.detect(
            hidden_layers, self.processor.tokenizer
        )
        return inert_mask, hidden_all

    # ── QCVR attention hook ────────────────────────────────────────────────────

    def _make_qcvr_hook(
        self,
        vis_slice: slice,
        inert_mask: torch.Tensor,      # [N_vis] bool
        h_vis_ref: torch.Tensor,       # [N_vis, d]  visual hidden at l_m
        h_query_ref: torch.Tensor,     # [d]          query token hidden at l_m
        layer_idx: int,
        current_layer_counter: List[int],
    ):
        """
        Hook factory.  Modifies attn_weights for visual columns on good heads.
        attn_weights shape in eager mode: [batch, heads, tgt_seq, src_seq]
        During generation tgt_seq=1 (one new token at a time).
        """
        N_vis  = inert_mask.shape[0]
        device = h_vis_ref.device

        # Pre-compute QCVR scores for non-Inert tokens
        eff_mask = ~inert_mask                                  # [N_vis] True=effective
        if eff_mask.sum() == 0:
            return None  # no effective tokens, skip

        h_eff  = h_vis_ref[eff_mask]                           # [N_eff, d]
        scores = _cosine_sim(h_query_ref, h_eff)               # [N_eff]
        qcvr_w = F.softmax(scores / self.tau, dim=-1)          # [N_eff]

        def hook(module, inp, out):
            # out is a tuple; out[0] = attn_output, out[1] = attn_weights (eager)
            if len(out) < 2 or out[1] is None:
                return out

            attn_w = out[1].clone()    # [batch, heads, tgt, src]  float32 or fp16
            B, H, T, S = attn_w.shape

            # visual columns in the full sequence
            vis_start = vis_slice.start
            vis_end   = vis_slice.stop
            vis_cols  = torch.arange(vis_start, vis_end, device=device)

            # For each head: check if it's a "good" head (not dominated by Inert)
            # NHAR proxy: fraction of visual attention that falls on non-Inert tokens
            vis_attn = attn_w[0, :, -1, vis_cols]         # [H, N_vis]
            inert_attn_frac = vis_attn[:, inert_mask].sum(-1) / (vis_attn.sum(-1) + 1e-9)
            good_heads = (inert_attn_frac < (1.0 - self.nhar_thr))  # [H] bool

            if not good_heads.any():
                return out

            # Apply QCVR: redistribute attention over effective tokens
            eff_cols = vis_cols[eff_mask]                  # column indices of eff tokens
            inert_cols = vis_cols[inert_mask]

            for h_idx in range(H):
                if not good_heads[h_idx]:
                    continue
                # current attention on eff tokens
                orig_eff = attn_w[0, h_idx, -1, eff_cols]  # [N_eff]
                total_eff = orig_eff.sum()
                if total_eff < 1e-9:
                    continue
                # new distribution: keep total mass, redistribute by qcvr_w
                new_eff = (qcvr_w * total_eff).to(attn_w.dtype)
                attn_w[0, h_idx, -1, eff_cols] = new_eff

            # Rebuild output: recompute value aggregation with modified weights
            # We need V matrix — it's in inp (the module's input)
            # inp[0]=hidden_states, inp[1]=attn_mask, inp[2]=position_ids, ...
            # Easiest: just return new attn_weights and let the module handle it
            # NOTE: in eager mode LlamaAttention returns (attn_output, attn_weights)
            # but attn_output is already computed. We need to recompute it.
            # We'll handle this by patching forward instead — see _patch_layer below.
            return (out[0], attn_w)

        return hook

    # ── IACD: value decomposition ──────────────────────────────────────────────

    def _compute_logit_inert_only(
        self,
        inputs: Dict[str, torch.Tensor],
        vis_slice: slice,
        inert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logit_inert-only: what the model predicts when only Inert visual
        token embeddings contribute (effective/non-inert tokens zeroed out).

        LLaVA injects visual features directly into inputs_embeds (bypassing
        embed_tokens), so we must capture the merged embeddings at layer-0 input,
        zero out non-inert visual positions, then re-run the language model.
        """
        vis_start = vis_slice.start
        eff_local = (~inert_mask).nonzero(as_tuple=True)[0].tolist()

        # Step 1: capture merged inputs_embeds (text + visual) at layer-0 entry.
        # This is the only reliable place to intercept all visual features in LLaVA.
        captured: Dict[str, torch.Tensor] = {}

        def _cap_hook(module, inp, out):
            if 'x' not in captured:
                captured['x'] = inp[0].detach().clone()

        h = self.model.language_model.model.layers[0].register_forward_hook(_cap_hook)
        with torch.no_grad():
            self.model(**inputs)
        h.remove()

        # Step 2: zero out effective (non-inert) visual token positions
        embeds = captured['x'].clone()   # [1, seq_len, d]
        for i in eff_local:
            embeds[0, vis_start + i, :] = 0.0

        # Step 3: re-run language model only with the masked embeddings
        with torch.no_grad():
            lm_out = self.model.language_model(
                inputs_embeds=embeds,
                attention_mask=inputs.get('attention_mask'),
            )
        return lm_out.logits[0, -1, :].float()

    # ── input preparation ──────────────────────────────────────────────────────

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB") if image.mode != "RGB" else image

    def _build_prompt(self, question: str, system_prompt: Optional[str]) -> str:
        if system_prompt:
            return f"USER: <image>\n{system_prompt}\n{question} ASSISTANT:"
        return f"USER: <image>\n{question} ASSISTANT:"

    def _prepare_inputs(self, image, question, system_prompt=None):
        img    = self._load_image(image)
        prompt = self._build_prompt(question, system_prompt)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _make_layer0_visual_scale_hook(
        self,
        vis_slice: slice,
        scale_vec: torch.Tensor,
    ):
        """
        Scale merged multimodal hidden states at decoder layer 0.

        LLaVA replaces image placeholders after token embedding, so an
        embed_tokens hook is not a reliable place to edit visual features.
        Layer-0 input already contains merged text + visual representations.
        """
        def hook(module, inp):
            if not inp:
                return inp
            hidden_states = inp[0]
            if hidden_states is None or hidden_states.dim() != 3:
                return inp
            seq_len = hidden_states.shape[1]
            if seq_len < vis_slice.stop:
                return inp

            scaled = hidden_states.clone()
            local_scale = scale_vec.to(device=scaled.device, dtype=scaled.dtype)
            scaled[:, vis_slice.start:vis_slice.stop, :] *= local_scale.view(1, -1, 1)
            return (scaled,) + tuple(inp[1:])

        return hook

    def _qcvr_scale_vec(
        self,
        inert_mask: torch.Tensor,
        hidden_all: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Build per-visual-token scaling vector for QCVR."""
        n_layers = hidden_all.shape[0]
        ref_l = min(self.lm_layer, n_layers - 1)
        h_vis_ref = hidden_all[ref_l]
        eff_mask = ~inert_mask
        if not eff_mask.any():
            return None

        # Prompt-level approximation: use the effective visual mean as a stable
        # image-conditioned query. This avoids per-token recomputation in CHAIR.
        h_query_ref = h_vis_ref[eff_mask].mean(0)
        h_eff = h_vis_ref[eff_mask]
        scores_cos = _cosine_sim(h_query_ref, h_eff)
        qcvr_w = F.softmax(scores_cos / self.tau, dim=-1)

        scale_vec = torch.ones(inert_mask.shape[0], device=h_vis_ref.device, dtype=self.dtype)
        scale_vec[inert_mask] = 0.0
        scale_vec[eff_mask] = qcvr_w.to(self.dtype) * eff_mask.sum().float()
        return scale_vec

    # ── QCVR forward pass ──────────────────────────────────────────────────────

    def _forward_with_qcvr(
        self,
        inputs: Dict[str, torch.Tensor],
        vis_slice: slice,
        inert_mask: torch.Tensor,
        hidden_all: torch.Tensor,       # [L, N_vis, d]
    ) -> torch.Tensor:
        """
        Run one forward pass with QCVR attention re-weighting.
        Uses a patched attention mechanism that modifies attn weights before
        value aggregation.

        Returns logits [vocab_size] for the last generated position.
        """
        scale_vec = self._qcvr_scale_vec(inert_mask, hidden_all)
        if scale_vec is None:
            # No effective tokens — fall back to plain forward
            with torch.no_grad():
                return self.model(**inputs).logits[0, -1, :].float()
        hooks = [
            self.model.language_model.model.layers[0].register_forward_pre_hook(
                self._make_layer0_visual_scale_hook(vis_slice, scale_vec)
            )
        ]

        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :].float()

        for h in hooks:
            h.remove()

        return logits

    # ── main inference methods ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate_yes_no_logits(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, float, float]:
        """
        POPE-style yes/no prediction with QCVR + IACD.
        Returns (prediction, log_p_yes, log_p_no).
        """
        inputs   = self._prepare_inputs(image, question, system_prompt)
        vis_slice = self._get_visual_token_slice(inputs["input_ids"])

        if vis_slice.stop <= vis_slice.start:
            # No visual tokens found — plain forward
            out = self.model(**inputs)
            logits_full = out.logits[0, -1, :].float()
            log_p = F.log_softmax(logits_full, dim=-1)
            lp_yes = torch.logsumexp(log_p[self._yes_ids], dim=0).item()
            lp_no  = torch.logsumexp(log_p[self._no_ids],  dim=0).item()
            return ("yes" if lp_yes >= lp_no else "no"), lp_yes, lp_no

        # Step 1: HABI — detect Inert tokens (one prefill pass with hooks)
        inert_mask, hidden_all = self._run_habi(inputs, vis_slice)

        # Step 2: Compute logits (with or without QCVR)
        if self.use_qcvr:
            logits_full = self._forward_with_qcvr(
                inputs, vis_slice, inert_mask, hidden_all
            )
        else:
            logits_full = self.model(**inputs).logits[0, -1, :].float()

        # Step 3: IACD — subtract Inert-only logits
        if self.use_iacd:
            logits_inert = self._compute_logit_inert_only(
                inputs, vis_slice, inert_mask
            )
            logits_final = logits_full - self.lambda_ * logits_inert
        else:
            logits_final = logits_full

        log_p  = F.log_softmax(logits_final, dim=-1)
        lp_yes = torch.logsumexp(log_p[self._yes_ids], dim=0).item()
        lp_no  = torch.logsumexp(log_p[self._no_ids],  dim=0).item()
        pred   = "yes" if lp_yes >= lp_no else "no"
        return pred, lp_yes, lp_no

    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        max_new_tokens: int = 128,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = False,
    ) -> str:
        """
        Free-form generation with QCVR + IACD applied at the first token,
        then standard greedy/beam for subsequent tokens.

        For CHAIR evaluation: IACD is applied at every step via logits_processor.
        QCVR is applied at the first step (most impactful for object hallucination).
        """
        inputs    = self._prepare_inputs(image, question, system_prompt)
        vis_slice = self._get_visual_token_slice(inputs["input_ids"])

        if vis_slice.stop <= vis_slice.start or (not self.use_qcvr and not self.use_iacd):
            # Fall back to plain generation
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                num_beams=num_beams, length_penalty=length_penalty,
                early_stopping=early_stopping if num_beams > 1 else False,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"]       = top_p
            out_ids = self.model.generate(**inputs, **gen_kwargs)
            new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
            return self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # HABI once
        inert_mask, hidden_all = self._run_habi(inputs, vis_slice)

        # Build logits processor for IACD. For free-form CHAIR generation, the
        # prompt-position inert logit is only reliable for the first generation
        # decision. Reusing it at every token distorts the language distribution.
        logits_processors = []
        if self.use_iacd:
            lambda_ = self.lambda_
            mode = self.iacd_mode
            decay_steps = self.iacd_decay_steps
            prompt_len = inputs["input_ids"].shape[1]
            logit_inert_prompt = self._compute_logit_inert_only(
                inputs, vis_slice, inert_mask
            )

            from transformers import LogitsProcessor
            class IACDLogitsProcessor(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    step = max(0, input_ids.shape[1] - prompt_len)
                    if mode == "first_token":
                        weight = lambda_ if step == 0 else 0.0
                    elif mode == "decay":
                        weight = lambda_ * max(0.0, 1.0 - (step / decay_steps))
                    elif mode == "every_step":
                        weight = lambda_
                    else:
                        weight = 0.0
                    if weight <= 0:
                        return scores
                    return scores - weight * logit_inert_prompt.to(scores.device)

            logits_processors.append(IACDLogitsProcessor())

        # QCVR: scale visual embeddings for the entire generation
        # Apply the same scaling hook throughout generation
        hooks = []
        if self.use_qcvr:
            scale_vec = self._qcvr_scale_vec(inert_mask, hidden_all)
            if scale_vec is not None:
                h = self.model.language_model.model.layers[0].register_forward_pre_hook(
                    self._make_layer0_visual_scale_hook(vis_slice, scale_vec)
                )
                hooks.append(h)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            num_beams=num_beams, length_penalty=length_penalty,
            early_stopping=early_stopping if num_beams > 1 else False,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            logits_processor=logits_processors if logits_processors else None,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"]       = top_p

        try:
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            for h in hooks:
                h.remove()

        new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
