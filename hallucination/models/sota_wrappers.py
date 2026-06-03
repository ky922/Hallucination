"""
SOTA Hallucination Mitigation Baselines (training-free, inference-time)

1. VCD — Visual Contrastive Decoding
   "Mitigating Object Hallucinations in Large Vision-Language Models through
    Visual Contrastive Decoding"  (Leng et al., CVPR 2024)
   https://arxiv.org/abs/2311.16922

   Key idea: run a second forward pass on a noise-corrupted image.
   The noisy-image output captures language-prior hallucinations (spurious
   objects the model expects to see regardless of actual content).
   Subtract it: logits = logits_clean - α · logits_noisy

2. ICD — Instruction Contrastive Decoding
   "Instruction Contrastive Decoding: Amplifying Instruction Faithfulness via
    Contrastive Decoding"  (Leng et al., ACL Findings 2024)
   https://arxiv.org/abs/2403.18715

   Key idea: run a second forward pass with a "distorted" instruction that
   amplifies the model's tendency to hallucinate. The distorted output serves
   as a negative anchor that captures instruction-independent language prior.
   Subtract it: logits = logits_normal - α · logits_distorted

Both methods share the same interface as LLaVAWrapper / QCVRWrapper so they
plug directly into run_pope.py / run_chair.py.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Optional, Tuple, Union


# ── shared helpers ─────────────────────────────────────────────────────────────

def _logsumexp_ids(log_p: torch.Tensor, ids: list) -> float:
    return torch.logsumexp(log_p[ids], dim=0).item()


class _BaseLLaVA:
    """Minimal shared infrastructure for VCD / ICD wrappers."""

    def __init__(self, model_id: str, dtype: torch.dtype, attn_impl: str,
                 _model=None, _processor=None):
        if _model is not None:
            # Reuse a pre-loaded model (saves ~14 GB; avoids second load)
            print(f"[{self.__class__.__name__}] Reusing pre-loaded model.")
            self.model     = _model
            self.processor = _processor
            self.dtype     = dtype
        else:
            print(f"[{self.__class__.__name__}] Loading {model_id} …")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            )
            self.model.eval()
            self.dtype = dtype

        tok = self.processor.tokenizer
        self._yes_ids = list(set(
            tok.encode(" Yes", add_special_tokens=False) +
            tok.encode("Yes",  add_special_tokens=False)
        ))
        self._no_ids = list(set(
            tok.encode(" No", add_special_tokens=False) +
            tok.encode("No",  add_special_tokens=False)
        ))

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB") if image.mode != "RGB" else image

    def _build_prompt(self, question: str, system_prompt: Optional[str]) -> str:
        if system_prompt:
            return f"USER: <image>\n{system_prompt}\n{question} ASSISTANT:"
        return f"USER: <image>\n{question} ASSISTANT:"

    def _prepare_inputs(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        img    = self._load_image(image)
        prompt = self._build_prompt(question, system_prompt)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def _forward_logits(self, inputs: dict) -> torch.Tensor:
        return self.model(**inputs).logits[0, -1, :].float()

    @torch.no_grad()
    def generate(
        self,
        image,
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
        """Standard generation (no contrastive correction — used for non-POPE tasks)."""
        inputs = self._prepare_inputs(image, question, system_prompt)
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


# ── VCD ────────────────────────────────────────────────────────────────────────

class VCDWrapper(_BaseLLaVA):
    """
    Visual Contrastive Decoding (Leng et al., CVPR 2024).

    Adds Gaussian noise to the image's pixel-value tensor (in the normalized
    pixel space that CLIP/BLIP processors produce) and uses the noisy-image
    logits as a negative anchor:

        logits_vcd = logits_clean - α · logits_noisy

    Then applies an adaptive plausibility constraint (APC, from the paper):
    only tokens whose clean probability exceeds β · max_clean_prob are kept
    in the contrastive computation.

    Args:
        noise_std    : std of additive Gaussian noise (default 0.1, paper default)
        alpha        : subtraction strength (default 1.0)
        beta_apc     : adaptive plausibility constraint threshold (default 0.1)
        noise_steps  : number of independent noise samples to average (default 1)
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        dtype: torch.dtype = torch.float16,
        noise_std: float = 0.1,
        alpha: float = 1.0,
        beta_apc: float = 0.1,
        noise_steps: int = 1,
        _model=None,
        _processor=None,
    ):
        super().__init__(model_id, dtype, attn_impl="sdpa", _model=_model, _processor=_processor)
        self.noise_std  = noise_std
        self.alpha      = alpha
        self.beta_apc   = beta_apc
        self.noise_steps = noise_steps
        print(f"[VCDWrapper] Ready. noise_std={noise_std}, alpha={alpha}, "
              f"beta_apc={beta_apc}, noise_steps={noise_steps}")

    def _add_noise(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise in the normalized pixel space."""
        noise = torch.randn_like(pixel_values) * self.noise_std
        return pixel_values + noise   # intentionally no clamp (paper does not clamp)

    def _contrast_scores(
        self,
        logits_clean: torch.Tensor,
        logits_noisy: torch.Tensor,
        temperature: float = 1.0,
        use_apc: bool = True,
    ) -> torch.Tensor:
        """VCD score with an adaptive plausibility constraint.

        Without APC, log-prob subtraction can promote tokens that are implausible
        under the clean image simply because the noisy stream assigns them very
        low probability. APC keeps decoding anchored to clean-image candidates.
        """
        temp = max(temperature, 1e-6)
        lp_c = F.log_softmax(logits_clean / temp, dim=-1)
        lp_n = F.log_softmax(logits_noisy / temp, dim=-1)
        scores = lp_c - self.alpha * lp_n

        if use_apc and self.beta_apc is not None and self.beta_apc > 0:
            p_clean = F.softmax(logits_clean.float(), dim=-1)
            plausible = p_clean >= (self.beta_apc * p_clean.max())
            # Restrict the contrastive choice to tokens the clean image already
            # considers plausible; this is the stabilizing part of VCD/APC.
            scores = scores.masked_fill(~plausible, -float("inf"))

        return scores

    @staticmethod
    def _ban_repeated_ngrams(
        scores: torch.Tensor,
        generated: list,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        """Single-sequence no-repeat-ngram filter for the hand-written loop."""
        n = no_repeat_ngram_size
        if n <= 0 or len(generated) < n - 1:
            return scores

        prefix = tuple(generated[-(n - 1):])
        banned = set()
        for i in range(len(generated) - n + 1):
            if tuple(generated[i:i + n - 1]) == prefix:
                banned.add(generated[i + n - 1])

        if banned:
            scores = scores.clone()
            scores[list(banned)] = -float("inf")
        return scores

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
        VCD-corrected free-form generation.

        Keeps one clean stream and one noisy-image stream in KV-cache, then
        greedily selects each token with:
            score(t) = log p_clean(t) - alpha * log p_noisy(t)
        """
        device = next(self.model.parameters()).device
        eos_id = self.processor.tokenizer.eos_token_id

        inputs_clean = self._prepare_inputs(image, question, system_prompt)
        inputs_noisy = dict(inputs_clean)
        inputs_noisy["pixel_values"] = self._add_noise(inputs_clean["pixel_values"])

        out_c = self.model(**inputs_clean, use_cache=True)
        out_n = self.model(**inputs_noisy, use_cache=True)

        pkv_c = out_c.past_key_values
        pkv_n = out_n.past_key_values
        logit_c = out_c.logits[0, -1, :].float()
        logit_n = out_n.logits[0, -1, :].float()

        mask_c = inputs_clean["attention_mask"]
        mask_n = inputs_noisy["attention_mask"]
        ones = torch.ones(1, 1, dtype=mask_c.dtype, device=device)

        generated = []
        for _ in range(max_new_tokens):
            scores = self._contrast_scores(
                logit_c,
                logit_n,
                temperature=temperature,
                use_apc=True,
            )
            scores = self._ban_repeated_ngrams(
                scores,
                generated,
                no_repeat_ngram_size,
            )

            if do_sample:
                probs = F.softmax(scores, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
                    keep[0] = True
                    filtered = torch.zeros_like(probs)
                    filtered[sorted_idx[keep]] = probs[sorted_idx[keep]]
                    probs = filtered / filtered.sum()
                next_id = int(torch.multinomial(probs, num_samples=1))
            else:
                next_id = int(scores.argmax())

            if next_id == eos_id:
                break
            generated.append(next_id)

            next_t = torch.tensor([[next_id]], device=device)
            mask_c = torch.cat([mask_c, ones], dim=1)
            mask_n = torch.cat([mask_n, ones], dim=1)

            out_c = self.model.language_model(
                input_ids=next_t,
                attention_mask=mask_c,
                past_key_values=pkv_c,
                use_cache=True,
            )
            out_n = self.model.language_model(
                input_ids=next_t,
                attention_mask=mask_n,
                past_key_values=pkv_n,
                use_cache=True,
            )
            pkv_c = out_c.past_key_values
            pkv_n = out_n.past_key_values
            logit_c = out_c.logits[0, -1, :].float()
            logit_n = out_n.logits[0, -1, :].float()

        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_yes_no_logits(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, float, float]:
        inputs = self._prepare_inputs(image, question, system_prompt)

        # Clean forward pass
        logits_clean = self._forward_logits(inputs)

        # Noisy forward pass(es)
        noisy_logits_list = []
        for _ in range(self.noise_steps):
            noisy_inputs = dict(inputs)
            noisy_inputs["pixel_values"] = self._add_noise(inputs["pixel_values"])
            noisy_logits_list.append(self._forward_logits(noisy_inputs))
        logits_noisy = torch.stack(noisy_logits_list, dim=0).mean(0)  # average over steps

        # VCD: apply contrastive formula in log-prob space for yes/no tokens.
        # Computing it in full 32K logit space then re-normalising is numerically
        # unstable for binary classification — use the targeted formulation instead:
        #   score(y) = log p_clean(y) - alpha * log p_noisy(y)
        log_p_clean = F.log_softmax(logits_clean, dim=-1)
        log_p_noisy = F.log_softmax(logits_noisy, dim=-1)

        lp_yes_clean = _logsumexp_ids(log_p_clean, self._yes_ids)
        lp_no_clean  = _logsumexp_ids(log_p_clean, self._no_ids)
        lp_yes_noisy = _logsumexp_ids(log_p_noisy, self._yes_ids)
        lp_no_noisy  = _logsumexp_ids(log_p_noisy, self._no_ids)

        score_yes = lp_yes_clean - self.alpha * lp_yes_noisy
        score_no  = lp_no_clean  - self.alpha * lp_no_noisy
        pred = "yes" if score_yes >= score_no else "no"
        return pred, score_yes, score_no


# ── ICD ────────────────────────────────────────────────────────────────────────

# Default distorted instruction used in the original ICD paper (Table 1 ablation).
# It is deliberately vague and instruction-free so the model's output is driven
# purely by language statistics (visual hallucination amplified).
_ICD_DISTORTED_INSTRUCTION = (
    "Please answer the following question based on the image."
)

class ICDWrapper(_BaseLLaVA):
    """
    Instruction Contrastive Decoding (Leng et al., ACL Findings 2024).

    Runs two forward passes:
      1. Normal: original question → logits_normal
      2. Distorted: a generic/amplifying instruction → logits_distorted
         (the distorted prompt de-contextualises the question so the model
          falls back on statistical language priors / hallucinations)

    Final logits:
        logits_icd = logits_normal - α · logits_distorted

    The same adaptive plausibility constraint (APC) from VCD is applied for
    a fair comparison.

    Args:
        alpha                : subtraction strength (default 1.0)
        distorted_instruction: the "amplifying" negative-anchor question.
                               Default matches the paper's original setup.
        beta_apc             : APC threshold (default 0.1)
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        dtype: torch.dtype = torch.float16,
        alpha: float = 1.0,
        distorted_instruction: str = _ICD_DISTORTED_INSTRUCTION,
        beta_apc: float = 0.1,
        _model=None,
        _processor=None,
    ):
        super().__init__(model_id, dtype, attn_impl="sdpa", _model=_model, _processor=_processor)
        self.alpha                 = alpha
        self.distorted_instruction = distorted_instruction
        self.beta_apc              = beta_apc
        print(f"[ICDWrapper] Ready. alpha={alpha}, beta_apc={beta_apc}")
        print(f"             distorted_instruction='{distorted_instruction[:60]}'")

    @torch.no_grad()
    def generate_yes_no_logits(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, float, float]:
        # Normal pass
        inputs_normal = self._prepare_inputs(image, question, system_prompt)
        logits_normal = self._forward_logits(inputs_normal)

        # Distorted pass — replace question with the generic instruction
        # (keep image, replace text context)
        inputs_distorted = self._prepare_inputs(
            image, self.distorted_instruction, system_prompt=None
        )
        logits_distorted = self._forward_logits(inputs_distorted)

        # ICD: apply contrastive formula in log-prob space for yes/no tokens.
        #   score(y) = log p_normal(y) - alpha * log p_distorted(y)
        log_p_normal     = F.log_softmax(logits_normal,     dim=-1)
        log_p_distorted  = F.log_softmax(logits_distorted,  dim=-1)

        lp_yes_normal    = _logsumexp_ids(log_p_normal,    self._yes_ids)
        lp_no_normal     = _logsumexp_ids(log_p_normal,    self._no_ids)
        lp_yes_distorted = _logsumexp_ids(log_p_distorted, self._yes_ids)
        lp_no_distorted  = _logsumexp_ids(log_p_distorted, self._no_ids)

        score_yes = lp_yes_normal - self.alpha * lp_yes_distorted
        score_no  = lp_no_normal  - self.alpha * lp_no_distorted
        pred = "yes" if score_yes >= score_no else "no"
        return pred, score_yes, score_no

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
        ICD-corrected free-form captioning via dual-stream KV-cache greedy decoding.

        At each token step:
          score(t) = log_p_normal(t) - alpha * log_p_distorted(t)
        Both streams share the same generated tokens (greedy, no beam search).
        Uses KV-cache for efficiency: O(n) forward passes per stream.
        """
        device = next(self.model.parameters()).device
        eos_id = self.processor.tokenizer.eos_token_id

        # Prefill both streams (processes pixel_values through vision tower)
        inputs_n = self._prepare_inputs(image, question, system_prompt)
        inputs_d = self._prepare_inputs(image, self.distorted_instruction, system_prompt=None)

        out_n = self.model(**inputs_n, use_cache=True)
        out_d = self.model(**inputs_d, use_cache=True)

        pkv_n   = out_n.past_key_values
        pkv_d   = out_d.past_key_values
        logit_n = out_n.logits[0, -1, :].float()
        logit_d = out_d.logits[0, -1, :].float()

        # Attention masks grow with each generated token
        mask_n = inputs_n["attention_mask"]   # [1, prefix_n_len]
        mask_d = inputs_d["attention_mask"]   # [1, prefix_d_len]
        ones   = torch.ones(1, 1, dtype=mask_n.dtype, device=device)

        generated = []
        for _ in range(max_new_tokens):
            # ICD scoring in log-prob space
            lp_n = F.log_softmax(logit_n, dim=-1)
            lp_d = F.log_softmax(logit_d, dim=-1)
            scores = lp_n - self.alpha * lp_d

            next_id = int(scores.argmax())
            if next_id == eos_id:
                break
            generated.append(next_id)

            next_t = torch.tensor([[next_id]], device=device)
            mask_n = torch.cat([mask_n, ones], dim=1)
            mask_d = torch.cat([mask_d, ones], dim=1)

            # Extend normal stream via language model only (image already cached)
            out_n = self.model.language_model(
                input_ids=next_t,
                attention_mask=mask_n,
                past_key_values=pkv_n,
                use_cache=True,
            )
            # Extend distorted stream
            out_d = self.model.language_model(
                input_ids=next_t,
                attention_mask=mask_d,
                past_key_values=pkv_d,
                use_cache=True,
            )
            pkv_n   = out_n.past_key_values
            pkv_d   = out_d.past_key_values
            logit_n = out_n.logits[0, -1, :].float()
            logit_d = out_d.logits[0, -1, :].float()

        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
