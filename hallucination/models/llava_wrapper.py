"""
LLaVA-1.5-7B wrapper supporting multiple generation strategies.

Generation modes:
  greedy       — do_sample=False, num_beams=1  (fastest, deterministic)
  beam_search  — do_sample=False, num_beams=N, length_penalty, no_repeat_ngram_size
  sampling     — do_sample=True, temperature, top_p
  prompt_eng   — any of the above + a system prompt prepended to the question

POPE-specific:
  generate_yes_no_logits() — returns log P("yes") and log P("no") directly from
  the first generated token's logits, avoiding free-text parse bias.

Usage:
    from models.llava_wrapper import LLaVAWrapper
    model = LLaVAWrapper()
    ans   = model.generate(image_path, "Is there a cat?")
    yn    = model.generate_yes_no_logits(image_path, "Is there a cat?")
"""

import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union


class LLaVAWrapper:
    """LLaVA-1.5-7B with configurable decoding for baseline comparison."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        print(f"[LLaVAWrapper] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Cache yes/no token ids — LLaVA tokenizer may produce multiple ids per word;
        # we take the first subword token of " Yes" / " No" (with leading space).
        tok = self.processor.tokenizer
        self._yes_ids: list = tok.encode(" Yes", add_special_tokens=False)
        self._no_ids:  list = tok.encode(" No",  add_special_tokens=False)
        # Also handle bare capitalisation variants
        self._yes_ids += tok.encode("Yes", add_special_tokens=False)
        self._no_ids  += tok.encode("No",  add_special_tokens=False)
        # deduplicate
        self._yes_ids = list(set(self._yes_ids))
        self._no_ids  = list(set(self._no_ids))

        print(f"[LLaVAWrapper] Model ready. "
              f"yes_ids={self._yes_ids}, no_ids={self._no_ids}")

    # ── internal helpers ───────────────────────────────────────────────────────

    def _build_prompt(self, question: str, system_prompt: Optional[str]) -> str:
        """Construct LLaVA-1.5 conversation string."""
        if system_prompt:
            return f"USER: <image>\n{system_prompt}\n{question} ASSISTANT:"
        return f"USER: <image>\n{question} ASSISTANT:"

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB") if image.mode != "RGB" else image

    def _prepare_inputs(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        img    = self._load_image(image)
        prompt = self._build_prompt(question, system_prompt)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    # ── public API ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str] = None,
        # decoding knobs
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        max_new_tokens: int = 128,
        # beam quality controls
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = False,
    ) -> str:
        """
        Generate a free-text response for one (image, question) pair.

        New beam-quality args:
            length_penalty        — >1 favours longer sequences, <1 shorter
            no_repeat_ngram_size  — prevents n-gram repetition (0 = disabled)
            early_stopping        — stop beam search when all beams hit EOS

        Returns:
            Decoded response string (stripped).
        """
        inputs = self._prepare_inputs(image, question, system_prompt)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping if num_beams > 1 else False,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"]       = top_p

        out_ids  = self.model.generate(**inputs, **gen_kwargs)
        new_ids  = out_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_yes_no_logits(
        self,
        image: Union[Image.Image, str],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, float, float]:
        """
        POPE-optimised inference: compare log P("Yes") vs log P("No") directly
        from the first generated token's logits — no free-text string matching.

        This eliminates the well-known "yes-bias" from greedy text generation
        and is the standard practice in recent POPE papers.

        Returns:
            (prediction, log_p_yes, log_p_no)
            prediction: "yes" if log_p_yes > log_p_no else "no"
            log_p_yes / log_p_no: log-probabilities of the two options
        """
        inputs = self._prepare_inputs(image, question, system_prompt)

        # One forward pass — we only need the next-token logits
        outputs = self.model(**inputs)
        logits  = outputs.logits[0, -1, :]          # [vocab_size]
        log_p   = F.log_softmax(logits.float(), dim=-1)

        # Aggregate log-probs over all surface forms of yes / no
        lp_yes = torch.logsumexp(log_p[self._yes_ids], dim=0).item()
        lp_no  = torch.logsumexp(log_p[self._no_ids],  dim=0).item()

        pred = "yes" if lp_yes >= lp_no else "no"
        return pred, lp_yes, lp_no

    def _prepare_batch_inputs(
        self,
        images: List[Union[Image.Image, str]],
        questions: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build batched inputs with left-padding (so logits[:, -1, :] is always valid)."""
        if system_prompts is None:
            system_prompts = [None] * len(images)
        pil_imgs = [self._load_image(img) for img in images]
        prompts  = [self._build_prompt(q, sp)
                    for q, sp in zip(questions, system_prompts)]
        orig_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=prompts, images=pil_imgs,
            return_tensors="pt", padding=True,
        )
        self.processor.tokenizer.padding_side = orig_side
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def generate_yes_no_logits_batch(
        self,
        images: List[Union[Image.Image, str]],
        questions: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
        batch_size: int = 8,
    ) -> List[Tuple[str, float, float]]:
        """
        Batched POPE inference — significant GPU speedup over per-sample calls.

        Returns list of (prediction, lp_yes, lp_no) tuples, one per input.
        batch_size: number of samples processed per GPU call (tune for VRAM).
        """
        results: List[Tuple[str, float, float]] = []
        n = len(images)
        if system_prompts is None:
            system_prompts = [None] * n

        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            b_img = images[start:end]
            b_q   = questions[start:end]
            b_sp  = system_prompts[start:end]

            inputs = self._prepare_batch_inputs(b_img, b_q, b_sp)
            outputs = self.model(**inputs)
            # Left-padded → last token position is always -1 for all samples
            logits_last = outputs.logits[:, -1, :].float()  # [B, V]
            log_p = F.log_softmax(logits_last, dim=-1)      # [B, V]

            yes_t = torch.tensor(self._yes_ids, device=log_p.device)
            no_t  = torch.tensor(self._no_ids,  device=log_p.device)
            lp_yes = torch.logsumexp(log_p[:, yes_t], dim=-1)  # [B]
            lp_no  = torch.logsumexp(log_p[:, no_t],  dim=-1)  # [B]

            for j in range(end - start):
                y, n_ = lp_yes[j].item(), lp_no[j].item()
                results.append(("yes" if y >= n_ else "no", y, n_))

        return results
