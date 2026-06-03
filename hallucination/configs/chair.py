"""CHAIR baseline and method presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Tuple


QUESTION = "Please describe this image in detail."

GENERATION_KEYS = {
    "do_sample",
    "temperature",
    "top_p",
    "num_beams",
    "length_penalty",
    "no_repeat_ngram_size",
    "early_stopping",
    "system_prompt",
    "max_new_tokens",
}

QCVR_INIT_KEYS = {
    "use_qcvr",
    "use_iacd",
    "lm_layer",
    "tau",
    "lambda_",
    "iacd_mode",
    "iacd_decay_steps",
    "question",
}


BASELINES: Dict[str, dict] = {
    "greedy": {
        "description": "Standard greedy decoding",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 64,
    },
    "beam_search": {
        "description": "Beam search (n=4, length_penalty=1.0, no-repeat=3)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 4,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "system_prompt": None,
        "max_new_tokens": 64,
    },
    "sampling_low_temp": {
        "description": "Sampling (t=0.3, top_p=0.9) low-temperature",
        "question": QUESTION,
        "stochastic": True,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 64,
    },
    "prompt_careful": {
        "description": "Careful grounding prompt + greedy",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": (
            "Describe only what you can clearly see in the image. "
            "Do not mention objects that are not visibly present."
        ),
        "max_new_tokens": 64,
    },
    "greedy_short_norepeat": {
        "description": "Greedy decoding control (48 tokens, no-repeat=3)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
    },
}


SOTA_BASELINES: Dict[str, dict] = {
    "vcd": {
        "description": "VCD greedy (visual contrastive decoding, alpha=1.0)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 64,
        "model_class": "VCDWrapper",
        "init_kwargs": {"noise_std": 0.1, "alpha": 1.0, "beta_apc": 0.1},
    },
    "vcd_tuned": {
        "description": "VCD tuned (APC, alpha=0.2, short no-repeat decoding)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "model_class": "VCDWrapper",
        "init_kwargs": {
            "noise_std": 0.05,
            "alpha": 0.2,
            "beta_apc": 0.05,
            "noise_steps": 3,
        },
    },
    "icd": {
        "description": "ICD greedy (dual-stream contrastive, alpha=1.0)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 64,
        "model_class": "ICDWrapper",
        "init_kwargs": {"alpha": 1.0, "beta_apc": 0.1},
    },
}


QCVR_CHAIR_BASELINES: Dict[str, dict] = {
    "qcvr_iacd": {
        "description": "QCVR + IACD (first-token, tuned for CHAIR)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "use_qcvr": True,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.2,
        "lambda_": 0.3,
        "iacd_mode": "first_token",
        "iacd_decay_steps": 8,
    },
    "qcvr_only": {
        "description": "QCVR only (zero-IACD control, CHAIR ablation)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "use_qcvr": True,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.2,
        "lambda_": 0.0,
        "iacd_mode": "first_token",
        "iacd_decay_steps": 8,
    },
    "iacd_first_token": {
        "description": "IACD first-token only (no QCVR, CHAIR ablation)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "use_qcvr": False,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.2,
        "lambda_": 0.3,
        "iacd_mode": "first_token",
        "iacd_decay_steps": 8,
    },
    "qcvr_iacd_decay": {
        "description": "QCVR + IACD decay (lambda=0.3, tau=0.2)",
        "question": QUESTION,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "use_qcvr": True,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.2,
        "lambda_": 0.3,
        "iacd_mode": "decay",
        "iacd_decay_steps": 8,
    },
}


DEFAULT_BASELINES = {
    k: BASELINES[k]
    for k in ("greedy", "beam_search", "sampling_low_temp", "prompt_careful")
}
DEFAULT_SOTA = {k: SOTA_BASELINES[k] for k in ("vcd", "icd")}
DEFAULT_QCVR = {"qcvr_iacd": QCVR_CHAIR_BASELINES["qcvr_iacd"]}


def baseline_choices() -> list[str]:
    return (
        list(BASELINES)
        + list(SOTA_BASELINES)
        + list(QCVR_CHAIR_BASELINES)
        + ["all", "required", "sota_all", "qcvr_all", "full"]
    )


def _copy(configs: Dict[str, dict]) -> Dict[str, dict]:
    return deepcopy(configs)


def _filtered_sota(args, source: Dict[str, dict]) -> Dict[str, dict]:
    return {
        k: v for k, v in _copy(source).items()
        if not ((args.no_icd and k == "icd") or (args.no_vcd and k == "vcd"))
    }


def select_baseline_groups(args) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """Return LLaVA, SOTA, and QCVR baseline groups for the parsed args."""
    if args.key_only or args.baseline in ("all", "required"):
        llava_baselines = _copy(DEFAULT_BASELINES)
        sota_baselines = {} if args.no_sota else _filtered_sota(args, DEFAULT_SOTA)
        qcvr_baselines = {} if args.no_qcvr else _copy(DEFAULT_QCVR)
    elif args.baseline == "full":
        llava_baselines = {
            k: deepcopy(BASELINES[k])
            for k in ("greedy", "beam_search", "sampling_low_temp", "prompt_careful")
        }
        sota_baselines = _filtered_sota(args, SOTA_BASELINES)
        qcvr_baselines = {} if args.no_qcvr else {
            k: deepcopy(QCVR_CHAIR_BASELINES[k])
            for k in ("qcvr_iacd", "qcvr_only", "iacd_first_token")
        }
    elif args.baseline == "sota_all":
        llava_baselines = {}
        sota_baselines = _filtered_sota(args, SOTA_BASELINES)
        qcvr_baselines = {}
    elif args.baseline == "qcvr_all":
        llava_baselines = {}
        sota_baselines = {}
        qcvr_baselines = {} if args.no_qcvr else _copy(QCVR_CHAIR_BASELINES)
    elif args.baseline in SOTA_BASELINES:
        llava_baselines = {}
        sota_baselines = (
            {} if args.no_sota else {args.baseline: deepcopy(SOTA_BASELINES[args.baseline])}
        )
        qcvr_baselines = {}
    elif args.baseline in QCVR_CHAIR_BASELINES:
        llava_baselines = {}
        sota_baselines = {}
        qcvr_baselines = (
            {} if args.no_qcvr else {args.baseline: deepcopy(QCVR_CHAIR_BASELINES[args.baseline])}
        )
    else:
        llava_baselines = {args.baseline: deepcopy(BASELINES[args.baseline])}
        sota_baselines = {}
        qcvr_baselines = {}

    return llava_baselines, sota_baselines, qcvr_baselines


def apply_unified_decoding(
    groups: Iterable[Dict[str, dict]],
    max_new_tokens: int,
    no_repeat_ngram_size: int,
) -> None:
    """Force a shared decoding budget across selected CHAIR methods."""
    if max_new_tokens <= 0 and no_repeat_ngram_size < 0:
        return

    for group in groups:
        for cfg in group.values():
            if max_new_tokens > 0:
                cfg["max_new_tokens"] = max_new_tokens
            if no_repeat_ngram_size >= 0:
                cfg["no_repeat_ngram_size"] = no_repeat_ngram_size
