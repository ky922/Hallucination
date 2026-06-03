"""POPE baseline and method presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple


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


BASELINES: Dict[str, dict] = {
    "greedy_logits": {
        "description": "Greedy + logits-based yes/no decision",
        "use_logits": True,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 8,
    },
    "beam_search_logits": {
        "description": "Beam search (n=4) + logits yes/no",
        "use_logits": True,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 4,
        "length_penalty": 0.9,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "system_prompt": None,
        "max_new_tokens": 16,
    },
    "sampling_low_temp": {
        "description": "Sampling (t=0.3, top_p=0.9) low-temperature",
        "use_logits": False,
        "stochastic": True,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "num_beams": 1,
        "system_prompt": None,
        "max_new_tokens": 16,
    },
    "prompt_careful_logits": {
        "description": "Careful grounding prompt + logits yes/no",
        "use_logits": True,
        "stochastic": False,
        "do_sample": False,
        "num_beams": 1,
        "system_prompt": (
            "Please answer based only on what is clearly visible in the image. "
            "Do not guess or assume the presence of any object."
        ),
        "max_new_tokens": 8,
    },
}


QCVR_BASELINES: Dict[str, dict] = {
    "qcvr_only": {
        "description": "QCVR only (attention re-weighting, no IACD)",
        "use_qcvr": True,
        "use_iacd": False,
        "lm_layer": 16,
        "tau": 0.1,
        "lambda_": 1.0,
        "stochastic": False,
    },
    "iacd_only": {
        "description": "IACD only (Inert-Anchored contrastive decoding, no QCVR)",
        "use_qcvr": False,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.1,
        "lambda_": 1.0,
        "stochastic": False,
    },
    "qcvr_iacd": {
        "description": "QCVR + IACD (full method)",
        "use_qcvr": True,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.1,
        "lambda_": 1.0,
        "stochastic": False,
    },
    "qcvr_iacd_tau02": {
        "description": "QCVR + IACD (tau=0.2, ablation)",
        "use_qcvr": True,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.2,
        "lambda_": 1.0,
        "stochastic": False,
    },
    "qcvr_iacd_lambda05": {
        "description": "QCVR + IACD (lambda=0.5, ablation)",
        "use_qcvr": True,
        "use_iacd": True,
        "lm_layer": 16,
        "tau": 0.1,
        "lambda_": 0.5,
        "stochastic": False,
    },
}


SOTA_BASELINES: Dict[str, dict] = {
    "vcd": {
        "description": "VCD (noise_std=0.1, alpha=1.0)",
        "model_class": "VCDWrapper",
        "init_kwargs": {"noise_std": 0.1, "alpha": 1.0, "beta_apc": 0.1},
        "stochastic": False,
    },
    "vcd_tuned": {
        "description": "VCD tuned (APC, alpha=0.2, noise_std=0.05, noise_steps=3)",
        "model_class": "VCDWrapper",
        "init_kwargs": {
            "noise_std": 0.05,
            "alpha": 0.2,
            "beta_apc": 0.05,
            "noise_steps": 3,
        },
        "stochastic": False,
    },
    "icd": {
        "description": "ICD (alpha=1.0)",
        "model_class": "ICDWrapper",
        "init_kwargs": {"alpha": 1.0, "beta_apc": 0.1},
        "stochastic": False,
    },
}


DEFAULT_BASELINES = {
    k: BASELINES[k]
    for k in (
        "greedy_logits",
        "beam_search_logits",
        "sampling_low_temp",
        "prompt_careful_logits",
    )
}
DEFAULT_SOTA = {k: SOTA_BASELINES[k] for k in ("vcd", "icd")}
DEFAULT_QCVR = {"qcvr_iacd": QCVR_BASELINES["qcvr_iacd"]}


def baseline_choices() -> list[str]:
    return (
        list(BASELINES)
        + list(QCVR_BASELINES)
        + list(SOTA_BASELINES)
        + ["all", "required", "qcvr_all", "sota_all"]
    )


def _copy(configs: Dict[str, dict]) -> Dict[str, dict]:
    return deepcopy(configs)


def select_baseline_groups(args) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """Return standard, SOTA, and QCVR baseline groups for the parsed args."""
    if args.slim or args.key_only or args.baseline in ("all", "required"):
        baselines = _copy(DEFAULT_BASELINES)
        if args.no_stochastic:
            baselines = {
                k: v for k, v in baselines.items()
                if not v.get("stochastic", False)
            }
        qcvr_baselines = {} if args.no_qcvr_baselines else _copy(DEFAULT_QCVR)
        if args.no_sota_baselines:
            sota_baselines = {}
        else:
            sota_baselines = {
                k: v for k, v in _copy(DEFAULT_SOTA).items()
                if not (args.no_vcd and k == "vcd")
            }
    elif args.baseline == "qcvr_all":
        baselines = {}
        qcvr_baselines = _copy(QCVR_BASELINES)
        sota_baselines = {}
    elif args.baseline == "sota_all":
        baselines = {}
        qcvr_baselines = {}
        sota_baselines = _copy(SOTA_BASELINES)
    elif args.baseline in QCVR_BASELINES:
        baselines = {}
        qcvr_baselines = {args.baseline: deepcopy(QCVR_BASELINES[args.baseline])}
        sota_baselines = {}
    elif args.baseline in SOTA_BASELINES:
        baselines = {}
        qcvr_baselines = {}
        sota_baselines = {args.baseline: deepcopy(SOTA_BASELINES[args.baseline])}
    else:
        baselines = {args.baseline: deepcopy(BASELINES[args.baseline])}
        qcvr_baselines = {}
        sota_baselines = {}

    return baselines, sota_baselines, qcvr_baselines
