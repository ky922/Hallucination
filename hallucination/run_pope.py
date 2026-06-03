#!/usr/bin/env python3
"""
Run POPE baselines on LLaVA-1.5-7B for reproducible benchmark evaluation.

Key improvements over the initial version:
  1. Logits-based yes/no decision (generate_yes_no_logits) eliminates text-parse bias
  2. Beam search has length_penalty / no_repeat_ngram_size / early_stopping
  3. Two sampling groups (low-temp / high-temp) for sensitivity analysis
  4. --n_runs: repeat stochastic baselines N times, report mean ± std
  5. Fixed seeds for reproducibility; full config saved alongside results
  6. Timestamped output directory for experiment versioning

Usage examples:
  # Full evaluation (all splits, all baselines, 3 runs for stochastic)
  python run_pope.py

  # Quick smoke-test — 50 samples, adversarial split only
  python run_pope.py --max_samples 50 --split adversarial

  # Single baseline, single split
  python run_pope.py --baseline greedy_logits --split popular

  # Stochastic baselines repeated 5 times
  python run_pope.py --n_runs 5 --split adversarial
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.llava_wrapper import LLaVAWrapper
from models.qcvr_wrapper import QCVRWrapper
from models.sota_wrappers import VCDWrapper, ICDWrapper
from eval.pope_eval import load_pope, parse_yes_no, compute_pope_metrics


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Baseline configurations ────────────────────────────────────────────────────
# "use_logits": True  → use generate_yes_no_logits() (no text parsing bias)
# "stochastic":  True → repeat n_runs times and report mean ± std
# All other keys are forwarded to generate() / generate_yes_no_logits()

BASELINES: Dict[str, dict] = {
    # ── Deterministic baselines ────────────────────────────────────────────────
    "greedy_logits": {
        "description":    "Greedy + logits-based yes/no decision",
        "use_logits":     True,
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 8,
    },
    "beam_search_logits": {
        "description":    "Beam search (n=4) + logits yes/no",
        "use_logits":     True,
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      4,
        "length_penalty": 0.9,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "system_prompt":  None,
        "max_new_tokens": 16,
    },
    # ── Stochastic baselines (will be run n_runs times) ────────────────────────
    "sampling_low_temp": {
        "description":    "Sampling (t=0.3, top_p=0.9) — low-temperature",
        "use_logits":     False,
        "stochastic":     True,
        "do_sample":      True,
        "temperature":    0.3,
        "top_p":          0.9,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 16,
    },
    # ── Prompt engineering ─────────────────────────────────────────────────────
    "prompt_careful_logits": {
        "description":    "Careful grounding prompt + logits yes/no",
        "use_logits":     True,
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  (
            "Please answer based only on what is clearly visible in the image. "
            "Do not guess or assume the presence of any object."
        ),
        "max_new_tokens": 8,
    },
}

# ── QCVR+IACD method configurations ───────────────────────────────────────────
# These use QCVRWrapper instead of LLaVAWrapper.
# "use_qcvr" / "use_iacd" control which components are active (for ablation).

QCVR_BASELINES: Dict[str, dict] = {
    "qcvr_only": {
        "description":    "QCVR only (attention re-weighting, no IACD)",
        "use_qcvr":       True,
        "use_iacd":       False,
        "lm_layer":       16,
        "tau":            0.1,
        "lambda_":        1.0,
        "stochastic":     False,
    },
    "iacd_only": {
        "description":    "IACD only (Inert-Anchored contrastive decoding, no QCVR)",
        "use_qcvr":       False,
        "use_iacd":       True,
        "lm_layer":       16,
        "tau":            0.1,
        "lambda_":        1.0,
        "stochastic":     False,
    },
    "qcvr_iacd": {
        "description":    "QCVR + IACD (full method)",
        "use_qcvr":       True,
        "use_iacd":       True,
        "lm_layer":       16,
        "tau":            0.1,
        "lambda_":        1.0,
        "stochastic":     False,
    },
    "qcvr_iacd_tau02": {
        "description":    "QCVR + IACD (tau=0.2, ablation)",
        "use_qcvr":       True,
        "use_iacd":       True,
        "lm_layer":       16,
        "tau":            0.2,
        "lambda_":        1.0,
        "stochastic":     False,
    },
    "qcvr_iacd_lambda05": {
        "description":    "QCVR + IACD (lambda=0.5, ablation)",
        "use_qcvr":       True,
        "use_iacd":       True,
        "lm_layer":       16,
        "tau":            0.1,
        "lambda_":        0.5,
        "stochastic":     False,
    },
}

# Key QCVR subset for explicit QCVR runs.
QCVR_CORE = {k: QCVR_BASELINES[k] for k in ("qcvr_only", "iacd_only", "qcvr_iacd")}

# ── SOTA comparison baselines ──────────────────────────────────────────────────
# These use VCDWrapper / ICDWrapper from models/sota_wrappers.py.
# "model_class": the class to instantiate, "init_kwargs": forwarded to __init__.

SOTA_BASELINES: Dict[str, dict] = {
    # VCD: Visual Contrastive Decoding (Leng et al., CVPR 2024)
    # https://arxiv.org/abs/2311.16922
    "vcd": {
        "description":  "VCD (noise_std=0.1, alpha=1.0) — CVPR 2024",
        "model_class":  "VCDWrapper",
        "init_kwargs":  {"noise_std": 0.1, "alpha": 1.0, "beta_apc": 0.1},
        "stochastic":   False,
    },
    "vcd_tuned": {
        "description":  "VCD tuned (APC, alpha=0.2, noise_std=0.05, noise_steps=3)",
        "model_class":  "VCDWrapper",
        "init_kwargs":  {"noise_std": 0.05, "alpha": 0.2, "beta_apc": 0.05, "noise_steps": 3},
        "stochastic":   False,
    },
    # ICD: Instruction Contrastive Decoding (Leng et al., ACL Findings 2024)
    # https://arxiv.org/abs/2403.18715
    "icd": {
        "description":  "ICD (alpha=1.0) — ACL Findings 2024",
        "model_class":  "ICDWrapper",
        "init_kwargs":  {"alpha": 1.0, "beta_apc": 0.1},
        "stochastic":   False,
    },
}

# ── Required baseline preset ──────────────────────────────────────────────────
# Standard inference + decoding strategy changes + prompt rewrite + 2 SOTA
# + our full method. Kept as the default `all` set for one global run.
DEFAULT_BASELINES = {k: BASELINES[k] for k in (
    "greedy_logits", "beam_search_logits",
    "sampling_low_temp",
    "prompt_careful_logits",
)}
DEFAULT_SOTA = {k: SOTA_BASELINES[k] for k in ("vcd", "icd")}
DEFAULT_QCVR = {"qcvr_iacd": QCVR_BASELINES["qcvr_iacd"]}

# Backward-compatible aliases for older launch scripts.
KEY_BASELINES = DEFAULT_BASELINES
KEY_SOTA      = DEFAULT_SOTA
KEY_QCVR      = DEFAULT_QCVR
SLIM_BASELINES = DEFAULT_BASELINES
SLIM_SOTA      = DEFAULT_SOTA
SLIM_QCVR      = DEFAULT_QCVR

# kwargs that go directly to generate() / generate_yes_no_logits()
_GEN_KEYS = {"do_sample", "temperature", "top_p", "num_beams",
             "length_penalty", "no_repeat_ngram_size", "early_stopping",
             "system_prompt", "max_new_tokens"}


# ── Single-run evaluation ──────────────────────────────────────────────────────

def run_one(
    model: LLaVAWrapper,
    data: List[dict],
    image_dir: str,
    config: dict,
    max_samples: int = -1,
    seed: int = 42,
    batch_size: int = 8,
) -> dict:
    """One evaluation pass. Returns compute_pope_metrics output."""
    set_seed(seed)
    gen_kwargs = {k: v for k, v in config.items() if k in _GEN_KEYS}
    use_logits = config.get("use_logits", False)
    preds, labels = [], []

    subset = data if max_samples <= 0 else data[:max_samples]

    # ── Batched logits path (LLaVAWrapper only) ────────────────────────────────
    if use_logits and hasattr(model, "generate_yes_no_logits_batch"):
        images    = [os.path.join(image_dir, item["image"]) for item in subset]
        questions = [item["text"] for item in subset]
        sp        = gen_kwargs.get("system_prompt")
        sys_prompts = [sp] * len(subset)

        results = model.generate_yes_no_logits_batch(
            images, questions, sys_prompts, batch_size=batch_size
        )
        preds  = [r[0] for r in results]
        labels = [item["label"] for item in subset]

    # ── Per-sample path (text gen / SOTA / QCVR wrappers) ─────────────────────
    else:
        # Prefetch images with a thread pool so disk I/O overlaps with GPU compute
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image as _PIL

        def _load(path):
            return _PIL.open(path).convert("RGB")

        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = [os.path.join(image_dir, item["image"]) for item in subset]
            # Submit all loads upfront; iterator yields in order
            futures = [pool.submit(_load, p) for p in paths]

            for item, fut in tqdm(
                zip(subset, futures),
                total=len(subset),
                desc=config["description"][:45],
                leave=False,
            ):
                img = fut.result()
                if use_logits:
                    pred, _, _ = model.generate_yes_no_logits(
                        img, item["text"],
                        system_prompt=gen_kwargs.get("system_prompt"),
                    )
                else:
                    response = model.generate(img, item["text"], **gen_kwargs)
                    pred = parse_yes_no(response)
                preds.append(pred)
                labels.append(item["label"])

    return compute_pope_metrics(preds, labels)


def run_baseline(
    model: LLaVAWrapper,
    data: List[dict],
    image_dir: str,
    config: dict,
    max_samples: int = -1,
    n_runs: int = 1,
    base_seed: int = 42,
) -> Tuple[dict, Optional[dict]]:
    """
    Run one baseline, repeat n_runs times if stochastic.

    Returns:
        (aggregated_metrics, std_metrics_or_None)
        For deterministic baselines n_runs is forced to 1.
    """
    is_stochastic = config.get("stochastic", False)
    actual_runs   = n_runs if is_stochastic else 1

    all_runs = []
    for i in range(actual_runs):
        seed = base_seed + i
        m = run_one(model, data, image_dir, config, max_samples, seed=seed)
        all_runs.append(m)

    # Aggregate
    scalar_keys = ["accuracy", "f1", "precision", "recall", "yes_ratio"]
    agg: dict = {}
    std_out: Optional[dict] = None

    if actual_runs == 1:
        agg = all_runs[0]
    else:
        for k in scalar_keys:
            vals = [r[k] for r in all_runs]
            agg[k] = round(mean(vals), 2)
        # Keep raw counts from last run only (counts per run differ by seed)
        for k in ("tp", "fp", "tn", "fn"):
            agg[k] = all_runs[-1][k]
        std_out = {k: round(stdev([r[k] for r in all_runs]), 2)
                   for k in scalar_keys}

    agg["n_runs"] = actual_runs
    return agg, std_out


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="POPE hallucination evaluation")
    parser.add_argument("--data_dir",    default="data/pope")
    parser.add_argument("--image_dir",   default="data/coco/val2014")
    parser.add_argument("--split",       default="all",
                        choices=["random", "popular", "adversarial", "all"])
    parser.add_argument("--baseline",    default="all",
                        choices=(list(BASELINES) + list(QCVR_BASELINES) +
                                 list(SOTA_BASELINES) +
                                 ["all", "required", "qcvr_all", "sota_all"]))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--n_runs",      type=int, default=1,
                        help="Repeat stochastic baselines N times (default 1)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  default="results/pope")
    parser.add_argument("--model_id",    default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--no_qcvr_baselines", action="store_true",
                        help="Skip QCVR baselines when --baseline all")
    parser.add_argument("--no_sota_baselines", action="store_true",
                        help="Skip VCD/ICD SOTA baselines when --baseline all")
    parser.add_argument("--no_stochastic", action="store_true",
                        help="Skip sampling (stochastic) baselines")
    parser.add_argument("--key_only", action="store_true",
                        help="Alias for --baseline required")
    parser.add_argument("--no_vcd", action="store_true",
                        help="Skip VCD when using --key_only or --slim")
    parser.add_argument("--slim", action="store_true",
                        help="Alias for --baseline required")
    args = parser.parse_args()

    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(args.output_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # Save experiment config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    splits    = (["random", "popular", "adversarial"]
                 if args.split == "all" else [args.split])

    # Determine which baseline sets to run
    if args.slim or args.key_only or args.baseline in ("all", "required"):
        raw_baselines  = DEFAULT_BASELINES
        if args.no_stochastic:
            raw_baselines = {k: v for k, v in raw_baselines.items()
                             if not v.get("stochastic", False)}
        baselines      = raw_baselines
        qcvr_baselines = {} if args.no_qcvr_baselines else DEFAULT_QCVR
        if args.no_sota_baselines:
            sota_baselines = {}
        else:
            sota_baselines = {k: v for k, v in DEFAULT_SOTA.items()
                              if not (args.no_vcd and k == "vcd")}
    elif args.baseline == "qcvr_all":
        baselines      = {}
        qcvr_baselines = QCVR_BASELINES
        sota_baselines = {}
    elif args.baseline == "sota_all":
        baselines      = {}
        qcvr_baselines = {}
        sota_baselines = SOTA_BASELINES
    elif args.baseline in QCVR_BASELINES:
        baselines      = {}
        qcvr_baselines = {args.baseline: QCVR_BASELINES[args.baseline]}
        sota_baselines = {}
    elif args.baseline in SOTA_BASELINES:
        baselines      = {}
        qcvr_baselines = {}
        sota_baselines = {args.baseline: SOTA_BASELINES[args.baseline]}
    else:
        baselines      = {args.baseline: BASELINES[args.baseline]}
        qcvr_baselines = {}
        sota_baselines = {}

    # Model caches (populated on demand, freed between phases)
    qcvr_model_cache: Dict[str, QCVRWrapper] = {}

    def get_qcvr_model(cfg: dict) -> QCVRWrapper:
        key = f"{cfg['use_qcvr']}_{cfg['use_iacd']}_{cfg['lm_layer']}_{cfg['tau']}_{cfg['lambda_']}"
        if key not in qcvr_model_cache:
            qcvr_model_cache[key] = QCVRWrapper(
                model_id=args.model_id,
                use_qcvr=cfg["use_qcvr"],
                use_iacd=cfg["use_iacd"],
                lm_layer=cfg["lm_layer"],
                tau=cfg["tau"],
                lambda_=cfg["lambda_"],
            )
        return qcvr_model_cache[key]

    _sota_classes = {"VCDWrapper": VCDWrapper, "ICDWrapper": ICDWrapper}
    sota_model_cache: Dict[str, object] = {}

    # VCD and ICD share the same base model as LLaVAWrapper (no extra load needed)
    # _raw_model / _raw_proc hold references into the LLaVAWrapper for reuse
    _raw_model: object = None
    _raw_proc:  object = None

    def get_sota_model(cfg: dict):
        key = cfg["description"]
        if key not in sota_model_cache:
            cls = _sota_classes[cfg["model_class"]]
            sota_model_cache[key] = cls(
                model_id=args.model_id,
                _model=_raw_model,
                _processor=_raw_proc,
                **cfg["init_kwargs"],
            )
        return sota_model_cache[key]

    all_results: dict = {split: {} for split in splits}
    t_start = time.time()

    # ── Phase 1 + 2 share one LLaVAWrapper (VCD/ICD reuse its weights) ──────
    if baselines or sota_baselines:
        model = LLaVAWrapper(model_id=args.model_id)
        _raw_model = model.model
        _raw_proc  = model.processor
    else:
        model = None

    # ── Phase 1: Standard baselines (greedy / beam / prompt / sampling) ──────
    if baselines:
        for split in splits:
            print(f"\n{'='*65}\nPOPE split: {split.upper()}\n{'='*65}")
            data = load_pope(args.data_dir, split)
            for name, cfg in baselines.items():
                agg, std = run_baseline(model, data, args.image_dir, cfg,
                                        max_samples=args.max_samples,
                                        n_runs=args.n_runs,
                                        base_seed=args.seed)
                entry = {"description": cfg["description"], "metrics": agg}
                if std:
                    entry["std"] = std
                all_results[split][name] = entry
                std_str = (f"  ±{std['accuracy']:.2f}" if std else "")
                print(f"  {name:<26}  "
                      f"Acc={agg['accuracy']:.2f}%{std_str}  "
                      f"F1={agg['f1']:.2f}%  "
                      f"Yes%={agg['yes_ratio']:.2f}%  "
                      f"(n={agg['n_runs']})")

    # ── Phase 2: SOTA baselines (VCDWrapper / ICDWrapper reuse Phase-1 model) ─
    if sota_baselines:
        print("\n--- SOTA comparison methods (VCD / ICD) ---")
    for name, cfg in sota_baselines.items():
        sm = get_sota_model(cfg)   # shares model.model — no extra GPU memory
        sota_cfg = {
            "description":   cfg["description"],
            "use_logits":    True,
            "stochastic":    False,
            "do_sample":     False,
            "num_beams":     1,
            "system_prompt": None,
            "max_new_tokens": 8,
        }
        for split in splits:
            data = load_pope(args.data_dir, split)
            print(f"\n  [{name}] POPE split: {split.upper()}")
            agg, std = run_baseline(sm, data, args.image_dir, sota_cfg,
                                    max_samples=args.max_samples,
                                    n_runs=args.n_runs,
                                    base_seed=args.seed)
            entry = {"description": cfg["description"], "metrics": agg}
            if std:
                entry["std"] = std
            all_results[split][name] = entry
            std_str = (f"  ±{std['accuracy']:.2f}" if std else "")
            print(f"  {name:<26}  "
                  f"Acc={agg['accuracy']:.2f}%{std_str}  "
                  f"F1={agg['f1']:.2f}%  "
                  f"Yes%={agg['yes_ratio']:.2f}%  "
                  f"(n={agg['n_runs']})")
        sota_model_cache.pop(cfg["description"], None)
        del sm

    # Unload shared model after Phase 1+2, before QCVR which needs eager attn
    if model is not None and qcvr_baselines:
        print("\n[INFO] Unloading base model to free GPU memory for QCVR...")
        import gc
        sota_model_cache.clear()
        _raw_model = None
        _raw_proc  = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Phase 3: QCVR/IACD baselines ───────────────────────────────────────
    if qcvr_baselines:
        print("\n--- QCVR/IACD methods ---")
    for name, cfg in qcvr_baselines.items():
        qm = get_qcvr_model(cfg)
        qcvr_cfg = {
            "description":   cfg["description"],
            "use_logits":    True,
            "stochastic":    False,
            "do_sample":     False,
            "num_beams":     1,
            "system_prompt": None,
            "max_new_tokens": 8,
        }
        for split in splits:
            data = load_pope(args.data_dir, split)
            print(f"\n  [{name}] POPE split: {split.upper()}")
            agg, std = run_baseline(qm, data, args.image_dir, qcvr_cfg,
                                    max_samples=args.max_samples,
                                    n_runs=args.n_runs,
                                    base_seed=args.seed)
            entry = {"description": cfg["description"], "metrics": agg}
            if std:
                entry["std"] = std
            all_results[split][name] = entry
            std_str = (f"  ±{std['accuracy']:.2f}" if std else "")
            print(f"  {name:<26}  "
                  f"Acc={agg['accuracy']:.2f}%{std_str}  "
                  f"F1={agg['f1']:.2f}%  "
                  f"Yes%={agg['yes_ratio']:.2f}%  "
                  f"(n={agg['n_runs']})")
        # Unload this QCVR model before loading the next one
        key = f"{cfg['use_qcvr']}_{cfg['use_iacd']}_{cfg['lm_layer']}_{cfg['tau']}_{cfg['lambda_']}"
        qcvr_model_cache.pop(key, None)
        import gc
        del qm
        gc.collect()
        torch.cuda.empty_cache()

    # Save per-split JSON results
    for split in splits:
        out_path = os.path.join(out_dir, f"pope_{split}.json")
        with open(out_path, "w") as f:
            json.dump(all_results[split], f, indent=2)
        print(f"  → {out_path}")

    elapsed = time.time() - t_start

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY — Accuracy (mean %) | elapsed {elapsed/60:.1f} min")
    print(f"{'='*80}")
    col_w = 12
    all_baseline_names = list(baselines) + list(sota_baselines) + list(qcvr_baselines)
    header = f"{'Baseline':<28}" + "".join(f"{s:>{col_w}}" for s in splits)
    print(header)
    print("-" * len(header))
    for name in all_baseline_names:
        row = f"{name:<28}"
        for s in splits:
            entry = all_results.get(s, {}).get(name, {})
            acc   = entry.get("metrics", {}).get("accuracy", float("nan"))
            std_v = entry.get("std",     {}).get("accuracy") if "std" in entry else None
            cell  = f"{acc:.2f}" + (f"±{std_v:.2f}" if std_v is not None else "")
            row  += f"{cell:>{col_w}}"
        print(row)

    out_path = os.path.join(out_dir, "pope_all.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
