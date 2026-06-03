#!/usr/bin/env python3
"""
Run CHAIR baselines on LLaVA-1.5-7B for reproducible benchmark evaluation.

Key improvements:
  1. Beam search with length_penalty / no_repeat_ngram_size / early_stopping
  2. Two sampling groups (low-temp t=0.3 / high-temp t=0.8)
  3. Fixed random seed + reproducible image sampling
  4. Timestamped output directory; full config saved per run
  5. --n_runs: repeat stochastic baselines, report mean ± std

Usage examples:
  # Full evaluation (500 images, 3 runs for stochastic)
  python run_chair.py

  # Quick test — 50 images
  python run_chair.py --num_samples 50

  # Single baseline
  python run_chair.py --baseline beam_search --num_samples 200
"""

import argparse
import gc
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
from models.sota_wrappers import VCDWrapper, ICDWrapper
from models.qcvr_wrapper import QCVRWrapper
from eval.chair_eval import load_coco_annotations, compute_chair


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Baseline configurations ────────────────────────────────────────────────────

BASELINES: Dict[str, dict] = {
    # ── Deterministic ─────────────────────────────────────────────────────────
    "greedy": {
        "description":    "Standard greedy decoding",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 64,
    },
    "beam_search": {
        "description":    "Beam search (n=4, length_penalty=1.0, no_repeat=3)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      4,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "system_prompt":  None,
        "max_new_tokens": 64,
    },
    # ── Stochastic (repeated n_runs times) ────────────────────────────────────
    "sampling_low_temp": {
        "description":    "Sampling (t=0.3, top_p=0.9) — low-temperature",
        "question":       "Please describe this image in detail.",
        "stochastic":     True,
        "do_sample":      True,
        "temperature":    0.3,
        "top_p":          0.9,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 64,
    },
    # ── Prompt engineering ─────────────────────────────────────────────────────
    "prompt_careful": {
        "description": "Careful grounding prompt + greedy",
        "question":    "Please describe this image in detail.",
        "stochastic":  False,
        "do_sample":   False,
        "num_beams":   1,
        "system_prompt": (
            "Describe only what you can clearly see in the image. "
            "Do not mention objects that are not visibly present."
        ),
        "max_new_tokens": 64,
    },
    "greedy_short_norepeat": {
        "description":    "Greedy decoding control (48 tokens, no-repeat=3)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
    },
}

_GEN_KEYS = {"do_sample", "temperature", "top_p", "num_beams",
             "length_penalty", "no_repeat_ngram_size", "early_stopping",
             "system_prompt", "max_new_tokens"}


# ── SOTA baseline configurations ───────────────────────────────────────────────

SOTA_BASELINES: Dict[str, dict] = {
    "vcd": {
        "description":    "VCD greedy (visual contrastive decoding, alpha=1.0)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 64,
        "model_class":    "VCDWrapper",
        "init_kwargs":    {"noise_std": 0.1, "alpha": 1.0, "beta_apc": 0.1},
    },
    "vcd_tuned": {
        "description":    "VCD tuned (APC, alpha=0.2, short no-repeat decoding)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "model_class":    "VCDWrapper",
        "init_kwargs":    {"noise_std": 0.05, "alpha": 0.2, "beta_apc": 0.05, "noise_steps": 3},
    },
    "icd": {
        "description":    "ICD greedy (dual-stream contrastive, alpha=1.0)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 64,
        "model_class":    "ICDWrapper",
        "init_kwargs":    {"alpha": 1.0, "beta_apc": 0.1},
    },
}

# Backward-compatible name for older code.
ICD_BASELINES = {"icd": SOTA_BASELINES["icd"]}


# ── QCVR+IACD baseline configurations ─────────────────────────────────────────
# Uses QCVRWrapper.generate() — IACD applied at every token via LogitsProcessor

QCVR_CHAIR_BASELINES: Dict[str, dict] = {
    "qcvr_iacd": {
        "description":    "QCVR + IACD (first-token, tuned for CHAIR)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        # QCVRWrapper init kwargs
        "use_qcvr":  True,
        "use_iacd":  True,
        "lm_layer":  16,
        "tau":       0.2,
        "lambda_":   0.3,
        "iacd_mode": "first_token",
        "iacd_decay_steps": 8,
    },
    "qcvr_only": {
        "description":    "QCVR only (zero-IACD control, CHAIR ablation)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "use_qcvr":  True,
        # Keep a zero-weight IACD processor so QCVR-only follows the same
        # generation code path as the tuned full method while making no IACD
        # logit change.
        "use_iacd":  True,
        "lm_layer":  16,
        "tau":       0.2,
        "lambda_":   0.0,
        "iacd_mode": "first_token",
        "iacd_decay_steps": 8,
    },
    "iacd_first_token": {
        "description":    "IACD first-token only (no QCVR, CHAIR ablation)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "use_qcvr":  False,
        "use_iacd":  True,
        "lm_layer":  16,
        "tau":       0.2,
        "lambda_":   0.3,
        "iacd_mode": "first_token",
        "iacd_decay_steps": 8,
    },
    "qcvr_iacd_decay": {
        "description":    "QCVR + IACD decay (lambda=0.3, tau=0.2)",
        "question":       "Please describe this image in detail.",
        "stochastic":     False,
        "do_sample":      False,
        "num_beams":      1,
        "system_prompt":  None,
        "max_new_tokens": 48,
        "no_repeat_ngram_size": 3,
        "use_qcvr":  True,
        "use_iacd":  True,
        "lm_layer":  16,
        "tau":       0.2,
        "lambda_":   0.3,
        "iacd_mode": "decay",
        "iacd_decay_steps": 8,
    },
}


# ── Required baseline preset ───────────────────────────────────────────────────
# Standard inference + decoding strategy changes + prompt rewrite + 2 SOTA
# + our full method.

DEFAULT_BASELINES = {k: BASELINES[k] for k in (
    "greedy", "beam_search", "sampling_low_temp", "prompt_careful",
)}
DEFAULT_SOTA = {k: SOTA_BASELINES[k] for k in ("vcd", "icd")}
DEFAULT_QCVR = {"qcvr_iacd": QCVR_CHAIR_BASELINES["qcvr_iacd"]}

# Backward-compatible aliases for older launch scripts.
KEY_BASELINES = DEFAULT_BASELINES
KEY_ICD       = DEFAULT_SOTA
KEY_QCVR      = DEFAULT_QCVR


# ── Helpers ────────────────────────────────────────────────────────────────────

def coco_image_filename(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def sample_image_ids(
    gt_annotations: dict,
    num_samples: int,
    seed: int,
    image_dir: str,
) -> List[int]:
    all_ids = sorted(gt_annotations.keys())
    random.seed(seed)
    random.shuffle(all_ids)

    selected = []
    for img_id in all_ids:
        path = os.path.join(image_dir, coco_image_filename(img_id))
        if os.path.exists(path):
            selected.append(img_id)
            if len(selected) >= num_samples:
                break

    if len(selected) < num_samples:
        print(f"[warn] Only found {len(selected)} valid images "
              f"(requested {num_samples}).")
    return selected


# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_one(
    model: LLaVAWrapper,
    image_ids: List[int],
    image_dir: str,
    gt_annotations: dict,
    config: dict,
    seed: int = 42,
) -> Tuple[dict, List[dict]]:
    set_seed(seed)
    question   = config.get("question", "Please describe this image in detail.")
    gen_kwargs = {k: v for k, v in config.items() if k in _GEN_KEYS}

    captions, valid_ids, records = [], [], []
    for img_id in tqdm(image_ids, desc=config["description"][:45], leave=False):
        img_path = os.path.join(image_dir, coco_image_filename(img_id))
        caption  = model.generate(img_path, question, **gen_kwargs)
        captions.append(caption)
        valid_ids.append(img_id)
        records.append({"image_id": img_id, "caption": caption})

    metrics = compute_chair(captions, valid_ids, gt_annotations)
    return metrics, records


def run_baseline(
    model: LLaVAWrapper,
    image_ids: List[int],
    image_dir: str,
    gt_annotations: dict,
    config: dict,
    n_runs: int = 1,
    base_seed: int = 42,
) -> Tuple[dict, Optional[dict], List[dict]]:
    """
    Run one baseline, repeat n_runs times for stochastic variants.

    Returns:
        (aggregated_metrics, std_or_None, captions_from_last_run)
    """
    is_stochastic = config.get("stochastic", False)
    actual_runs   = n_runs if is_stochastic else 1

    all_metrics, last_records = [], []
    for i in range(actual_runs):
        m, records = run_one(model, image_ids, image_dir, gt_annotations,
                             config, seed=base_seed + i)
        all_metrics.append(m)
        last_records = records

    scalar_keys = ["CHAIRs", "CHAIRi"]
    agg: dict = {}
    std_out: Optional[dict] = None

    if actual_runs == 1:
        agg = all_metrics[0]
    else:
        for k in scalar_keys:
            vals = [r[k] for r in all_metrics]
            agg[k] = round(mean(vals), 2)
        # carry non-scalar fields from last run
        for k in all_metrics[-1]:
            if k not in agg:
                agg[k] = all_metrics[-1][k]
        std_out = {k: round(stdev([r[k] for r in all_metrics]), 2)
                   for k in scalar_keys}

    agg["n_runs"] = actual_runs
    return agg, std_out, last_records


# ── Main ───────────────────────────────────────────────────────────────────────

def _save_one(out_dir, name, cfg, agg, std, records):
    out = {
        "config":   {k: v for k, v in cfg.items() if k != "description"},
        "metrics":  agg,
        "std":      std,
        "captions": records,
    }
    out_path = os.path.join(out_dir, f"chair_{name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {out_path}")


def _print_row(name, agg, std):
    std_str = (f"  ±{std['CHAIRs']:.2f}/{std['CHAIRi']:.2f}" if std else "")
    print(f"  {name:<26}  CHAIRs={agg['CHAIRs']:.2f}%  "
          f"CHAIRi={agg['CHAIRi']:.2f}%{std_str}  (n={agg['n_runs']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="CHAIR hallucination evaluation")
    parser.add_argument("--image_dir",       default="data/coco/val2014")
    parser.add_argument("--annotation_file",
                        default="data/coco/annotations/instances_val2014.json")
    parser.add_argument("--num_samples",  type=int, default=500)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--baseline",     default="all",
                        choices=(list(BASELINES) + list(SOTA_BASELINES) +
                                 list(QCVR_CHAIR_BASELINES) +
                                 ["all", "required", "sota_all", "qcvr_all",
                                  "full"]))
    parser.add_argument("--n_runs",       type=int, default=1,
                        help="Repeat stochastic baselines N times (default 1)")
    parser.add_argument("--output_dir",   default="results/chair")
    parser.add_argument("--model_id",     default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--key_only",     action="store_true",
                        help="Alias for --baseline required")
    parser.add_argument("--no_icd",       action="store_true",
                        help="Skip ICD baselines")
    parser.add_argument("--no_sota",      action="store_true",
                        help="Skip VCD/ICD SOTA baselines")
    parser.add_argument("--no_vcd",       action="store_true",
                        help="Skip VCD SOTA baseline")
    parser.add_argument("--no_qcvr",      action="store_true",
                        help="Skip QCVR+IACD baselines")
    parser.add_argument("--unify_tokens", type=int, default=0,
                        help="If >0, force max_new_tokens to this value for ALL "
                             "methods (removes decoding-length confound).")
    parser.add_argument("--unify_no_repeat", type=int, default=-1,
                        help="If >=0, force no_repeat_ngram_size to this value "
                             "for ALL methods (removes no-repeat confound).")
    args = parser.parse_args()

    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(args.output_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading COCO annotations...")
    gt_annotations = load_coco_annotations(args.annotation_file)

    print(f"Sampling {args.num_samples} images (seed={args.seed})...")
    image_ids = sample_image_ids(gt_annotations, args.num_samples,
                                 args.seed, args.image_dir)
    print(f"Using {len(image_ids)} images.")

    with open(os.path.join(out_dir, "image_ids.json"), "w") as f:
        json.dump(image_ids, f)

    # Determine which baseline sets to run
    if args.key_only or args.baseline in ("all", "required"):
        llava_baselines = KEY_BASELINES
        if args.no_sota:
            sota_baselines = {}
        else:
            sota_baselines = {
                k: v for k, v in DEFAULT_SOTA.items()
                if not ((args.no_icd and k == "icd") or (args.no_vcd and k == "vcd"))
            }
        qcvr_baselines  = {} if args.no_qcvr else DEFAULT_QCVR
    elif args.baseline == "full":
        # Complete main-table + ablation set, one process / one output dir, so a
        # single --unify_tokens/--unify_no_repeat run yields a fully fair table.
        llava_baselines = {k: BASELINES[k] for k in (
            "greedy", "beam_search", "sampling_low_temp", "prompt_careful")}
        sota_baselines = {
            k: v for k, v in SOTA_BASELINES.items()
            if not ((args.no_icd and k == "icd") or (args.no_vcd and k == "vcd"))
        }
        qcvr_baselines = {} if args.no_qcvr else {
            k: QCVR_CHAIR_BASELINES[k]
            for k in ("qcvr_iacd", "qcvr_only", "iacd_first_token")}
    elif args.baseline == "sota_all":
        llava_baselines = {}
        sota_baselines  = {
            k: v for k, v in SOTA_BASELINES.items()
            if not ((args.no_icd and k == "icd") or (args.no_vcd and k == "vcd"))
        }
        qcvr_baselines  = {}
    elif args.baseline == "qcvr_all":
        llava_baselines = {}
        sota_baselines  = {}
        qcvr_baselines  = {} if args.no_qcvr else QCVR_CHAIR_BASELINES
    elif args.baseline in SOTA_BASELINES:
        llava_baselines = {}
        sota_baselines  = {} if args.no_sota else {args.baseline: SOTA_BASELINES[args.baseline]}
        qcvr_baselines  = {}
    elif args.baseline in QCVR_CHAIR_BASELINES:
        llava_baselines = {}
        sota_baselines  = {}
        qcvr_baselines  = {} if args.no_qcvr else {args.baseline: QCVR_CHAIR_BASELINES[args.baseline]}
    else:
        llava_baselines = {args.baseline: BASELINES[args.baseline]}
        sota_baselines  = {}
        qcvr_baselines  = {}

    # ── Unified decoding override ─────────────────────────────────────────────
    # For a fair main-table comparison, every method must use the SAME generation
    # budget. Otherwise the headline CHAIR gain conflates the QCVR/IACD effect
    # with a shorter / no-repeat decoding config (see ablation "short greedy").
    if args.unify_tokens > 0 or args.unify_no_repeat >= 0:
        for _set in (llava_baselines, sota_baselines, qcvr_baselines):
            for _name, _cfg in _set.items():
                if args.unify_tokens > 0:
                    _cfg["max_new_tokens"] = args.unify_tokens
                if args.unify_no_repeat >= 0:
                    _cfg["no_repeat_ngram_size"] = args.unify_no_repeat
        print(f"[unify] All methods forced to "
              f"max_new_tokens={args.unify_tokens or 'unchanged'}, "
              f"no_repeat_ngram_size="
              f"{args.unify_no_repeat if args.unify_no_repeat >= 0 else 'unchanged'}")

    all_metrics: dict = {}
    t_start = time.time()

    # ── Phase 1: LLaVAWrapper baselines ───────────────────────────────────────
    if llava_baselines:
        print("\n" + "="*65)
        print("PHASE 1 — LLaVAWrapper baselines")
        print("="*65)
        model = LLaVAWrapper(model_id=args.model_id)
        _raw_model = model.model
        _raw_proc  = model.processor

        for name, cfg in llava_baselines.items():
            print(f"\n{'='*60}\n{cfg['description']}\n{'='*60}")
            agg, std, records = run_baseline(
                model, image_ids, args.image_dir, gt_annotations, cfg,
                n_runs=args.n_runs, base_seed=args.seed,
            )
            all_metrics[name] = {"metrics": agg, "std": std}
            _print_row(name, agg, std)
            _save_one(out_dir, name, cfg, agg, std, records)
    else:
        _raw_model, _raw_proc = None, None
        model = None

    # ── Phase 2: SOTA baselines (reuse LLaVAWrapper model weights) ────────────
    if sota_baselines:
        print("\n" + "="*65)
        print("PHASE 2 — SOTA baselines")
        print("="*65)
        if _raw_model is None:
            # Load model fresh if Phase 1 was skipped
            _tmp = LLaVAWrapper(model_id=args.model_id)
            _raw_model, _raw_proc = _tmp.model, _tmp.processor

        sota_classes = {"VCDWrapper": VCDWrapper, "ICDWrapper": ICDWrapper}
        for name, cfg in sota_baselines.items():
            print(f"\n{'='*60}\n{cfg['description']}\n{'='*60}")
            cls = sota_classes[cfg["model_class"]]
            sota_model = cls(
                model_id=args.model_id,
                _model=_raw_model,
                _processor=_raw_proc,
                **cfg["init_kwargs"],
            )
            gen_cfg = {
                k: v for k, v in cfg.items()
                if k not in {"model_class", "init_kwargs"}
            }
            agg, std, records = run_baseline(
                sota_model, image_ids, args.image_dir, gt_annotations, gen_cfg,
                n_runs=args.n_runs, base_seed=args.seed,
            )
            all_metrics[name] = {"metrics": agg, "std": std}
            _print_row(name, agg, std)
            _save_one(out_dir, name, cfg, agg, std, records)

    # ── Phase 3: QCVR+IACD baselines (requires its own model load) ────────────
    if qcvr_baselines:
        print("\n" + "="*65)
        print("PHASE 3 — QCVR+IACD baselines (reloading model)")
        print("="*65)
        # Free Phase 1/2 model to reclaim GPU memory
        if model is not None:
            del model
        if _raw_model is not None:
            del _raw_model, _raw_proc
        gc.collect()
        torch.cuda.empty_cache()

        for name, cfg in qcvr_baselines.items():
            print(f"\n{'='*60}\n{cfg['description']}\n{'='*60}")
            qcvr_model = QCVRWrapper(
                model_id=args.model_id,
                use_qcvr=cfg["use_qcvr"],
                use_iacd=cfg["use_iacd"],
                lm_layer=cfg["lm_layer"],
                tau=cfg["tau"],
                lambda_=cfg["lambda_"],
                iacd_mode=cfg.get("iacd_mode", "first_token"),
                iacd_decay_steps=cfg.get("iacd_decay_steps", 8),
            )
            # Strip QCVR init kwargs before passing to generate()
            qcvr_keys = {
                "use_qcvr", "use_iacd", "lm_layer", "tau", "lambda_",
                "iacd_mode", "iacd_decay_steps", "question",
            }
            gen_cfg = {k: v for k, v in cfg.items() if k not in qcvr_keys}
            gen_cfg["question"] = cfg.get("question",
                                          "Please describe this image in detail.")
            agg, std, records = run_baseline(
                qcvr_model, image_ids, args.image_dir, gt_annotations,
                {**gen_cfg, "description": cfg["description"]},
                n_runs=args.n_runs, base_seed=args.seed,
            )
            all_metrics[name] = {"metrics": agg, "std": std}
            _print_row(name, agg, std)
            _save_one(out_dir, name, cfg, agg, std, records)
            del qcvr_model
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"CHAIR RESULTS SUMMARY | elapsed {elapsed/60:.1f} min")
    print(f"{'='*65}")
    print(f"{'Baseline':<26} {'CHAIRs':>9} {'CHAIRi':>9}")
    print("-" * 48)
    for name, entry in all_metrics.items():
        m = entry["metrics"]
        s = entry.get("std")
        s_str = f"±{s['CHAIRs']:.2f}" if s else ""
        i_str = f"±{s['CHAIRi']:.2f}" if s else ""
        print(f"{name:<26} {m['CHAIRs']:>6.2f}%{s_str:>5}  "
              f"{m['CHAIRi']:>6.2f}%{i_str:>5}")

    summary_path = os.path.join(out_dir, "chair_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
