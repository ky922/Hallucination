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
import os
import random
import sys
import time
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.chair import (  # noqa: E402
    GENERATION_KEYS,
    QCVR_INIT_KEYS,
    QUESTION,
    apply_unified_decoding,
    baseline_choices,
    select_baseline_groups,
)
from utils.runtime import release_memory, save_json, set_seed, timestamped_output_dir  # noqa: E402
from models.llava_wrapper import LLaVAWrapper
from models.sota_wrappers import VCDWrapper, ICDWrapper
from models.qcvr_wrapper import QCVRWrapper
from eval.chair_eval import load_coco_annotations, compute_chair


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
    question = config.get("question", QUESTION)
    gen_kwargs = {k: v for k, v in config.items() if k in GENERATION_KEYS}

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
    save_json(out_path, out)
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
                        choices=baseline_choices())
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
    out_dir = timestamped_output_dir(args.output_dir)

    save_json(os.path.join(out_dir, "config.json"), vars(args))

    print("Loading COCO annotations...")
    gt_annotations = load_coco_annotations(args.annotation_file)

    print(f"Sampling {args.num_samples} images (seed={args.seed})...")
    image_ids = sample_image_ids(gt_annotations, args.num_samples,
                                 args.seed, args.image_dir)
    print(f"Using {len(image_ids)} images.")

    save_json(os.path.join(out_dir, "image_ids.json"), image_ids)

    llava_baselines, sota_baselines, qcvr_baselines = select_baseline_groups(args)

    # ── Unified decoding override ─────────────────────────────────────────────
    # For a fair main-table comparison, every method must use the SAME generation
    # budget. Otherwise the headline CHAIR gain conflates the QCVR/IACD effect
    # with a shorter / no-repeat decoding config (see ablation "short greedy").
    apply_unified_decoding(
        (llava_baselines, sota_baselines, qcvr_baselines),
        args.unify_tokens,
        args.unify_no_repeat,
    )
    if args.unify_tokens > 0 or args.unify_no_repeat >= 0:
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
        release_memory()

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
            gen_cfg = {k: v for k, v in cfg.items() if k not in QCVR_INIT_KEYS}
            gen_cfg["question"] = cfg.get("question", QUESTION)
            agg, std, records = run_baseline(
                qcvr_model, image_ids, args.image_dir, gt_annotations,
                {**gen_cfg, "description": cfg["description"]},
                n_runs=args.n_runs, base_seed=args.seed,
            )
            all_metrics[name] = {"metrics": agg, "std": std}
            _print_row(name, agg, std)
            _save_one(out_dir, name, cfg, agg, std, records)
            del qcvr_model
            release_memory()

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
    save_json(summary_path, all_metrics)
    print(f"\nAll results → {out_dir}/")


if __name__ == "__main__":
    main()
