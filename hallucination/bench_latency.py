#!/usr/bin/env python3
"""
Latency / throughput benchmark for the CHAIR generation path.

Training-free decoding methods are not computation-free. This script measures
the inference overhead with a per-method latency table on a small image subset.

Reported per method:
  - seconds / image (mean +- std)
  - new tokens generated / second  (throughput)
  - relative slowdown vs greedy

Usage:
  python bench_latency.py --num_samples 30 \
      --methods greedy qcvr_iacd iacd_first vcd icd \
      --max_new_tokens 64 --no_repeat 3
"""

import argparse
import gc
import json
import os
import sys
import time
from statistics import mean, pstdev

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.llava_wrapper import LLaVAWrapper
from models.sota_wrappers import VCDWrapper, ICDWrapper
from models.qcvr_wrapper import QCVRWrapper
from eval.chair_eval import load_coco_annotations

QUESTION = "Please describe this image in detail."


def coco_fn(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def sample_ids(gt, n, seed, image_dir):
    import random
    ids = sorted(gt.keys())
    random.seed(seed)
    random.shuffle(ids)
    out = []
    for i in ids:
        if os.path.exists(os.path.join(image_dir, coco_fn(i))):
            out.append(i)
            if len(out) >= n:
                break
    return out


def build(method, model_id):
    """Return (model, gen_kwargs_extra). gen kwargs for tokens are added later."""
    if method == "greedy":
        return LLaVAWrapper(model_id=model_id), {}
    if method == "vcd":
        return VCDWrapper(model_id=model_id, noise_std=0.1, alpha=1.0,
                          beta_apc=0.1), {}
    if method == "icd":
        return ICDWrapper(model_id=model_id, alpha=1.0, beta_apc=0.1), {}
    if method in ("qcvr_iacd", "iacd_first", "qcvr_only"):
        cfg = dict(use_qcvr=True, use_iacd=True, lm_layer=16, tau=0.2,
                   lambda_=0.3, iacd_mode="first_token")
        if method == "iacd_first":
            cfg["use_qcvr"] = False
        if method == "qcvr_only":
            cfg["lambda_"] = 0.0
        return QCVRWrapper(model_id=model_id, **cfg), {}
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default="data/coco/val2014")
    ap.add_argument("--annotation_file",
                    default="data/coco/annotations/instances_val2014.json")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--no_repeat", type=int, default=3)
    ap.add_argument("--methods", nargs="+",
                    default=["greedy", "iacd_first", "qcvr_iacd", "vcd", "icd"])
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--output", default="results/latency.json")
    args = ap.parse_args()

    gt = load_coco_annotations(args.annotation_file)
    ids = sample_ids(gt, args.num_samples, args.seed, args.image_dir)
    print(f"Benchmarking on {len(ids)} images, max_new_tokens={args.max_new_tokens}")

    tok = None
    results = {}
    for method in args.methods:
        print(f"\n=== {method} ===")
        model, _ = build(method, args.model_id)
        if tok is None:
            tok = model.processor.tokenizer
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                          no_repeat_ngram_size=args.no_repeat)

        # warmup (excluded)
        _ = model.generate(os.path.join(args.image_dir, coco_fn(ids[0])),
                           QUESTION, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times, ntoks = [], []
        for img_id in tqdm(ids, desc=method, leave=False):
            path = os.path.join(args.image_dir, coco_fn(img_id))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            cap = model.generate(path, QUESTION, **gen_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            ntoks.append(len(tok.encode(cap, add_special_tokens=False)))

        spi = mean(times)
        spi_sd = pstdev(times) if len(times) > 1 else 0.0
        tok_per_s = sum(ntoks) / sum(times) if sum(times) else 0.0
        results[method] = {
            "sec_per_image": round(spi, 3),
            "sec_per_image_std": round(spi_sd, 3),
            "tokens_per_sec": round(tok_per_s, 2),
            "mean_new_tokens": round(mean(ntoks), 1),
            "n": len(ids),
        }
        print(f"  {spi:.3f} s/img (+-{spi_sd:.3f})  "
              f"{tok_per_s:.1f} tok/s  mean_tokens={mean(ntoks):.1f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # relative slowdown vs greedy
    base = results.get("greedy", {}).get("sec_per_image")
    print("\n==================== LATENCY SUMMARY ====================")
    print(f"{'method':<14}{'s/img':>10}{'tok/s':>10}{'slowdown':>11}")
    for m, r in results.items():
        slow = f"{r['sec_per_image']/base:.2f}x" if base else "-"
        print(f"{m:<14}{r['sec_per_image']:>10.3f}{r['tokens_per_sec']:>10.1f}"
              f"{slow:>11}")
        r["slowdown_vs_greedy"] = (round(r["sec_per_image"] / base, 2)
                                   if base else None)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
