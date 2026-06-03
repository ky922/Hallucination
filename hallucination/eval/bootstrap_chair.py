#!/usr/bin/env python3
"""
Paired bootstrap significance test for CHAIR between two caption sets.

Motivation
----------
The headline CHAIR gain (greedy CHAIRi 19.59 -> ours 16.62) conflates two
changes: (a) decoding config (64 tokens, repeats) -> (48 tokens, no_repeat=3),
and (b) the QCVR intervention. The "short greedy" control already reaches
CHAIRi 17.07 using the SAME decoding config as ours, so the QCVR-attributable
delta is only ~0.45. This script tests whether that residual delta is
statistically distinguishable from zero via a paired image-level bootstrap.

CHAIRi = sum_i hallucinated_i / sum_i mentioned_i   (ratio of sums)
CHAIRs = mean_i [ hallucinated_i > 0 ]

Both are recomputed on each bootstrap resample of images (paired: the SAME
resampled image set is used for both methods, since captions are per-image).

Usage
-----
python eval/bootstrap_chair.py \
    --a results/chair_qcvr_tuned500/20260602_182203/chair_qcvr_iacd.json \
    --b results/chair_ablation_greedy_short/20260602_184253/chair_greedy_short_norepeat.json \
    --annotation_file data/coco/annotations/instances_val2014.json \
    --n_boot 10000 --seed 42 --label_a "QCVR" --label_b "ShortGreedy"

Interpretation: delta = metric(B) - metric(A). For CHAIRi/CHAIRs (lower=better),
positive delta means A is better than B. We report the 95% CI of delta and the
one-sided bootstrap p-value for "A is NOT better than B" (delta <= 0).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.chair_eval import load_coco_annotations, extract_objects  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402


def load_captions(path):
    """Return dict image_id -> caption string."""
    d = json.load(open(path))
    caps = d.get("captions", d if isinstance(d, list) else [])
    out = {}
    for rec in caps:
        out[int(rec["image_id"])] = rec["caption"]
    return out, d.get("metrics", {})


def per_image_counts(captions, gt, lemmatizer):
    """Return arrays (mentioned, hallucinated) aligned to a fixed image order."""
    ids = sorted(captions.keys())
    mentioned = np.zeros(len(ids), dtype=np.float64)
    hallucinated = np.zeros(len(ids), dtype=np.float64)
    for k, img_id in enumerate(ids):
        pred = extract_objects(captions[img_id], lemmatizer)
        g = gt.get(img_id, set())
        hall = pred - g
        mentioned[k] = len(pred)
        hallucinated[k] = len(hall)
    return ids, mentioned, hallucinated


def chairi(mentioned, hallucinated, idx):
    m = mentioned[idx].sum()
    h = hallucinated[idx].sum()
    return (h / m * 100.0) if m > 0 else 0.0


def chairs(mentioned, hallucinated, idx):
    return float((hallucinated[idx] > 0).mean() * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="caption json A (e.g. ours)")
    ap.add_argument("--b", required=True, help="caption json B (e.g. control)")
    ap.add_argument("--annotation_file",
                    default="data/coco/annotations/instances_val2014.json")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    args = ap.parse_args()

    print(f"Loading COCO annotations from {args.annotation_file} ...")
    gt = load_coco_annotations(args.annotation_file)
    lem = WordNetLemmatizer()

    caps_a, m_a = load_captions(args.a)
    caps_b, m_b = load_captions(args.b)

    common = sorted(set(caps_a) & set(caps_b))
    if len(common) != len(caps_a) or len(common) != len(caps_b):
        print(f"[warn] image-id mismatch: A={len(caps_a)} B={len(caps_b)} "
              f"common={len(common)}; using {len(common)} paired images.")
    caps_a = {i: caps_a[i] for i in common}
    caps_b = {i: caps_b[i] for i in common}

    ids_a, men_a, hal_a = per_image_counts(caps_a, gt, lem)
    ids_b, men_b, hal_b = per_image_counts(caps_b, gt, lem)
    assert ids_a == ids_b, "image order mismatch after pairing"
    n = len(ids_a)

    full = np.arange(n)
    ci_a = chairi(men_a, hal_a, full)
    ci_b = chairi(men_b, hal_b, full)
    cs_a = chairs(men_a, hal_a, full)
    cs_b = chairs(men_b, hal_b, full)

    print("\n=== Point estimates (n={}) ===".format(n))
    print(f"  {args.label_a:<14} CHAIRi={ci_a:6.2f}  CHAIRs={cs_a:6.2f}")
    print(f"  {args.label_b:<14} CHAIRi={ci_b:6.2f}  CHAIRs={cs_b:6.2f}")
    print(f"  delta(B-A)     CHAIRi={ci_b-ci_a:+6.2f}  CHAIRs={cs_b-cs_a:+6.2f}")
    print("  (positive delta => A is better / lower hallucination)")

    rng = np.random.default_rng(args.seed)
    d_ci = np.empty(args.n_boot)
    d_cs = np.empty(args.n_boot)
    for t in range(args.n_boot):
        idx = rng.integers(0, n, size=n)  # paired resample
        d_ci[t] = chairi(men_b, hal_b, idx) - chairi(men_a, hal_a, idx)
        d_cs[t] = chairs(men_b, hal_b, idx) - chairs(men_a, hal_a, idx)

    def report(name, point, dist):
        lo, hi = np.percentile(dist, [2.5, 97.5])
        # one-sided p: prob that A is NOT better (delta <= 0)
        p = float((dist <= 0).mean())
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {name}: delta={point:+.2f}  95%CI=[{lo:+.2f}, {hi:+.2f}]  "
              f"p(A not better)={p:.4f}  -> {sig}")

    print(f"\n=== Paired bootstrap (n_boot={args.n_boot}, seed={args.seed}) ===")
    print(f"  Testing whether {args.label_a} improves over {args.label_b}:")
    report("CHAIRi", ci_b - ci_a, d_ci)
    report("CHAIRs", cs_b - cs_a, d_cs)


if __name__ == "__main__":
    main()
