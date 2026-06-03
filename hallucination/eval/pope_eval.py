"""
POPE evaluation utilities.

POPE (Polling-based Object Probing Evaluation) asks binary Yes/No questions
about object existence. Three splits:
  random     — randomly sampled negative objects
  popular    — frequently co-occurring negative objects (harder)
  adversarial — adversarially chosen negatives (hardest)

Metrics:
  accuracy  — (TP + TN) / N
  f1        — harmonic mean of precision and recall
  yes_ratio — fraction of "yes" predictions (reveals answer-bias)
"""

import json
import os
from typing import Dict, List


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pope(data_dir: str, split: str) -> List[Dict]:
    """
    Load a POPE split from JSONL file.

    Expected filename: coco_pope_{split}.json  (one JSON object per line)
    Each line: {"question_id": ..., "image": "COCO_val2014_*.jpg",
                "text": "Is there a ... ?", "label": "yes"|"no"}

    Args:
        data_dir: directory containing the pope *.json files
        split: "random" | "popular" | "adversarial"

    Returns:
        List of sample dicts.
    """
    path = os.path.join(data_dir, f"coco_pope_{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"POPE file not found: {path}\n"
            "Run download_data.sh first, or set --data_dir correctly."
        )
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_yes_no(response: str) -> str:
    """
    Extract a yes/no decision from a (possibly verbose) model response.
    Falls back to "yes" when ambiguous (matches LLaVA's known answer bias).
    """
    r = response.strip().lower()
    # Direct prefix match — most common case
    if r.startswith("yes"):
        return "yes"
    if r.startswith("no"):
        return "no"
    # Substring search when model adds extra text
    has_yes = "yes" in r
    has_no  = "no"  in r
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "yes"  # default / ambiguous


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_pope_metrics(preds: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Compute POPE metrics from prediction and label lists.

    Args:
        preds:  list of "yes"/"no" strings (model predictions)
        labels: list of "yes"/"no" strings (ground truth)

    Returns:
        Dict with keys: accuracy, f1, precision, recall, yes_ratio
        (all values are percentages, rounded to 2 decimal places)
    """
    if len(preds) != len(labels):
        raise ValueError(f"len(preds)={len(preds)} != len(labels)={len(labels)}")

    tp = fp = tn = fn = 0
    for p, l in zip(preds, labels):
        p = p.lower().strip()
        l = l.lower().strip()
        if   p == "yes" and l == "yes": tp += 1
        elif p == "yes" and l == "no":  fp += 1
        elif p == "no"  and l == "no":  tn += 1
        else:                           fn += 1

    n         = tp + fp + tn + fn
    accuracy  = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    yes_ratio = (tp + fp) / n

    return {
        "accuracy":  round(accuracy  * 100, 2),
        "f1":        round(f1        * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall    * 100, 2),
        "yes_ratio": round(yes_ratio * 100, 2),
        # raw counts for debugging
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
