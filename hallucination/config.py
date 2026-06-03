"""Central configuration — edit paths here before running."""
import os

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# ── Data paths ─────────────────────────────────────────────────────────────────
BASE_DATA_DIR = "data"

COCO_IMAGE_DIR       = os.path.join(BASE_DATA_DIR, "coco", "val2014")
COCO_ANNOTATION_FILE = os.path.join(BASE_DATA_DIR, "coco", "annotations",
                                    "instances_val2014.json")
POPE_DATA_DIR        = os.path.join(BASE_DATA_DIR, "pope")

# ── Evaluation settings ────────────────────────────────────────────────────────
CHAIR_NUM_SAMPLES = 500
CHAIR_SEED        = 42
POPE_SPLITS       = ["random", "popular", "adversarial"]

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "results"
