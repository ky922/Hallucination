#!/usr/bin/env bash
# Small smoke test for the model, data, CUDA, POPE, and CHAIR paths.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========================================"
echo "Smoke test started"
echo "========================================"

echo "[1/4] Checking CUDA"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true
"$PYTHON_BIN" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo
echo "[2/4] Checking data"
if [ ! -f "data/coco/annotations/instances_val2014.json" ]; then
  echo "Missing COCO annotations. Run: bash download_data.sh"
  exit 1
fi
if [ ! -d "data/coco/val2014" ]; then
  echo "Missing COCO val2014 images. Run: bash download_data.sh"
  exit 1
fi
if [ ! -f "data/pope/coco_pope_random.json" ]; then
  echo "Missing POPE annotations. Run: bash download_data.sh"
  exit 1
fi

echo
echo "[3/4] Running POPE smoke test: 10 samples, random split"
"$PYTHON_BIN" run_pope.py \
  --max_samples 10 \
  --split random \
  --baseline greedy_logits \
  --output_dir results/quick_test/pope

POPE_JSON=$(find results/quick_test/pope -name pope_random.json -type f 2>/dev/null | sort | tail -1 || true)
if [ -n "${POPE_JSON:-}" ]; then
  echo "Latest POPE result: $POPE_JSON"
  "$PYTHON_BIN" -m json.tool "$POPE_JSON" | grep -A 8 '"metrics"' || true
fi

echo
echo "[4/4] Running CHAIR smoke test: 10 images"
"$PYTHON_BIN" run_chair.py \
  --num_samples 10 \
  --baseline greedy \
  --output_dir results/quick_test/chair

CHAIR_JSON=$(find results/quick_test/chair -name chair_summary.json -type f 2>/dev/null | sort | tail -1 || true)
if [ -n "${CHAIR_JSON:-}" ]; then
  echo "Latest CHAIR result: $CHAIR_JSON"
  "$PYTHON_BIN" -m json.tool "$CHAIR_JSON"
fi

echo
echo "========================================"
echo "Smoke test finished"
echo "Next commands:"
echo "  bash reproduce.sh full"
echo "  bash reproduce.sh pope"
echo "  bash reproduce.sh chair"
echo "========================================"
