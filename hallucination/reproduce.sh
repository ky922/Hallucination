#!/usr/bin/env bash
# Run the hallucination experiments or selected subsets.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-full}"
CHAIR_SAMPLES="${CHAIR_SAMPLES:-500}"
POPE_SAMPLES="${POPE_SAMPLES:--1}"

run_smoke() {
  bash quick_test.sh
}

run_pope() {
  "$PYTHON_BIN" run_pope.py \
    --baseline required \
    --split all \
    --max_samples "$POPE_SAMPLES" \
    --output_dir results/pope

  "$PYTHON_BIN" run_pope.py \
    --baseline vcd_tuned \
    --split all \
    --max_samples "$POPE_SAMPLES" \
    --output_dir results/pope_vcd_tuned

  "$PYTHON_BIN" run_pope.py \
    --baseline qcvr_all \
    --split all \
    --max_samples "$POPE_SAMPLES" \
    --output_dir results/pope_qcvr_ablation
}

run_chair() {
  "$PYTHON_BIN" run_chair.py \
    --baseline full \
    --num_samples "$CHAIR_SAMPLES" \
    --unify_tokens 64 \
    --unify_no_repeat 3 \
    --output_dir results/chair_unified

  "$PYTHON_BIN" run_chair.py \
    --baseline qcvr_all \
    --num_samples "$CHAIR_SAMPLES" \
    --unify_tokens 64 \
    --unify_no_repeat 3 \
    --output_dir results/chair_qcvr_core_unified
}

run_latency() {
  "$PYTHON_BIN" bench_latency.py \
    --num_samples "${LATENCY_SAMPLES:-30}" \
    --max_new_tokens 64 \
    --no_repeat 3 \
    --methods greedy iacd_first qcvr_iacd vcd icd \
    --output results/latency.json
}

case "$MODE" in
  smoke)
    run_smoke
    ;;
  pope)
    run_pope
    ;;
  chair)
    run_chair
    ;;
  latency)
    run_latency
    ;;
  full)
    run_pope
    run_chair
    run_latency
    ;;
  *)
    echo "Usage: bash reproduce.sh {smoke|pope|chair|latency|full}"
    exit 2
    ;;
esac
