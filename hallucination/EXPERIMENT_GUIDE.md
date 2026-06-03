# Hallucination Experiment Guide

This guide records the reproducible local workflow. It intentionally avoids
machine-specific SSH hosts, passwords, or cloud-instance details.

## 1. Environment

Run from the `hallucination/` directory.

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet omw-1.4
```

Recommended hardware: one RTX 4090-class GPU with 24 GB memory.

## 2. Data

```bash
bash download_data.sh
```

Expected layout:

```text
data/pope/coco_pope_{random,popular,adversarial}.json
data/coco/annotations/instances_val2014.json
data/coco/val2014/COCO_val2014_*.jpg
```

The data directory is ignored by Git.

## 3. Smoke Test

```bash
bash reproduce.sh smoke
```

This runs 10 POPE samples and 10 CHAIR samples to check model, data, and CUDA.

## 4. Full Reproduction

```bash
bash reproduce.sh full
```

Equivalent partial commands:

```bash
bash reproduce.sh pope
bash reproduce.sh chair
bash reproduce.sh latency
```

The full run:

1. Runs POPE baselines, VCD-tuned, and QCVR/IACD ablations.
2. Runs CHAIR with a unified decoding budget.
3. Runs the QCVR/IACD CHAIR component ablation.
4. Runs a small latency benchmark.

## 5. Main Entry Points

```bash
python run_pope.py --baseline required --split all
python run_pope.py --baseline qcvr_all --split all
python run_chair.py --baseline full --num_samples 500 --unify_tokens 64 --unify_no_repeat 3
python run_chair.py --baseline qcvr_all --num_samples 500 --unify_tokens 64 --unify_no_repeat 3
python bench_latency.py --num_samples 30 --max_new_tokens 64 --no_repeat 3
```

## 6. Outputs

All outputs are written under `results/` and are ignored by Git because they can
include generated captions, logs, and machine-specific run folders.

Cloud-instance launch commands and one-off remote helper scripts are intentionally
not part of the repository. Use `reproduce.sh` or the direct entry points above
on any GPU machine with the required data layout.
