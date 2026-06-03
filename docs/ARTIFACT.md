# Artifact Documentation

This document summarizes the code artifact boundary, hardware assumptions,
runtime expectations, and commands needed to reproduce the experiments.

## Scope

Included:

- training-free inference methods: QCVR and IACD;
- model wrappers for LLaVA-1.5-7B, VCD, ICD, and QCVR/IACD;
- POPE and CHAIR evaluation code;
- smoke, partial, and full reproduction scripts;
- latency measurement code.

Not included:

- manuscript source, Overleaf packages, generated figures, or tables;
- generated captions, raw logs, or timestamped result folders;
- datasets, model weights, and model cache directories;
- machine-specific SSH commands or one-off cloud execution scripts.

External model, dataset, and baseline assets are listed in `docs/ASSETS.md`.

## Hardware

The intended hardware is one RTX 4090-class GPU with 24 GB memory. Smaller GPUs
may require quantization, CPU offload, or smaller subsets. The smoke test is the
recommended first check on a new machine.

## Software

The code was written for Python 3.10+ with PyTorch, Transformers, Accelerate,
Pillow, pycocotools, and NLTK. Install dependencies from:

```bash
cd hallucination
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet omw-1.4
```

Conda users can instead run:

```bash
conda env create -f ../environment.yml
conda activate hallucination-lvlm
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet omw-1.4
```

## Data

Download POPE and COCO val2014:

```bash
cd hallucination
bash download_data.sh
```

Expected layout:

```text
data/pope/coco_pope_{random,popular,adversarial}.json
data/coco/annotations/instances_val2014.json
data/coco/val2014/COCO_val2014_*.jpg
```

The data directory is ignored by Git.

## Smoke Test

```bash
cd hallucination
bash reproduce.sh smoke
```

This runs 10 POPE samples and 10 CHAIR images. It checks CUDA availability,
data paths, model loading, generation, and metric computation.

## Full Reproduction

```bash
cd hallucination
bash reproduce.sh full
```

The full script runs:

1. POPE required baselines and selected SOTA/QCVR comparisons.
2. CHAIR baselines under a unified decoding budget.
3. QCVR/IACD CHAIR ablations.
4. A small latency benchmark.

Environment variables:

```bash
POPE_SAMPLES=-1       # all POPE samples; set a positive value for a subset
CHAIR_SAMPLES=500     # number of COCO images for CHAIR
LATENCY_SAMPLES=30    # number of images for latency
PYTHON_BIN=python     # Python executable
```

## Metrics

POPE reports Accuracy, Precision, Recall, F1, and yes-ratio for yes/no object
existence questions.

CHAIR reports sentence-level hallucination, instance-level hallucination, object
coverage statistics, and generated captions for each evaluated method.

Latency reports seconds per image, generated-token throughput, and slowdown
relative to greedy decoding.

## Outputs

All generated outputs are written under `hallucination/results/` and ignored by
Git. Typical paths are:

```text
results/pope/
results/pope_vcd_tuned/
results/pope_qcvr_ablation/
results/chair_unified/
results/chair_qcvr_core_unified/
results/latency.json
```

## Expected Runtime

On an RTX 4090-class GPU, the full suite is expected to take roughly 6 to 12 GPU
hours. Runtime depends on disk speed, model cache location, data availability,
and whether all baselines are rerun from scratch.

## Reproducibility Notes

- Python entry points default to seed 42.
- CHAIR reproduction uses a unified decoding budget of `max_new_tokens=64` and
  `no_repeat_ngram_size=3`.
- Stochastic baselines can be repeated with `--n_runs`.
- Generated outputs are timestamped to avoid overwriting earlier runs.
- Precomputed results are intentionally not committed; rerun the scripts to
  regenerate them.
