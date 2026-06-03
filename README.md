# Hallucination Mitigation for LVLMs

This repository contains the runnable experiment code for a training-free
hallucination mitigation project on LLaVA-1.5-7B. The method uses two
inference-time components:

- **QCVR**: Query-Conditioned Visual Relevance calibration, which filters inert
  visual tokens and reweights effective visual evidence.
- **IACD**: Inert-Anchored Contrastive Decoding, which builds an inert-token
  negative anchor and subtracts its logits.

The experiments evaluate hallucination on **POPE** and **CHAIR** with standard
inference, decoding baselines, prompt baselines, and contrastive-decoding SOTA
baselines.

## Repository Layout

```text
hallucination/
  models/                 LLaVA, QCVR/IACD, VCD, and ICD wrappers
  eval/                   POPE, CHAIR, and bootstrap evaluation code
  run_pope.py             POPE evaluation entry point
  run_chair.py            CHAIR evaluation entry point
  bench_latency.py        Latency benchmark
  reproduce.sh            Smoke/full reproduction script
```

Large datasets, generated caption files, raw timestamped runs, model caches,
experiment results, logs, paper files, and cloud-instance helper scripts are
intentionally excluded from Git.

## Setup

The full experiments were run with LLaVA-1.5-7B on a 24 GB RTX 4090-class GPU.

```bash
git clone https://github.com/ky922/Hallucination.git
cd Hallucination/hallucination

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m nltk.downloader punkt averaged_perceptron_tagger wordnet omw-1.4
bash download_data.sh
```

The default model is `llava-hf/llava-1.5-7b-hf`. Change it with
`--model_id` on the evaluation scripts if needed.

## Reproduce

Run a small environment check first:

```bash
cd hallucination
bash reproduce.sh smoke
```

Run the full experiment suite:

```bash
bash reproduce.sh full
```

The full mode runs POPE, CHAIR, QCVR/IACD ablations, and a latency benchmark.

Useful partial runs:

```bash
bash reproduce.sh pope
bash reproduce.sh chair
bash reproduce.sh latency
```

Expected runtime on an RTX 4090-class GPU is roughly 6 to 12 GPU hours,
depending on I/O, model cache location, and whether all ablations are rerun.

## Outputs

All generated outputs are written under `hallucination/results/` and are ignored
by Git. Keep or archive them outside the repository when preparing reports.

## What Is Not Included

This repository is kept to runnable experiment code and general reproduction
scripts only. It does not include the paper source, Overleaf packages, generated
results, raw run logs, machine-specific SSH commands, or one-off cloud execution
wrappers.
