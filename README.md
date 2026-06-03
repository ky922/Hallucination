# Hallucination Mitigation for LVLMs

This repository contains the runnable experiment code for a training-free
hallucination mitigation project on open-source large vision-language models.
The current implementation targets LLaVA-1.5-7B and evaluates hallucination on
POPE and CHAIR.

The method has two inference-time components:

- **QCVR**: Query-Conditioned Visual Relevance calibration, which filters inert
  visual tokens and reweights effective visual evidence.
- **IACD**: Inert-Anchored Contrastive Decoding, which builds an inert-token
  negative anchor and subtracts its logits during decoding.

## Repository Scope

This repository is kept as a research-code artifact. It includes:

- model wrappers for LLaVA, QCVR/IACD, VCD, and ICD;
- POPE and CHAIR evaluation code;
- command-line experiment entry points;
- data download, smoke-test, and reproduction scripts;
- artifact documentation for environment, runtime, and outputs.

It intentionally does not include paper source files, Overleaf packages,
generated results, raw logs, generated captions, machine-specific SSH commands,
cloud-instance helper scripts, model weights, or datasets.

## Layout

```text
hallucination/
  configs/                Baseline presets and experiment group selection
  models/                 LLaVA, QCVR/IACD, VCD, and ICD wrappers
  eval/                   POPE, CHAIR, and bootstrap evaluation code
  utils/                  Shared runtime helpers
  run_pope.py             POPE evaluation entry point
  run_chair.py            CHAIR evaluation entry point
  bench_latency.py        Latency benchmark
  download_data.sh        POPE and COCO val2014 download helper
  quick_test.sh           Small GPU/data sanity test
  reproduce.sh            Smoke, partial, and full reproduction runner
docs/
  ARTIFACT.md             Artifact and reproducibility notes
  ASSETS.md               External model, dataset, and method assets
  CODE_STRUCTURE.md       Code organization and execution flow
  REPRODUCIBILITY.md      Short checklist for reviewers/users
scripts/
  static_check.sh          Local shell/Python syntax checks
environment.yml           Optional conda environment
```

## Environment

The experiments were designed for one RTX 4090-class GPU with 24 GB memory.
The default model is `llava-hf/llava-1.5-7b-hf`.

```bash
git clone https://github.com/ky922/Hallucination.git
cd Hallucination/hallucination

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet omw-1.4
```

Alternatively, create a conda environment:

```bash
conda env create -f ../environment.yml
conda activate hallucination-lvlm
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet omw-1.4
```

Download POPE and COCO val2014 data:

```bash
bash download_data.sh
```

Expected data layout:

```text
hallucination/data/pope/coco_pope_{random,popular,adversarial}.json
hallucination/data/coco/annotations/instances_val2014.json
hallucination/data/coco/val2014/COCO_val2014_*.jpg
```

## Reproduction

Run a small sanity check first:

```bash
bash reproduce.sh smoke
```

Run the full experiment suite:

```bash
bash reproduce.sh full
```

Useful partial runs:

```bash
bash reproduce.sh pope
bash reproduce.sh chair
bash reproduce.sh latency
```

The full mode runs POPE baselines, CHAIR baselines under a unified decoding
budget, QCVR/IACD ablations, and a small latency benchmark. Expected runtime on
an RTX 4090-class GPU is roughly 6 to 12 GPU hours, depending on storage, model
cache location, and whether all ablations are rerun.

All generated outputs are written under `hallucination/results/`, which is
ignored by Git.

Run local syntax checks:

```bash
cd ..
bash scripts/static_check.sh
```

## Main Commands

```bash
python run_pope.py --baseline required --split all
python run_pope.py --baseline qcvr_all --split all
python run_chair.py --baseline full --num_samples 500 --unify_tokens 64 --unify_no_repeat 3
python run_chair.py --baseline qcvr_all --num_samples 500 --unify_tokens 64 --unify_no_repeat 3
python bench_latency.py --num_samples 30 --max_new_tokens 64 --no_repeat 3
```

Use `--model_id` on the evaluation scripts to change the Hugging Face model.

## Artifact Notes

See [docs/ARTIFACT.md](docs/ARTIFACT.md) for hardware assumptions, expected
outputs, metrics, runtime estimates, and reproducibility notes. See
[docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) for the code organization,
[docs/ASSETS.md](docs/ASSETS.md) for external assets, and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a short checklist.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
