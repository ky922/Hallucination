# External Assets

This repository does not redistribute datasets, model weights, or third-party
baseline repositories. The code downloads or loads these assets from their
public sources when the user runs the setup commands.

## Models

- LLaVA-1.5-7B: default Hugging Face model id
  `llava-hf/llava-1.5-7b-hf`.
  <https://huggingface.co/llava-hf/llava-1.5-7b-hf>

Users should verify the current license and terms of use for any model they
load with `--model_id`.

## Datasets and Benchmarks

- POPE: object-existence yes/no hallucination benchmark. The download helper
  fetches COCO POPE split files from the public POPE repository.
  <https://github.com/AoiDragon/POPE>
- COCO val2014: image and annotation source for POPE image paths and CHAIR
  caption hallucination evaluation.
  <https://cocodataset.org/#download>

Datasets are downloaded into `hallucination/data/` and ignored by Git.

## Baseline Methods

The code includes local wrappers for comparison methods used in the experiments:

- standard greedy, beam-search, sampling, and prompt baselines;
- VCD-style visual contrastive decoding;
- ICD-style instruction contrastive decoding;
- QCVR and IACD variants for ablation.

These wrappers are included as experiment code. Any paper using the repository
should cite the original method papers for the corresponding baselines and
benchmarks.

## Generated Assets

Generated captions, metric JSON files, latency outputs, logs, and paper-specific
tables or figures are not tracked. They are written under `hallucination/results/`
when experiments are run.
