# Reproducibility Checklist

This checklist is intended for reviewers or users who want to verify the code
artifact without inspecting paper files or precomputed results.

## Repository Contents

- [ ] No generated result folders are tracked.
- [ ] No paper source files or Overleaf packages are tracked.
- [ ] No machine-specific SSH commands or cloud helper scripts are tracked.
- [ ] Experiment entry points are present under `hallucination/`.
- [ ] Artifact notes are present under `docs/`.

## Environment

- [ ] Python 3.10+ environment is available.
- [ ] Dependencies are installed from `hallucination/requirements.txt` or
      `environment.yml`.
- [ ] NLTK resources are installed.
- [ ] One CUDA GPU is visible to PyTorch.

Optional local syntax check:

```bash
bash scripts/static_check.sh
```

## Data

- [ ] `data/pope/coco_pope_random.json` exists.
- [ ] `data/pope/coco_pope_popular.json` exists.
- [ ] `data/pope/coco_pope_adversarial.json` exists.
- [ ] `data/coco/annotations/instances_val2014.json` exists.
- [ ] `data/coco/val2014/` contains COCO val2014 images.

## Smoke Test

```bash
cd hallucination
bash reproduce.sh smoke
```

Expected behavior:

- POPE runs on 10 random-split samples.
- CHAIR runs on 10 COCO images.
- Outputs are written to `results/quick_test/`.

## Full Run

```bash
cd hallucination
bash reproduce.sh full
```

Expected behavior:

- POPE runs required, VCD, and QCVR/IACD comparison settings.
- CHAIR runs full and QCVR/IACD ablation settings.
- Latency benchmark writes `results/latency.json`.
- All outputs remain ignored by Git.
