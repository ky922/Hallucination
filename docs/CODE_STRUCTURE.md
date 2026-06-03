# Code Structure

The repository separates experiment configuration, model wrappers, benchmark
evaluation, and command-line orchestration.

## Directories

```text
hallucination/configs/
  Baseline presets, QCVR/IACD ablation presets, and CLI group selection.

hallucination/models/
  Model wrappers for LLaVA, QCVR/IACD, VCD, and ICD.

hallucination/eval/
  Benchmark-specific metric code for POPE, CHAIR, and CHAIR bootstrap tests.

hallucination/utils/
  Shared runtime helpers for seeding, timestamped output directories, JSON
  writing, and GPU memory cleanup.

hallucination/run_pope.py
hallucination/run_chair.py
  Thin command-line entry points. They parse arguments, select presets from
  `configs/`, manage model loading phases, and call the benchmark evaluators.
```

## Execution Flow

POPE:

1. `run_pope.py` parses CLI arguments.
2. `configs/pope.py` selects standard, SOTA, and QCVR/IACD method groups.
3. The script loads LLaVA for standard and SOTA phases.
4. QCVR/IACD methods are loaded in a separate phase to control GPU memory.
5. Metrics are computed by `eval/pope_eval.py` and saved under `results/`.

CHAIR:

1. `run_chair.py` parses CLI arguments and samples COCO image ids.
2. `configs/chair.py` selects standard, SOTA, and QCVR/IACD method groups.
3. Optional unified decoding overrides are applied to all selected methods.
4. The script runs LLaVA, SOTA, and QCVR/IACD phases.
5. Metrics are computed by `eval/chair_eval.py` and saved under `results/`.

This split keeps the experiment presets readable without forcing reviewers to
scan long runner scripts.
