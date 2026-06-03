#!/usr/bin/env bash
# Local syntax checks for the code artifact.
set -euo pipefail
cd "$(dirname "$0")/.."

bash -n hallucination/reproduce.sh
bash -n hallucination/download_data.sh
bash -n hallucination/quick_test.sh

python -m py_compile $(git ls-files \
  'hallucination/*.py' \
  'hallucination/eval/*.py' \
  'hallucination/models/*.py')
