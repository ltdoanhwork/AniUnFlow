#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-unflow_animerun_tf}"
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic

if conda run -n "${ENV_NAME}" python -V >/dev/null 2>&1; then
  echo "Conda env '${ENV_NAME}' already exists. Reusing it."
else
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  "tensorflow[and-cuda]==2.16.2" \
  tf_slim \
  pillow \
  matplotlib \
  pypng \
  rarfile

conda run -n "${ENV_NAME}" python - <<'PY'
import tensorflow as tf
import tf_slim
print("tensorflow", tf.__version__)
print("tf_slim", tf_slim.__version__ if hasattr(tf_slim, "__version__") else "ok")
PY

echo "Environment '${ENV_NAME}' is ready."
