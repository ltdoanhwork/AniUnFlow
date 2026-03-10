#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-unflow_animerun_tf}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Reusing it."
else
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  tensorflow==2.16.2 \
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
