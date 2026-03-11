#!/usr/bin/env bash
set -euo pipefail
export CONDA_NO_PLUGINS=true
export TF_CUDNN_USE_AUTOTUNE=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

ENV_NAME="${ENV_NAME:-unflow_animerun_tf}"
EXPERIMENT="${EXPERIMENT:-unflow_animerun_c_medium}"
CONFIG_PROFILE="${CONFIG_PROFILE:-models/UnFlow/configs/animerun_medium.ini}"
OVERWRITE="${OVERWRITE:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNFLOW_ROOT="${REPO_ROOT}/models/UnFlow"
if [[ "${CONFIG_PROFILE}" = /* ]]; then
  CONFIG_PATH="${CONFIG_PROFILE}"
else
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PROFILE}"
fi

cp "${CONFIG_PATH}" "${UNFLOW_ROOT}/config.ini"

cd "${UNFLOW_ROOT}/src"

CUDA_WHEEL_LIBS="$(
  conda run -n "${ENV_NAME}" python -c "
import importlib.util
import os
modules = [
    'nvidia.cudnn', 'nvidia.cublas', 'nvidia.cuda_runtime',
    'nvidia.cufft', 'nvidia.curand', 'nvidia.cusolver',
    'nvidia.cusparse', 'nvidia.nccl', 'nvidia.cuda_cupti',
]
paths = []
for mod in modules:
    spec = importlib.util.find_spec(mod)
    if spec and spec.submodule_search_locations:
        lib_dir = os.path.join(spec.submodule_search_locations[0], 'lib')
        if os.path.isdir(lib_dir):
            paths.append(lib_dir)
print(':'.join(paths))
" | tr -d '\r'
)"
if [[ -n "${CUDA_WHEEL_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_WHEEL_LIBS}:${LD_LIBRARY_PATH:-}"
fi

OW_FLAG=""
if [[ "${OVERWRITE}" == "1" ]]; then
  OW_FLAG="--ow"
fi

conda run -n "${ENV_NAME}" python run.py --ex "${EXPERIMENT}" ${OW_FLAG}
