#!/usr/bin/env bash
set -euo pipefail
export CONDA_NO_PLUGINS=true

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

OW_FLAG=""
if [[ "${OVERWRITE}" == "1" ]]; then
  OW_FLAG="--ow"
fi

conda run -n "${ENV_NAME}" python run.py --ex "${EXPERIMENT}" ${OW_FLAG}
