#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-unflow_animerun_tf}"
EXPERIMENT="${EXPERIMENT:-unflow_animerun_c_medium}"
CONFIG_PROFILE="${CONFIG_PROFILE:-models/UnFlow/configs/animerun_medium.ini}"
NUM_PAIRS="${NUM_PAIRS:--1}"
GPU_ID="${GPU_ID:-0}"
RESULTS_JSON="${RESULTS_JSON:-workspaces/unflow_animerun/results_${EXPERIMENT}.json}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNFLOW_ROOT="${REPO_ROOT}/models/UnFlow"
if [[ "${CONFIG_PROFILE}" = /* ]]; then
  CONFIG_PATH="${CONFIG_PROFILE}"
else
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PROFILE}"
fi
if [[ "${RESULTS_JSON}" = /* ]]; then
  RESULTS_PATH="${RESULTS_JSON}"
else
  RESULTS_PATH="${REPO_ROOT}/${RESULTS_JSON}"
fi

mkdir -p "$(dirname "${RESULTS_PATH}")"
cp "${CONFIG_PATH}" "${UNFLOW_ROOT}/config.ini"

cd "${UNFLOW_ROOT}/src"

conda run -n "${ENV_NAME}" python eval_gui.py \
  --dataset animerun \
  --variant test \
  --ex "${EXPERIMENT}" \
  --num "${NUM_PAIRS}" \
  --num_vis 0 \
  --gpu "${GPU_ID}" \
  --no_display \
  --results_json "${RESULTS_PATH}"
