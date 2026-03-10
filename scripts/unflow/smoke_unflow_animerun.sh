#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-unflow_animerun_tf}"
EXPERIMENT="${EXPERIMENT:-unflow_animerun_smoke}"
CONFIG_PROFILE="${CONFIG_PROFILE:-models/UnFlow/configs/animerun_medium.ini}"
GPU_ID="${GPU_ID:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNFLOW_ROOT="${REPO_ROOT}/models/UnFlow"
TMP_CONFIG="${UNFLOW_ROOT}/configs/animerun_smoke.tmp.ini"

cp "${REPO_ROOT}/${CONFIG_PROFILE}" "${TMP_CONFIG}"

# Reduce iterations for smoke run.
sed -i 's/^num_iters = .*/num_iters = 100/' "${TMP_CONFIG}"
sed -i 's/^save_interval = .*/save_interval = 100/' "${TMP_CONFIG}"

CONFIG_PROFILE="models/UnFlow/configs/animerun_smoke.tmp.ini" \
EXPERIMENT="${EXPERIMENT}" \
OVERWRITE=1 \
ENV_NAME="${ENV_NAME}" \
"${REPO_ROOT}/scripts/unflow/train_unflow_animerun.sh"

CONFIG_PROFILE="models/UnFlow/configs/animerun_smoke.tmp.ini" \
EXPERIMENT="${EXPERIMENT}" \
NUM_PAIRS=50 \
GPU_ID="${GPU_ID}" \
ENV_NAME="${ENV_NAME}" \
RESULTS_JSON="workspaces/unflow_animerun/results_${EXPERIMENT}_smoke.json" \
"${REPO_ROOT}/scripts/unflow/eval_unflow_animerun.sh"

echo "Smoke pipeline finished."
