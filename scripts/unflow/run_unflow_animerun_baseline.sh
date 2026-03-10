#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"${REPO_ROOT}/scripts/unflow/train_unflow_animerun.sh"
"${REPO_ROOT}/scripts/unflow/eval_unflow_animerun.sh"
