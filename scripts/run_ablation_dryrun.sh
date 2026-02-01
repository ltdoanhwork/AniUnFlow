#!/bin/bash
# ============================================================================
# Dry Run All Ablation Experiments
# ============================================================================
# Quick verification that all configs work before starting real training.
# ============================================================================

set -e

PYTHON="/srv/conda/envs/serverai/opf/bin/python"
SCRIPT="scripts/train_unsup_animerun.py"
CONFIG_DIR="configs/ablation"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================"
echo "  Dry Run: All Ablation Experiments"
echo "============================================"
echo ""

experiments=(
    "01_baseline"
    "02_sam_original"
    "03_soft_boundary"
    "04_learned_embed"
    "05_temporal_prop"
    "06_attention_bias"
    "07_full_v2"
)

failed=()

for exp in "${experiments[@]}"; do
    echo -n "Testing $exp... "
    if $PYTHON $SCRIPT --config "${CONFIG_DIR}/${exp}.yaml" --dry-run > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        failed+=("$exp")
    fi
done

echo ""
echo "============================================"

if [ ${#failed[@]} -eq 0 ]; then
    echo -e "${GREEN}All experiments passed dry run!${NC}"
    exit 0
else
    echo -e "${RED}Failed experiments:${NC}"
    for f in "${failed[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
