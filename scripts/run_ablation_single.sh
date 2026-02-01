#!/bin/bash
# ============================================================================
# Run Single Ablation Experiment
# ============================================================================
# Usage: ./run_ablation_single.sh <experiment_number>
#
# Examples:
#   ./run_ablation_single.sh 01   # Run baseline
#   ./run_ablation_single.sh 07   # Run full v2
# ============================================================================

set -e

PYTHON="/srv/conda/envs/serverai/opf/bin/python"
SCRIPT="scripts/train_unsup_animerun.py"
CONFIG_DIR="configs/ablation"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_number>"
    echo ""
    echo "Available experiments:"
    echo "  01 - Baseline (No SAM)"
    echo "  02 - SAM Original"
    echo "  03 - Soft Boundary Only"
    echo "  04 - Learned Embeddings Only"
    echo "  05 - Temporal Propagation Only"
    echo "  06 - Attention Bias Only"
    echo "  07 - Full SAM v2"
    exit 1
fi

EXP_NUM=$1

case $EXP_NUM in
    01) CONFIG="01_baseline" ;;
    02) CONFIG="02_sam_original" ;;
    03) CONFIG="03_soft_boundary" ;;
    04) CONFIG="04_learned_embed" ;;
    05) CONFIG="05_temporal_prop" ;;
    06) CONFIG="06_attention_bias" ;;
    07) CONFIG="07_full_v2" ;;
    *)
        echo "Invalid experiment number: $EXP_NUM"
        exit 1
        ;;
esac

echo "Running experiment: $CONFIG"
echo "Config: ${CONFIG_DIR}/${CONFIG}.yaml"
echo "Started at: $(date)"

$PYTHON $SCRIPT --config "${CONFIG_DIR}/${CONFIG}.yaml"

echo "Completed at: $(date)"
