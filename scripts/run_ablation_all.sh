#!/bin/bash
# ============================================================================
# SAM Improvement Ablation Study - Run All Experiments
# ============================================================================
# This script runs all ablation experiments sequentially.
# Each experiment trains for 50 epochs and logs to TensorBoard.
#
# Experiments:
#   01: Baseline (No SAM)
#   02: SAM Original (UnSAMFlow style)
#   03: Soft Boundary Only
#   04: Learned Embeddings Only
#   05: Temporal Propagation Only
#   06: Attention Bias Only
#   07: Full SAM v2 (All Improvements)
# ============================================================================

set -e  # Exit on error

# Python interpreter
PYTHON="/srv/conda/envs/serverai/opf/bin/python"
SCRIPT="scripts/train_unsup_animerun.py"
CONFIG_DIR="configs/ablation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  SAM Improvement Ablation Study${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Function to run single experiment
run_experiment() {
    local config_name=$1
    local config_path="${CONFIG_DIR}/${config_name}.yaml"
    
    echo -e "${YELLOW}Starting experiment: ${config_name}${NC}"
    echo "Config: $config_path"
    echo "Started at: $(date)"
    echo ""
    
    $PYTHON $SCRIPT --config $config_path
    
    echo -e "${GREEN}Completed: ${config_name}${NC}"
    echo "Finished at: $(date)"
    echo ""
}

# Run all experiments
experiments=(
    "01_baseline"
    "02_sam_original"
    "03_soft_boundary"
    "04_learned_embed"
    "05_temporal_prop"
    "06_attention_bias"
    "07_full_v2"
)

for exp in "${experiments[@]}"; do
    run_experiment "$exp"
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  All experiments completed!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results saved in outputs/ablation_*/"
echo "TensorBoard logs in outputs/ablation_*/tb_*/"
echo ""
echo "To compare results:"
echo "  tensorboard --logdir outputs/"
