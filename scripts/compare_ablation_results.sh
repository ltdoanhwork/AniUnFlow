#!/bin/bash
# ============================================================================
# Compare Ablation Results
# ============================================================================
# Extracts and compares final EPE metrics from all ablation experiments.
# ============================================================================

echo "============================================"
echo "  Ablation Study Results Comparison"
echo "============================================"
echo ""

OUTPUT_DIR="outputs"

# Header
printf "%-25s %10s %10s %10s %10s\n" "Experiment" "EPE" "1px" "3px" "5px"
printf "%s\n" "---------------------------------------------------------------------"

for exp_dir in $OUTPUT_DIR/ablation_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        
        # Try to extract metrics from latest checkpoint or log
        log_file="$exp_dir/training.log"
        if [ -f "$log_file" ]; then
            # Extract last validation metrics (simplified - adjust based on actual log format)
            epe=$(grep "val/epe" "$log_file" | tail -1 | awk '{print $NF}' 2>/dev/null || echo "N/A")
            px1=$(grep "val/1px" "$log_file" | tail -1 | awk '{print $NF}' 2>/dev/null || echo "N/A")
            px3=$(grep "val/3px" "$log_file" | tail -1 | awk '{print $NF}' 2>/dev/null || echo "N/A")
            px5=$(grep "val/5px" "$log_file" | tail -1 | awk '{print $NF}' 2>/dev/null || echo "N/A")
            
            printf "%-25s %10s %10s %10s %10s\n" "$exp_name" "$epe" "$px1" "$px3" "$px5"
        else
            printf "%-25s %10s\n" "$exp_name" "(not run yet)"
        fi
    fi
done

echo ""
echo "============================================"
echo ""
echo "For detailed comparison, use TensorBoard:"
echo "  tensorboard --logdir outputs/"
