import sys
from pathlib import Path
import argparse
import yaml
import torch
import glob
import re
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.trainer_ddflow import DDFlowTrainer
from scripts.train_ddflow import build_dataset, build_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDFlow checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--output", type=str, default="ddflow_eval_results.csv", help="Output CSV file for results")
    return parser.parse_args()

def extract_epoch(ckpt_path):
    # Try to extract epoch number from filename like ckpt_ddflow_e050.pth
    match = re.search(r'e(\d+)', Path(ckpt_path).name)
    if match:
        return int(match.group(1))
    return -1

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Setup workspace (doesn't matter much for eval, but trainer needs it)
    # We can overwrite workspace to a temp dir or keep as is
    
    print(f"Initializing DDFlow Trainer with config: {args.config}")
    trainer = DDFlowTrainer(cfg, workspace=Path(cfg["workspace"]))
    
    # Build validation loader
    print("Building validation dataset...")
    val_ds = build_dataset(cfg["data"]["val"], is_test=True)
    val_loader = build_loader(val_ds, cfg["data"]["val"])
    print(f"Validation set size: {len(val_ds)}")
    
    # Find checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(list(ckpt_dir.glob("*.pth")))
    
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    print(f"Found {len(ckpts)} checkpoints.")
    
    results = []
    
    for ckpt_path in tqdm(ckpts, desc="Evaluating checkpoints"):
        ckpt_path = str(ckpt_path)
        epoch = extract_epoch(ckpt_path)
        
        # skip best_ddflow.pth if we want to sort strictly by epoch, 
        # or include it. Let's include everything but note the name.
        is_best = "best_ddflow" in ckpt_path
        
        print(f"\nEvaluating: {ckpt_path}")
        
        # Load checkpoint
        try:
            state_dict = torch.load(ckpt_path, map_location=trainer.device)
            trainer.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
            continue
            
        # Validate
        metrics = trainer.validate(val_loader, epoch=epoch)
        
        # Record results
        row = {
            "checkpoint": Path(ckpt_path).name,
            "epoch": epoch,
            "is_best": is_best
        }
        row.update(metrics)
        results.append(row)
        
        print(f"Result: EPE={metrics.get('epe', -1):.4f}")

    # Save to CSV
    df = pd.DataFrame(results)
    # Sort by epoch
    df = df.sort_values(by="epoch")
    
    print("\nEvaluation Summary:")
    print(df.to_string(index=False))
    
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
