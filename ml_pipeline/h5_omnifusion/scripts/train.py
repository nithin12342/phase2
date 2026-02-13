import argparse
import torch
import numpy as np
import random
from pathlib import Path
import sys
import os

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
h5_omnifusion_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(h5_omnifusion_root))

from config.model_config import H5Config, ComputeTier
from config.training_config import TrainingConfig
from src.models.h5_omnifusion import H5OmniFusion
from src.models.h5_omnifusion import H5OmniFusion
from src.training.trainer import H5Trainer
from src.data.h5_dataset import H5OmniFusionDataset, h5_collate_fn, create_h5_dataloaders_kfold
from src.data.augmentation import DataAugmentation


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Train H5-OmniFusion")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to H5_OmniFusion_Output directory with .h5 files")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to merged_labels.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    parser.add_argument("--tier", type=str, default="nano", choices=["nano", "micro", "medium"], help="Compute tier")
    parser.add_argument("--d_model", type=int, help="Override d_model size")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers (0 for Colab)")
    
    parser.add_argument("--folds", type=int, default=5, help="Total number of folds for K-Fold CV")
    parser.add_argument("--fold", "--fold_idx", type=int, default=0, dest="fold", help="Current fold index (0 to folds-1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--drop_modalities", type=str, nargs='*', default=[], 
                        help="List of modalities to zero out (ablation): audio, text, video, face, tabular")

    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tier_map = {
        'nano': ComputeTier.NANO,
        'micro': ComputeTier.MICRO,
        'medium': ComputeTier.MEDIUM
    }
    tier = tier_map.get(args.tier, ComputeTier.NANO)
    
    config = H5Config.from_tier(tier)
    if args.d_model:
        config.d_model = args.d_model
        
    train_config = TrainingConfig()
    
    train_config.loss.focal_alpha = 0.65 # Balanced focus on minority class
    train_config.loss.focal_gamma = 2.0 # Standard focusing parameter
    train_config.loss.label_smoothing = 0.05
    train_config.loss.decision_threshold = 0.45 # Closer to balanced threshold
    
    train_config.n_epochs = args.epochs
    train_config.batch_size = args.batch_size
    train_config.optimizer.lr = args.lr
    train_config.num_workers = args.num_workers
    train_config.seed = args.seed
    
    print(f"\nConfiguration:")
    print(f"  Tier: {args.tier}")
    print(f"  Dimension: {config.d_model}")
    print(f"  Params: Audio={config.audio.backbone}, Text={config.text.backbone}")
    print(f"  Folds: {args.folds} (Current: {args.fold})")
    print(f"  Labels CSV: {args.labels_csv}")
    
    print("\nInitializing model...")
    model = H5OmniFusion(config)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    print(f"\nLoading data from {args.data_dir}...")
    print(f"  Using labels from {args.labels_csv}")
    try:
        train_loader, val_loader, test_loader = create_h5_dataloaders_kfold(
            h5_dir=args.data_dir,
            labels_csv=args.labels_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_folds=args.folds,
            fold_idx=args.fold,
            seed=args.seed
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        import traceback
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return

    
    trainer = H5Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_config,
        device=device,
        criterion=None,  # Will be set below
        drop_modalities=args.drop_modalities
    )
    
    from src.training.losses import FocalLoss
    trainer.criterion = FocalLoss(
        alpha=0.65,  # Balanced focus
        gamma=2.0    # Standard focus
    )
    print(f"  Loss Function: FocalLoss(alpha=0.65, gamma=2.0)")
    print(f"  Decision Threshold: {train_config.loss.decision_threshold}")
    
    if args.resume:
        import os
        if not os.path.exists(args.resume):
            print(f"⚠️ Checkpoint not found: {args.resume}")
            print("   Starting training from scratch...")
        elif f"_{args.tier}_" not in args.resume:
            print(f"⚠️ Checkpoint tier mismatch! Checkpoint is for different tier.")
            print(f"   Expected tier '{args.tier}' in checkpoint name.")
            print("   Starting training from scratch...")
        else:
            trainer.resume_from_checkpoint(args.resume)
    
    print("\nStarting training...")
    save_path = output_dir / f"h5_omnifusion_{args.tier}_fold{args.fold}_best.pt"
    trainer.train(save_path=str(save_path))
    
    print(f"\nTraining complete! Best model saved to {save_path}")

if __name__ == "__main__":
    main()
