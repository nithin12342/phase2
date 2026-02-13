
import argparse
import torch
import numpy as np
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
from src.training.trainer import H5Trainer
from src.data.h5_dataset import create_h5_dataloaders_kfold
from src.training.ensemble import EnsembleTrainer
from src.utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser(description="Train H5-OmniFusion Ensemble")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="ensemble_checkpoints")
    parser.add_argument("--tier", type=str, default="nano", choices=["nano"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tier_map = {'nano': ComputeTier.NANO}
    config = H5Config.from_tier(tier_map.get(args.tier, ComputeTier.NANO))
    
    train_config = TrainingConfig()
    train_config.loss.focal_alpha = 0.25
    train_config.loss.focal_gamma = 2.0
    train_config.loss.label_smoothing = 0.05
    train_config.n_epochs = args.epochs
    train_config.batch_size = args.batch_size
    train_config.optimizer.lr = args.lr
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    def model_factory():
        return H5OmniFusion(config)
    
    def trainer_factory(model):
        pass 
    
    
    def train_loader_factory(seed):
        train, _, _ = create_h5_dataloaders_kfold(
            h5_dir=args.data_dir,
            labels_csv=args.labels_csv,
            batch_size=args.batch_size,
            num_workers=0,
            n_folds=5,
            fold_idx=args.fold,
            seed=seed # Different seed for shuffling/splitting
        )
        return train
        
    _, val_loader, _ = create_h5_dataloaders_kfold(
        h5_dir=args.data_dir,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        num_workers=0,
        n_folds=5,
        fold_idx=args.fold,
        seed=42
    )

    pass

    
    ensemble = EnsembleTrainer(
        model_factory=model_factory,
        trainer_factory=None, # UNUSED effectively if we override
        seeds=args.seeds,
        output_dir=str(output_path)
    )
    
    
    
    
    

    print(f"Start Ensemble Training for Fold {args.fold} (Nano)")
    print(f"Seeds: {args.seeds}")
    
    models = []
    results = []
    
    for seed in args.seeds:
        print(f"\nTraining Seed {seed}...")
        set_seed(seed)
        
        train_loader, val_loader_seed, _ = create_h5_dataloaders_kfold(
            h5_dir=args.data_dir,
            labels_csv=args.labels_csv,
            batch_size=args.batch_size,
            num_workers=0,
            n_folds=5,
            fold_idx=args.fold,
            seed=seed
        )
        
        model = H5OmniFusion(config)
        
        trainer = H5Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader_seed, # Use seeded val loader or fixed? 
            
            config=train_config,
            device=device
        )
        
        
        t_loader, v_loader, _ = create_h5_dataloaders_kfold(
            h5_dir=args.data_dir,
            labels_csv=args.labels_csv,
            batch_size=args.batch_size,
            num_workers=0,
            n_folds=5,
            fold_idx=args.fold,
            seed=42 # FIXED partition
        )
        
        
        trainer.train_loader = t_loader
        trainer.val_loader = v_loader
        
        save_name = output_path / f"nano_fold{args.fold}_seed{seed}.pt"
        best_f1 = trainer.train(save_path=str(save_name))
        
        results.append({'seed': seed, 'f1': best_f1, 'path': str(save_name)})
        
    print("\nEnsemble Results:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
