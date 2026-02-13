"""
H5-OmniFusion Training Script
=============================
CLI entry point for training with K-fold cross-validation
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import TrainingConfig, get_config
from src.dataset import create_dataloaders
from src.trainer import H5Trainer
from src.checkpointing import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train H5-OmniFusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--h5_dir", type=str, required=True,
                        help="Directory containing H5 files")
    parser.add_argument("--labels_csv", type=str, default=None,
                        help="Optional CSV file with labels")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold to train (if None, train all folds)")
    
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model definition (optional)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    
    return parser.parse_args()


def create_model(config: TrainingConfig, device: str = "cuda"):
    """Create H5-OmniFusion model"""
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "..", "ml_pipeline", "h5_omnifusion"
        ))
        from core_fusion_models.h5_omnifusion import H5OmniFusion
        
        model = H5OmniFusion(
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_latents=config.model.n_latents,
            n_experts=config.model.n_experts,
            n_quality_features=config.model.n_quality_features,
            dropout=config.model.encoder_dropout
        )
        return model.to(device)
    except ImportError:
        print("⚠️  Could not import H5OmniFusion model")
        print("   Create a dummy model for testing...")
        
        class DummyModel(torch.nn.Module):
            def __init__(self, d_model=256):
                super().__init__()
                self.encoder = torch.nn.Linear(768 * 5, d_model)
                self.classifier = torch.nn.Linear(d_model, 2)
                self.regressor = torch.nn.Linear(d_model, 1)
            
            def forward(self, inputs):
                x = torch.cat([
                    inputs["audio_embedding"],
                    inputs["text_embedding"],
                    inputs["video_embedding"],
                    inputs["face_embedding"],
                    inputs["tabular_embedding"],
                ], dim=-1)
                
                h = torch.relu(self.encoder(x))
                binary_logits = self.classifier(h)
                phq8_pred = self.regressor(h).squeeze(-1) * 24  # Scale to 0-24
                
                return {
                    "binary_logits": binary_logits,
                    "phq8_pred": phq8_pred,
                }, {}
        
        return DummyModel(config.model.d_model).to(device)


def train_fold(
    fold: int,
    config: TrainingConfig,
    args: argparse.Namespace
) -> dict:
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"🔄 Training Fold {fold}/{config.data.n_folds - 1}")
    print(f"{'='*60}")
    
    train_loader, val_loader, info = create_dataloaders(
        h5_dir=args.h5_dir,
        labels_csv=args.labels_csv,
        batch_size=config.data.batch_size,
        n_folds=config.data.n_folds,
        fold_idx=fold,
        num_workers=config.data.num_workers,
        seed=config.seed
    )
    
    print(f"📊 Train samples: {info['n_train']} | Val samples: {info['n_val']}")
    print(f"📊 Train positive ratio: {info['train_positive_ratio']:.2%}")
    print(f"📊 Val positive ratio: {info['val_positive_ratio']:.2%}")
    
    model = create_model(config, args.device)
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = H5Trainer(model, config, args.device)
    
    resume_path = args.resume
    if resume_path is None:
        resume_path = trainer.checkpoint_manager.find_latest(fold)
    
    results = trainer.train(train_loader, val_loader, fold, resume_path)
    
    return results


def main():
    args = parse_args()
    
    config = get_config(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    config.optimizer.learning_rate = args.lr
    config.optimizer.weight_decay = args.weight_decay
    config.data.h5_dir = args.h5_dir
    config.data.labels_csv = args.labels_csv
    config.data.batch_size = args.batch_size
    config.data.n_folds = args.n_folds
    config.checkpoint.save_dir = args.output_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print(f"📝 Saved config to {config_path}")
    
    set_seed(config.seed)
    
    if args.fold is not None:
        results = train_fold(args.fold, config, args)
        all_results = {args.fold: results}
    else:
        all_results = {}
        for fold in range(config.data.n_folds):
            results = train_fold(fold, config, args)
            all_results[fold] = results
    
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    
    metrics_names = ["f1", "auc_roc", "accuracy", "phq8_mae"]
    for metric in metrics_names:
        values = [r["best_metrics"].get(metric, 0) for r in all_results.values()]
        mean_val = sum(values) / len(values)
        std_val = (sum((v - mean_val)**2 for v in values) / len(values)) ** 0.5
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            k: {
                "best_metrics": v["best_metrics"],
                "best_epoch": v["best_epoch"]
            }
            for k, v in all_results.items()
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")


if __name__ == "__main__":
    main()
