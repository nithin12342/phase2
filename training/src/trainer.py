"""
H5-OmniFusion Trainer Module
============================
Main training loop with curriculum phases
"""

import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, Optional, Any, Tuple
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import TrainingConfig, Phase
from src.losses import CompositeLoss, mixup_data, mixup_criterion
from src.metrics import MetricsTracker, format_metrics, check_targets_achieved
from src.checkpointing import CheckpointManager, EarlyStopping, set_seed


class H5Trainer:
    """
    Main trainer for H5-OmniFusion model.
    Implements 4-phase curriculum training with all strategy specifications.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = self._create_optimizer()
        
        self.scheduler = None
        
        self.criterion = CompositeLoss(
            lambda_cls=config.loss.lambda_cls,
            lambda_phq=config.loss.lambda_phq,
            lambda_orth=config.loss.lambda_orth,
            focal_alpha=config.loss.focal.alpha,
            focal_gamma=config.loss.focal.gamma,
            label_smoothing=config.loss.label_smoothing
        )
        
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.checkpoint.save_dir,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_best_only=config.checkpoint.save_best_only,
            save_every_epoch=config.checkpoint.save_every_epoch
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            monitor=config.early_stopping.monitor,
            mode=config.early_stopping.mode
        )
        
        self.scaler = GradScaler() if config.mixed_precision else None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer.name == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.name}")
    
    def _create_scheduler(self, total_steps: int):
        """Create OneCycleLR scheduler"""
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.scheduler.max_lr,
            total_steps=total_steps,
            pct_start=self.config.scheduler.pct_start,
            anneal_strategy=self.config.scheduler.anneal_strategy,
            div_factor=self.config.scheduler.div_factor,
            final_div_factor=self.config.scheduler.final_div_factor
        )
    
    def train_epoch(
        self,
        train_loader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        loss_components = {"loss_cls": 0, "loss_phq": 0, "loss_orth": 0}
        
        phase = self.config.get_current_phase(epoch)
        use_mixup = self.config.regularization.mixup_prob > 0 and phase != Phase.WARMUP
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            batch = self._to_device(batch)
            
            inputs = {
                "audio_embedding": batch["audio_embedding"],
                "text_embedding": batch["text_embedding"],
                "video_embedding": batch["video_embedding"],
                "face_embedding": batch["face_embedding"],
                "tabular_embedding": batch["tabular_embedding"],
                "quality_scores": batch["quality_scores"],
            }
            targets = {
                "binary_label": batch["binary_label"],
                "phq8_score": batch["phq8_score"],
            }
            
            if use_mixup and torch.rand(1).item() < self.config.regularization.mixup_prob:
                inputs, mixed_targets, lam = mixup_data(
                    inputs, targets, self.config.regularization.mixup_alpha
                )
                use_mixup_loss = True
            else:
                use_mixup_loss = False
                lam = 1.0
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs, aux = self.model(inputs)
                    
                    if use_mixup_loss:
                        loss, components = mixup_criterion(
                            self.criterion, outputs, mixed_targets, lam
                        )
                    else:
                        loss, components = self.criterion(outputs, targets)
                
                loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.regularization.gradient_clip_norm
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.global_step += 1
            else:
                outputs, aux = self.model(inputs)
                
                if use_mixup_loss:
                    loss, components = mixup_criterion(
                        self.criterion, outputs, mixed_targets, lam
                    )
                else:
                    loss, components = self.criterion(outputs, targets)
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.regularization.gradient_clip_norm
                    )
                    self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.global_step += 1
            
            total_loss += components["loss_total"]
            for k, v in components.items():
                if k in loss_components:
                    loss_components[k] += v
            
            self.train_metrics.update(
                outputs["binary_logits"].detach(),
                targets["binary_label"],
                outputs["phq8_pred"].detach(),
                targets["phq8_score"]
            )
            
            pbar.set_postfix({
                "loss": f"{components['loss_total']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        metrics = self.train_metrics.compute()
        metrics["loss"] = total_loss / len(train_loader)
        for k, v in loss_components.items():
            metrics[k] = v / len(train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0
        
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        for batch in pbar:
            batch = self._to_device(batch)
            
            inputs = {
                "audio_embedding": batch["audio_embedding"],
                "text_embedding": batch["text_embedding"],
                "video_embedding": batch["video_embedding"],
                "face_embedding": batch["face_embedding"],
                "tabular_embedding": batch["tabular_embedding"],
                "quality_scores": batch["quality_scores"],
            }
            targets = {
                "binary_label": batch["binary_label"],
                "phq8_score": batch["phq8_score"],
            }
            
            if self.scaler:
                with autocast():
                    outputs, aux = self.model(inputs)
                    loss, components = self.criterion(outputs, targets)
            else:
                outputs, aux = self.model(inputs)
                loss, components = self.criterion(outputs, targets)
            
            total_loss += components["loss_total"]
            
            self.val_metrics.update(
                outputs["binary_logits"],
                targets["binary_label"],
                outputs["phq8_pred"],
                targets["phq8_score"]
            )
        
        metrics = self.val_metrics.compute()
        metrics["loss"] = total_loss / len(val_loader)
        
        return metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        fold: int = 0,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Dict with best metrics and training history
        """
        set_seed(self.config.seed)
        
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.epochs
        self._create_scheduler(total_steps)
        
        start_epoch = 1
        if resume_from:
            info = self.checkpoint_manager.load(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            start_epoch = info["epoch"] + 1
            print(f"📂 Resuming from epoch {start_epoch}")
        
        history = {"train": [], "val": []}
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting training: Fold {fold}, Epochs {start_epoch}-{self.config.epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, self.config.epochs + 1):
            self.current_epoch = epoch
            phase = self.config.get_current_phase(epoch)
            
            print(f"\n📍 Epoch {epoch}/{self.config.epochs} | Phase: {phase.name}")
            print("-" * 50)
            
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train"].append(train_metrics)
            
            val_metrics = self.validate(val_loader)
            history["val"].append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n📊 Train: {format_metrics(train_metrics)}")
            print(f"📊 Val:   {format_metrics(val_metrics)}")
            print(f"⏱️  Epoch time: {epoch_time:.1f}s")
            
            target_status = check_targets_achieved(val_metrics, self.config.targets)
            if all(target_status.values()):
                print("\n🎉 ALL TARGETS ACHIEVED!")
            
            is_best = self.checkpoint_manager.is_better(val_metrics.get("f1", 0))
            self.checkpoint_manager.save(
                epoch, self.model, self.optimizer, self.scheduler,
                val_metrics, self.config, fold, is_best
            )
            
            if is_best:
                self.best_metrics = val_metrics.copy()
                print(f"🏆 New best F1: {val_metrics['f1']:.4f}")
            
            if self.early_stopping(val_metrics):
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break
            
            if phase == Phase.WARMUP and epoch == 5:
                if train_metrics["loss"] > 10:
                    print("⚠️  Warning: High loss after warmup phase")
            
            elif phase == Phase.INITIAL and epoch == 10:
                if val_metrics["f1"] < 0.50:
                    print("⚠️  Warning: F1 < 0.50 at epoch 10, consider adjusting hyperparameters")
        
        print(f"\n{'='*60}")
        print("✅ Training Complete!")
        print(f"Best validation F1: {self.best_metrics.get('f1', 0):.4f}")
        print(f"{'='*60}\n")
        
        return {
            "best_metrics": self.best_metrics,
            "history": history,
            "best_epoch": self.checkpoint_manager.best_epoch
        }
    
    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
