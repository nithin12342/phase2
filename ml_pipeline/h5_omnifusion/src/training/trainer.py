"""H5-OmniFusion Trainer with Focal Loss and 5-Fold CV support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random

try:
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
except ImportError:
    print("Warning: sklearn not installed, metrics won't be computed")


class FocalLossBinary(nn.Module):
    """Focal Loss for binary classification with sigmoid output (numerically stable)."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) raw logits (before sigmoid)
            targets: (B,) binary labels (0 or 1)
        """
        logits = logits.squeeze(-1).float()
        targets_float = targets.float()
        
        if self.label_smoothing > 0:
            targets_float = targets_float * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')
        
        bce_loss = torch.clamp(bce_loss, max=100.0)
        
        pt = torch.clamp(torch.exp(-bce_loss), min=1e-7, max=1.0 - 1e-7)
        
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        
        modulating_factor = torch.clamp((1.0 - pt) ** self.gamma, max=10.0)
        focal_loss = alpha_t * modulating_factor * bce_loss
        
        result = focal_loss.mean()
        if torch.isnan(result):
            return torch.tensor(1.0, device=logits.device, requires_grad=True)
        return result


class H5Trainer:
    """
    Trainer for H5-OmniFusion model.
    
    Implements:
    - Focal Loss with class weighting
    - AdamW optimizer with OneCycleLR
    - Mixed precision training
    - Checkpoint saving/resuming
    - Metric computation (F1, AUC-ROC, Accuracy)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        test_loader=None,
        device: str = 'cuda',
        criterion=None,  # Allow injecting custom loss function
        drop_modalities: list = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.drop_modalities = drop_modalities or []
        
        if criterion is not None:
            self.focal_loss = criterion
        else:
            alpha = getattr(config.loss, 'focal_alpha', 0.25)
            gamma = getattr(config.loss, 'focal_gamma', 2.0)
            smoothing = getattr(config.loss, 'label_smoothing', 0.05)
            
            self.focal_loss = FocalLossBinary(
                alpha=alpha,
                gamma=gamma,
                label_smoothing=smoothing
            )
        
        self.mse_loss = nn.MSELoss()
        
        self.lambda_cls = config.loss.lambda_cls
        self.lambda_phq = config.loss.lambda_phq
        self.lambda_orth = config.loss.lambda_orth
        
        self.threshold = getattr(config.loss, 'decision_threshold', 0.35)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps
        )
        
        total_steps = len(train_loader) * config.n_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.optimizer.lr,
            total_steps=total_steps,
            pct_start=config.scheduler.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )
        
        self.scaler = GradScaler() if config.mixed_precision else None
        self.use_amp = config.mixed_precision
        
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
    def train(self, save_path: str = "best_model.pt"):
        """Run full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.config.n_epochs} epochs")
        print(f"{'='*60}\n")
        
        latest_path = save_path.replace('_best.', '_latest.').replace('best.', 'latest.')
        if latest_path == save_path:
            latest_path = save_path.replace('.pt', '_latest.pt')
        collapse_counter = 0  # Track consecutive TN=0 epochs
        
        for epoch in range(self.current_epoch, self.config.n_epochs):
            self.current_epoch = epoch
            
            train_metrics = self._train_epoch()
            
            val_metrics = self._evaluate(self.val_loader, desc="Validating")
            
            test_metrics = {}
            if self.test_loader:
                test_metrics = self._evaluate(self.test_loader, desc="Testing")
            
            print(f"Epoch {epoch+1}/{self.config.n_epochs}")
            
            print(f"  {'Metric':<10} | {'Train':<10} | {'Val':<10} | {'Test':<10}")
            print(f"  {'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}")
            
            metrics_to_show = ['loss', 'f1', 'auc', 'accuracy', 'precision', 'recall']
            for m in metrics_to_show:
                t_val = train_metrics.get(m, 0.0)
                v_val = val_metrics.get(m, 0.0)
                test_val = test_metrics.get(m, 0.0)
                print(f"  {m.title():<10} | {t_val:<10.4f} | {v_val:<10.4f} | {test_val:<10.4f}")
            
            tp = val_metrics.get('tp', 0)
            tn = val_metrics.get('tn', 0)
            fp = val_metrics.get('fp', 0)
            fn = val_metrics.get('fn', 0)
            print(f"  Val CM: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            
            if test_metrics:
                tp_t = test_metrics.get('tp', 0)
                tn_t = test_metrics.get('tn', 0)
                fp_t = test_metrics.get('fp', 0)
                fn_t = test_metrics.get('fn', 0)
                print(f"  Test CM: TP={tp_t}, TN={tn_t}, FP={fp_t}, FN={fn_t}")
            
            if tn == 0 and fp > 0:
                collapse_counter += 1
                if collapse_counter >= 3:
                    print(f"  🚨 COLLAPSE DETECTED: TN=0 for {collapse_counter} consecutive epochs!")
                    print(f"     Model predicts ALL samples as Depressed. Consider reducing batch size.")
            else:
                collapse_counter = 0
            
            if val_metrics['recall'] < 0.2:
                print(f"  ⚠️ WARNING: Low validation recall ({val_metrics['recall']:.4f}) - model may be collapsing!")
            
            self._save_checkpoint(latest_path)
            
            val_spec = val_metrics.get('specificity', 0)
            val_j = val_metrics['recall'] + val_spec - 1  # Youden's J
            if val_j > self.best_val_f1:
                self.best_val_f1 = val_j
                self.patience_counter = 0
                self._save_checkpoint(save_path)
                print(f"  ✓ New best model saved! Val J={val_j:.4f} (Sens={val_metrics['recall']:.3f}, Spec={val_spec:.3f})")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
                
            print()
        
        print(f"\nTraining complete! Best Val J: {self.best_val_f1:.4f}")
        return self.best_val_f1
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        skipped_batches = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch+1}")
        
        for batch in pbar:
            batch = self._move_to_device(batch)
            
            self.optimizer.zero_grad()
            
            try:
                if self.use_amp:
                    with autocast():
                        loss, is_valid, outputs = self._compute_loss(batch, return_outputs=True)
                    
                    if not is_valid:
                        skipped_batches += 1
                        self.scheduler.step()
                        continue
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, is_valid, outputs = self._compute_loss(batch, return_outputs=True)
                    
                    if not is_valid:
                        skipped_batches += 1
                        self.scheduler.step()
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                
                self.scheduler.step()
                
                total_loss += loss.item()
                
                n_batches += 1
                
                probs = outputs['binary_prob'].squeeze(-1).detach()
                preds = (probs > self.threshold).long()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['targets']['binary'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'skip': skipped_batches})
                
            except RuntimeError as e:
                print(f"[Warning] Runtime error in batch: {e}")
                skipped_batches += 1
                self.scheduler.step()
                continue
        
        if skipped_batches > 0:
            print(f"[Info] Skipped {skipped_batches} batches due to NaN")
        
        avg_loss = total_loss / max(n_batches, 1)
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _compute_loss(self, batch: Dict, return_outputs: bool = False):
        """Compute composite loss with numerical stability."""
        if self.drop_modalities:
            for modality in self.drop_modalities:
                feature_key = f"{modality}_features"
                if feature_key in batch:
                    batch[feature_key] = torch.zeros_like(batch[feature_key])

        outputs, orth_loss = self.model(batch)
        
        targets = batch['targets']
        
        if not torch.isfinite(outputs['binary_logit']).all():
            print("[Warning] Non-finite values in binary_logit, skipping batch")
            ret = (torch.tensor(0.0, device=self.device), False)
            return (*ret, outputs) if return_outputs else ret
        
        cls_loss = self.focal_loss(outputs['binary_logit'], targets['binary'])
        
        phq_pred = outputs['phq_score'].squeeze()
        phq_target = targets['phq_score'] / 24.0
        phq_pred_norm = phq_pred / 24.0
        phq_loss = self.mse_loss(phq_pred_norm, phq_target)
        
        if isinstance(orth_loss, torch.Tensor) and torch.isnan(orth_loss):
            orth_loss = torch.tensor(0.0, device=self.device)
        elif not isinstance(orth_loss, torch.Tensor):
            orth_loss = torch.tensor(float(orth_loss), device=self.device)
        
        cls_loss = torch.clamp(cls_loss, max=10.0)
        phq_loss = torch.clamp(phq_loss, max=10.0)
        orth_loss = torch.clamp(orth_loss, max=1.0)
        
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_phq * phq_loss +
            self.lambda_orth * orth_loss
        )
        
        if torch.isnan(total_loss):
            print(f"[Warning] NaN total loss, skipping batch")
            ret = (torch.tensor(0.0, device=self.device), False)
            return (*ret, outputs) if return_outputs else ret
        
        ret = (total_loss, True)
        return (*ret, outputs) if return_outputs else ret

    @torch.no_grad()
    def _evaluate(self, loader, desc="Validating") -> Dict[str, float]:
        """Run evaluation and compute metrics."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(loader, desc=desc):
            batch = self._move_to_device(batch)
            
            loss, is_valid, outputs = self._compute_loss(batch, return_outputs=True)
            if is_valid:
                total_loss += loss.item()
                n_batches += 1
            
            probs = outputs['binary_prob'].squeeze(-1)  # (B,)
            preds = (probs > self.threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['targets']['binary'].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = total_loss / max(n_batches, 1)
        
        return metrics

    def _compute_metrics(self, all_labels, all_preds, all_probs):
        """Helper to calculate metrics from arrays."""
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = {
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
        }
        
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metrics['auc'] = 0.5
        
        try:
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            metrics['tp'] = int(tp)
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception:
            metrics['tp'] = metrics['tn'] = metrics['fp'] = metrics['fn'] = 0
            metrics['specificity'] = 0.0
            
        return metrics
    
    def _move_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in value.items()}
            else:
                result[key] = value
        return result
    
    def _save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_val_f1': self.best_val_f1,
            'random_states': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            }
        }
        
        if torch.cuda.is_available():
            checkpoint['random_states']['cuda'] = torch.cuda.get_rng_state_all()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def resume_from_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        new_total_steps = len(self.train_loader) * self.config.n_epochs
        if hasattr(self.scheduler, 'total_steps'):
            old_total_steps = self.scheduler.total_steps
            if new_total_steps != old_total_steps:
                print(f"[Info] Adjusting scheduler: {old_total_steps} -> {new_total_steps} total steps")
                self.scheduler.total_steps = new_total_steps
        
        if 'random_states' in checkpoint:
            try:
                rng_state = checkpoint['random_states']['torch']
                if hasattr(rng_state, 'cpu'):
                    rng_state = rng_state.cpu()
                if rng_state.dtype != torch.uint8:
                    rng_state = rng_state.to(torch.uint8)
                torch.set_rng_state(rng_state)
            except Exception as e:
                print(f"[Warning] Could not restore torch RNG state: {e}")
            
            try:
                np.random.set_state(checkpoint['random_states']['numpy'])
            except Exception as e:
                print(f"[Warning] Could not restore numpy RNG state: {e}")
            
            try:
                random.setstate(checkpoint['random_states']['python'])
            except Exception as e:
                print(f"[Warning] Could not restore python RNG state: {e}")
            
            if torch.cuda.is_available() and 'cuda' in checkpoint['random_states']:
                try:
                    cuda_states = checkpoint['random_states']['cuda']
                    cuda_states = [s.to(torch.uint8) if hasattr(s, 'to') else s for s in cuda_states]
                    torch.cuda.set_rng_state_all(cuda_states)
                except Exception as e:
                    print(f"[Warning] Could not restore CUDA RNG state: {e}")

        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {self.current_epoch}, best F1: {self.best_val_f1:.4f}")

