"""
H5-OmniFusion: Quick Fix Implementation
Copy-paste these solutions into your training code

PRIORITY ORDER:
1. Fix data loading (CRITICAL)
2. Add monitoring
3. Reduce model size
4. Tune hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
import copy


def create_balanced_dataloader(dataset, batch_size=8, num_workers=4):
    """
    Creates a DataLoader with proper class balancing
    
    Args:
        dataset: Your PyTorch Dataset with 'label' field
        batch_size: Batch size (keep at 8 for small dataset)
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader with balanced sampling
    """
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]['label'])  # Adjust key if different
    
    class_counts = Counter(labels)
    print(f"Class distribution: {dict(class_counts)}")
    
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    class_weights = {
        cls: total_samples / (num_classes * count)
        for cls, count in class_counts.items()
    }
    
    print(f"Class weights: {class_weights}")
    
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # CRITICAL: Must be True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


class ImprovedFocalLoss(nn.Module):
    """
    Focal Loss optimized for your imbalance level
    Alpha=0.3 and Gamma=2.0 are research-backed for 70:30 imbalance
    """
    def __init__(self, alpha=0.3, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits or probabilities)
            targets: Ground truth labels (0 or 1)
        """
        if not torch.is_floating_point(inputs):
            inputs = inputs.float()
        
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        ce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
        
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BatchMonitor:
    """
    Monitor batch statistics during training
    Alerts if no positive samples appear
    """
    def __init__(self):
        self.batch_stats = []
        
    def log_batch(self, batch_idx, targets):
        """Call this for every batch during training"""
        num_positive = targets.sum().item()
        batch_size = len(targets)
        positive_ratio = num_positive / batch_size
        
        self.batch_stats.append({
            'batch_idx': batch_idx,
            'positive_count': num_positive,
            'positive_ratio': positive_ratio
        })
        
        if num_positive == 0:
            print(f"⚠️ WARNING: Batch {batch_idx} has NO positive samples!")
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: {num_positive}/{batch_size} positive "
                  f"({positive_ratio:.1%})")
    
    def get_summary(self):
        """Get summary statistics"""
        if not self.batch_stats:
            return "No batches logged"
        
        avg_positive_ratio = np.mean([s['positive_ratio'] for s in self.batch_stats])
        batches_with_zero = sum(1 for s in self.batch_stats if s['positive_count'] == 0)
        
        return {
            'total_batches': len(self.batch_stats),
            'avg_positive_ratio': avg_positive_ratio,
            'batches_with_no_positives': batches_with_zero,
            'expected_positive_ratio': 0.30  # Your dataset ratio
        }


def find_optimal_threshold(model, dataloader, device, metric='f1'):
    """
    Find optimal classification threshold
    Based on GHOST algorithm (Esposito et al., 2021)
    
    Args:
        model: Your trained model
        dataloader: Validation DataLoader
        device: torch.device
        metric: 'f1', 'g-mean', or 'balanced_accuracy'
    
    Returns:
        best_threshold, best_score
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    print("Finding optimal threshold...")
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels']
            
            outputs = model(features)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_score = 0
    best_threshold = 0.5
    threshold_scores = []
    
    for threshold in np.arange(0.05, 0.51, 0.01):
        predictions = (all_probs >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(all_labels, predictions, zero_division=0)
        elif metric == 'g-mean':
            try:
                tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = np.sqrt(sensitivity * specificity)
            except:
                score = 0
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(all_labels, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        threshold_scores.append((threshold, score))
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    threshold_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 thresholds for {metric}:")
    for thresh, score in threshold_scores[:5]:
        print(f"  Threshold: {thresh:.3f}, Score: {score:.4f}")
    
    print(f"\nOptimal threshold: {best_threshold:.3f}")
    print(f"Best {metric}: {best_score:.4f}")
    
    return best_threshold, best_score


class ImprovedEarlyStopping:
    """
    Early stopping optimized for small imbalanced datasets
    """
    def __init__(self, patience=15, min_epochs=30, min_delta=0.001, mode='max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_epochs: Minimum epochs before early stopping can trigger
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like F1, 'min' for loss
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.epochs_no_improve = 0
        self.best_epoch = 0
        self.best_model_state = None
        
    def __call__(self, epoch, score, model):
        """
        Returns True if should stop, False otherwise
        """
        if epoch < self.min_epochs:
            if self._is_improvement(score):
                self.best_score = score
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.epochs_no_improve = 0
            print(f"✓ New best score: {score:.4f} at epoch {epoch}")
        else:
            self.epochs_no_improve += 1
            print(f"No improvement for {self.epochs_no_improve} epochs "
                  f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.epochs_no_improve >= self.patience:
                print(f"\n⛔ Early stopping triggered at epoch {epoch}")
                print(f"Loading best model from epoch {self.best_epoch}")
                return True
        
        return False
    
    def _is_improvement(self, score):
        """Check if score is an improvement"""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:  # min
            return score < self.best_score - self.min_delta
    
    def load_best_model(self, model):
        """Load the best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"Loaded best model from epoch {self.best_epoch}")


class TrainingHealthChecker:
    """
    Monitor training health and detect issues early
    """
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def log(self, **kwargs):
        """Log metrics"""
        for key, value in kwargs.items():
            self.metrics_history[key].append(value)
    
    def check_health(self, epoch):
        """
        Check if training is healthy
        Returns list of warnings
        """
        warnings = []
        
        if len(self.metrics_history['train_loss']) >= 10:
            recent_losses = self.metrics_history['train_loss'][-10:]
            early_avg = np.mean(recent_losses[:5])
            late_avg = np.mean(recent_losses[5:])
            
            if late_avg >= early_avg * 0.95:  # Not decreasing by at least 5%
                warnings.append("⚠️ Training loss not decreasing")
        
        if len(self.metrics_history['val_f1']) >= 5:
            if all(f1 == 0 for f1 in self.metrics_history['val_f1'][-5:]):
                warnings.append("🚨 CRITICAL: F1 score stuck at 0 for 5 epochs")
        
        if 'positive_samples_seen' in self.metrics_history:
            recent_pos = self.metrics_history['positive_samples_seen'][-1]
            if recent_pos == 0:
                warnings.append("🚨 CRITICAL: No positive samples in last epoch")
        
        if len(self.metrics_history['val_loss']) >= 3:
            recent_val_losses = self.metrics_history['val_loss'][-3:]
            if any(loss > 10 for loss in recent_val_losses):
                warnings.append("⚠️ Validation loss exploding")
        
        if warnings:
            print(f"\n{'='*60}")
            print(f"HEALTH CHECK - EPOCH {epoch}")
            print(f"{'='*60}")
            for warning in warnings:
                print(warning)
            print(f"{'='*60}\n")
        
        return warnings


def train_with_all_fixes(model, train_dataset, val_dataset, config, device):
    """
    Complete training function with all fixes integrated
    
    Args:
        model: Your H5-OmniFusion model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration object with hyperparameters
        device: torch.device
    
    Returns:
        trained_model, optimal_threshold, training_history
    """
    print("="*70)
    print("TRAINING WITH ALL FIXES APPLIED")
    print("="*70)
    
    print("\n[1/7] Creating balanced data loaders...")
    train_loader = create_balanced_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print("\n[2/7] Initializing improved components...")
    criterion = ImprovedFocalLoss(alpha=0.3, gamma=2.0)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01  # L2 regularization
    )
    
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    early_stopping = ImprovedEarlyStopping(
        patience=15,
        min_epochs=30
    )
    
    print("\n[3/7] Initializing monitoring...")
    batch_monitor = BatchMonitor()
    health_checker = TrainingHealthChecker()
    
    print(f"\n[4/7] Starting training for {config.epochs} epochs...")
    print("="*70)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        positive_samples_seen = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            batch_monitor.log_batch(batch_idx, labels)
            positive_samples_seen += labels.sum().item()
            
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            loss = criterion(probs, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        model.eval()
        val_loss = 0
        all_val_probs = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['labels']
                
                outputs = model(features)
                probs = torch.sigmoid(outputs)
                
                loss = criterion(probs.cpu(), labels.float())
                val_loss += loss.item()
                
                all_val_probs.extend(probs.cpu().numpy().flatten())
                all_val_labels.extend(labels.numpy().flatten())
        
        val_preds = (np.array(all_val_probs) >= 0.5).astype(int)
        val_f1 = f1_score(all_val_labels, val_preds, zero_division=0)
        val_auc = roc_auc_score(all_val_labels, all_val_probs) if len(set(all_val_labels)) > 1 else 0.5
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        health_checker.log(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_f1=val_f1,
            val_auc=val_auc,
            positive_samples_seen=positive_samples_seen
        )
        
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val F1: {val_f1:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        print(f"  Positive samples seen: {positive_samples_seen}")
        
        health_checker.check_health(epoch)
        
        if early_stopping(epoch, val_f1, model):
            early_stopping.load_best_model(model)
            break
    
    print("\n[5/7] Finding optimal threshold...")
    optimal_threshold, best_f1 = find_optimal_threshold(
        model, val_loader, device, metric='f1'
    )
    
    print("\n[6/7] Final evaluation with optimal threshold...")
    model.eval()
    final_probs = []
    final_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['labels']
            
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            
            final_probs.extend(probs.cpu().numpy().flatten())
            final_labels.extend(labels.numpy().flatten())
    
    final_preds = (np.array(final_probs) >= optimal_threshold).astype(int)
    final_f1 = f1_score(final_labels, final_preds)
    final_auc = roc_auc_score(final_labels, final_probs)
    
    print(f"\nFinal Results:")
    print(f"  F1 Score: {final_f1:.4f}")
    print(f"  AUC Score: {final_auc:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    
    print("\n[7/7] Batch statistics summary:")
    batch_summary = batch_monitor.get_summary()
    print(f"  Total batches processed: {batch_summary['total_batches']}")
    print(f"  Average positive ratio: {batch_summary['avg_positive_ratio']:.1%}")
    print(f"  Expected positive ratio: {batch_summary['expected_positive_ratio']:.1%}")
    print(f"  Batches with no positives: {batch_summary['batches_with_no_positives']}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return model, optimal_threshold, health_checker.metrics_history


if __name__ == "__main__":
    """
    Example usage - adapt to your code structure
    """
    
    class Config:
        batch_size = 8
        learning_rate = 1e-4
        epochs = 50
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    
    print("\n✅ All fixes ready to use!")
    print("Copy the functions you need into your training code.")
    print("\nPRIORITY ORDER:")
    print("1. create_balanced_dataloader() - FIX DATA LOADING")
    print("2. BatchMonitor - ADD MONITORING")
    print("3. ImprovedFocalLoss - BETTER LOSS FUNCTION")
    print("4. find_optimal_threshold() - OPTIMIZE THRESHOLD")
    print("5. train_with_all_fixes() - COMPLETE INTEGRATION")
