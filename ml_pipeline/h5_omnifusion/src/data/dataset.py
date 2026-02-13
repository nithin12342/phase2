"""DAIC-WOZ Dataset loader for pre-extracted features."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    StratifiedKFold = None

class DAICWOZFeatureDataset(Dataset):
    """
    DAIC-WOZ Dataset using pre-extracted SOTA features.
    
    Loads features from:
    - Features_SOTA_2025/audio/{pid}_audio.npy
    - Features_SOTA_2025/text/{pid}_text.npy
    - Features_SOTA_2025/video/{pid}_video.npy
    - Features_SOTA_2025/image/{pid}_image.npy (face)
    - Features_SOTA_2025/combined/{pid}_features.npz
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        labels_path: Optional[str] = None,
        feature_type: str = 'individual',  # 'individual' or 'combined'
        participant_ids: Optional[List[int]] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to DAIC-WOZ_Datasets folder
            split: 'train', 'dev', or 'test'
            labels_path: Path to labels CSV (auto-detected if None)
            feature_type: 'individual' for separate modality files, 'combined' for npz
            participant_ids: Optional list of PIDs to override split logic
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_type = feature_type
        
        self.features_dir = self.data_dir / 'Features_SOTA_2025'
        self.labels_dir = self.data_dir / 'Extended-DAIC-WOZ' / 'labels'
        
        self.labels_df = self._load_labels(labels_path)
        
        if participant_ids is not None:
            self.participant_ids = participant_ids
        else:
            self.participant_ids = self._get_split_ids(split)
            print(f"Loaded {len(self.participant_ids)} samples for {split} split")
        
        pos_count = 0
        neg_count = 0
        for pid in self.participant_ids[:min(5, len(self.participant_ids))]:
            lbls = self._get_labels(pid)
            if lbls['binary'].item() == 1:
                pos_count += 1
            else:
                neg_count += 1
            print(f"[DEBUG] PID {pid}: binary={lbls['binary'].item()}, phq={lbls['phq_score'].item():.1f}")
        
        for pid in self.participant_ids:
            lbls = self._get_labels(pid)
            if lbls['binary'].item() == 1:
                pos_count += 1
            else:
                neg_count += 1
        print(f"[DEBUG] Label distribution: {pos_count} positive, {neg_count} negative")
    
    def _load_labels(self, labels_path: Optional[str]) -> pd.DataFrame:
        """Load labels from CSV."""
        if labels_path:
            print(f"[DEBUG] Loading labels from provided path: {labels_path}")
            return pd.read_csv(labels_path)
        
        script_dir = Path(__file__).parent  # h5_omnifusion/data/ directory
        possible_paths = [
            script_dir / 'labels.csv',  # PRIORITY: repo's embedded labels file
            self.labels_dir / 'Detailed_PHQ8_Labels.csv',
            self.labels_dir / 'detailed_lables.csv',
            self.data_dir / 'Extended-DAIC-WOZ' / 'metadata_mapped.csv',
            self.data_dir / 'labels.csv',
            self.data_dir / 'train_split_Depression_AVEC2017.csv',
        ]
        
        print(f"[DEBUG] Searching for labels in:")
        for path in possible_paths:
            print(f"  - {path} (exists: {path.exists()})")
            if path.exists():
                df = pd.read_csv(path)
                print(f"[DEBUG] ✓ Found labels file: {path}")
                print(f"[DEBUG] Labels shape: {df.shape}")
                print(f"[DEBUG] Columns: {list(df.columns)}")
                for col in df.columns:
                    if 'binary' in col.lower() or 'phq' in col.lower() or 'label' in col.lower():
                        if df[col].dtype in ['int64', 'float64']:
                            print(f"[DEBUG] {col} distribution: {df[col].value_counts().to_dict()}")
                return df
        
        print("⚠️ WARNING: Labels file not found! Using dummy labels (ALL ZEROS).")
        print("⚠️ This will cause the model to learn nothing!")
        return pd.DataFrame({
            'Participant_ID': list(range(300, 500)),
            'PHQ8_Score': [0] * 200,
            'PHQ8_Binary': [0] * 200,
        })
    
    def _get_split_ids(self, split: str) -> List[int]:
        """Get participant IDs for the given split."""
        if 'Split' in self.labels_df.columns:
            split_df = self.labels_df[self.labels_df['Split'] == split]
            id_col = [c for c in split_df.columns if 'id' in c.lower() or 'participant' in c.lower()]
            if id_col and len(split_df) > 0:
                pids = split_df[id_col[0]].tolist()
                print(f"[DEBUG] Found {len(pids)} PIDs for {split} split from Split column")
                return pids
        
        split_file = self.labels_dir / f'{split}_split.csv'
        
        if split_file.exists():
            split_df = pd.read_csv(split_file)
            id_col = [c for c in split_df.columns if 'id' in c.lower() or 'participant' in c.lower()]
            if id_col:
                return split_df[id_col[0]].tolist()
        
        audio_dir = self.features_dir / 'audio'
        if audio_dir.exists():
            pids = []
            for f in audio_dir.glob('*_audio.npy'):
                try:
                    pid = int(f.stem.split('_')[0])
                    pids.append(pid)
                except:
                    pass
            
            pids = sorted(pids)
            n = len(pids)
            if split == 'train':
                return pids[:int(n * 0.7)]
            elif split == 'dev':
                return pids[int(n * 0.7):int(n * 0.85)]
            else:
                return pids[int(n * 0.85):]
        
        return list(range(300, 350))  # Dummy IDs
    
    def __len__(self) -> int:
        return len(self.participant_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        pid = self.participant_ids[idx]
        
        if self.feature_type == 'combined':
            return self._load_combined(pid)
        else:
            return self._load_individual(pid)
    
    def _load_individual(self, pid: int) -> Dict[str, torch.Tensor]:
        """Load individual modality features."""
        sample = {}
        
        audio_path = self.features_dir / 'audio' / f'{pid}_audio.npy'
        if audio_path.exists():
            sample['audio_features'] = torch.from_numpy(np.load(audio_path)).float()
        else:
            sample['audio_features'] = torch.zeros(768)
        
        text_path = self.features_dir / 'text' / f'{pid}_text.npy'
        if text_path.exists():
            sample['text_features'] = torch.from_numpy(np.load(text_path)).float()
        else:
            sample['text_features'] = torch.zeros(768)
        
        video_path = self.features_dir / 'video' / f'{pid}_video.npy'
        if video_path.exists():
            sample['video_features'] = torch.from_numpy(np.load(video_path)).float()
        else:
            sample['video_features'] = torch.zeros(768)
        
        image_path = self.features_dir / 'image' / f'{pid}_image.npy'
        if image_path.exists():
            sample['face_features'] = torch.from_numpy(np.load(image_path)).float()
        else:
            sample['face_features'] = torch.zeros(768)
        
        sample['tabular_features'] = torch.zeros(768)
        
        sample['targets'] = self._get_labels(pid)
        sample['participant_id'] = pid
        
        return sample
    
    def _load_combined(self, pid: int) -> Dict[str, torch.Tensor]:
        """Load combined features from npz."""
        combined_path = self.features_dir / 'combined' / f'{pid}_features.npz'
        
        if combined_path.exists():
            data = np.load(combined_path)
            sample = {
                'audio_features': torch.from_numpy(data.get('audio', np.zeros(768))).float(),
                'text_features': torch.from_numpy(data.get('text', np.zeros(768))).float(),
                'video_features': torch.from_numpy(data.get('video', np.zeros(768))).float(),
                'face_features': torch.from_numpy(data.get('image', np.zeros(768))).float(),
                'tabular_features': torch.zeros(768),
            }
        else:
            sample = self._load_individual(pid)
        
        sample['targets'] = self._get_labels(pid)
        sample['participant_id'] = pid
        
        return sample
    
    def _get_labels(self, pid: int) -> Dict[str, torch.Tensor]:
        """Get labels for participant."""
        id_cols = ['Participant_ID', 'participant_id', 'ID', 'id']
        
        for id_col in id_cols:
            if id_col in self.labels_df.columns:
                row = self.labels_df[self.labels_df[id_col] == pid]
                if len(row) > 0:
                    row = row.iloc[0]
                    
                    phq_cols = ['PHQ8_Score', 'PHQ_Score', 'phq8_score', 'phq_score', 'PHQ_Binary']
                    phq_score = 0
                    for col in phq_cols:
                        if col in row and not pd.isna(row[col]):
                            phq_score = float(row[col])
                            break
                    
                    binary_cols = ['PHQ8_Binary', 'PHQ_Binary', 'binary', 'label']
                    binary = 0
                    for col in binary_cols:
                        if col in row and not pd.isna(row[col]):
                            binary = int(row[col])
                            break
                    
                    if binary == 0 and phq_score >= 10:
                        binary = 1
                    
                    return {
                        'binary': torch.tensor(binary, dtype=torch.long),
                        'phq_score': torch.tensor(phq_score, dtype=torch.float),
                    }
        
        return {
            'binary': torch.tensor(0, dtype=torch.long),
            'phq_score': torch.tensor(0.0, dtype=torch.float),
        }


class DAICWOZDataset(DAICWOZFeatureDataset):
    """Alias for DAICWOZFeatureDataset for backward compatibility."""
    pass


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    result = {}
    
    for key in batch[0].keys():
        if key == 'targets':
            result['targets'] = {
                'binary': torch.stack([b['targets']['binary'] for b in batch]),
                'phq_score': torch.stack([b['targets']['phq_score'] for b in batch]),
            }
        elif key == 'participant_id':
            result['participant_id'] = [b['participant_id'] for b in batch]
        elif torch.is_tensor(batch[0][key]):
            result[key] = torch.stack([b[key] for b in batch])
    
    return result


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    folds: int = 1,
    fold_idx: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders with support for K-Fold CV.
    
    Args:
        data_dir: Dataset root
        batch_size: Batch size
        num_workers: Number of workers
        folds: Number of folds (1 = standard train/val split, >1 = K-Fold)
        fold_idx: Current fold index (0 to folds-1)
    """
    
    test_dataset = DAICWOZFeatureDataset(data_dir, split='test')
    
    train_ds_full = DAICWOZFeatureDataset(data_dir, split='train')
    dev_ds_full = DAICWOZFeatureDataset(data_dir, split='dev')
    
    all_train_pids = train_ds_full.participant_ids + dev_ds_full.participant_ids
    all_train_pids = sorted(list(set(all_train_pids)))
    
    if folds > 1 and StratifiedKFold:
        print(f"Setting up {folds}-Fold CV (Fold {fold_idx}) with {len(all_train_pids)} samples")
        
        labels = []
        valid_pids = []
        
        for pid in all_train_pids:
            lbls = train_ds_full._get_labels(pid)
            labels.append(lbls['binary'].item())
            valid_pids.append(pid)
            
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        
        for i, (train_idx, val_idx) in enumerate(skf.split(valid_pids, labels)):
            if i == fold_idx:
                train_pids = [valid_pids[j] for j in train_idx]
                val_pids = [valid_pids[j] for j in val_idx]
                break
        else:
            raise ValueError(f"Fold index {fold_idx} out of range for {folds} folds")
            
        train_dataset = DAICWOZFeatureDataset(data_dir, participant_ids=train_pids)
        val_dataset = DAICWOZFeatureDataset(data_dir, participant_ids=val_pids)
        
    else:
        if folds > 1:
            print("Warning: StratifiedKFold not available, using standard split or dummy fold")
        
        train_dataset = train_ds_full
        val_dataset = dev_ds_full
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Fold {fold_idx}/{folds}: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
