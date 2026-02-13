"""
Tabular Preprocessing Module
Implements Steps 35-40 and R50-R53 from H5-OmniFusion specification.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import DEVICE, ensure_768_dim, safe_embedding


class MissingValueImputer:
    """
    Impute missing values using median (numeric) or mode (categorical).
    Steps 35, R50.
    """
    
    def __init__(self):
        self.numeric_medians = {}
        self.categorical_modes = {}
    
    def fit(self, df: pd.DataFrame, numeric_cols: List[str] = None, 
            categorical_cols: List[str] = None) -> 'MissingValueImputer':
        """Learn imputation values from data."""
        if numeric_cols:
            for col in numeric_cols:
                if col in df.columns:
                    self.numeric_medians[col] = df[col].median()
        
        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    self.categorical_modes[col] = df[col].mode().iloc[0] if not df[col].mode().empty else ''
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation."""
        df = df.copy()
        
        for col, median in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        
        for col, mode in self.categorical_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str] = None,
                      categorical_cols: List[str] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, numeric_cols, categorical_cols)
        return self.transform(df)


class CategoricalEncoder:
    """
    Encode categorical variables using one-hot or label encoding.
    Steps 36, R51.
    """
    
    def __init__(self, method: str = 'onehot'):
        self.method = method
        self.encodings = {}
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'CategoricalEncoder':
        """Learn encodings from data."""
        for col in columns:
            if col in df.columns:
                unique_values = df[col].unique().tolist()
                self.encodings[col] = {v: i for i, v in enumerate(unique_values)}
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding."""
        df = df.copy()
        
        for col, encoding in self.encodings.items():
            if col not in df.columns:
                continue
            
            if self.method == 'onehot':
                for val in encoding.keys():
                    df[f'{col}_{val}'] = (df[col] == val).astype(int)
                df = df.drop(columns=[col])
            else:
                df[col] = df[col].map(encoding).fillna(-1).astype(int)
        
        return df


class NumericalNormalizer:
    """
    Normalize numerical features using Z-score or min-max.
    Steps 37, R52.
    """
    
    def __init__(self, method: str = 'zscore'):
        self.method = method
        self.stats = {}
    
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'NumericalNormalizer':
        """Learn normalization statistics."""
        for col in columns:
            if col in df.columns:
                if self.method == 'zscore':
                    self.stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std() + 1e-8
                    }
                else:  # minmax
                    self.stats[col] = {
                        'min': df[col].min(),
                        'max': df[col].max() + 1e-8
                    }
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization."""
        df = df.copy()
        
        for col, stats in self.stats.items():
            if col not in df.columns:
                continue
            
            if self.method == 'zscore':
                df[col] = (df[col] - stats['mean']) / stats['std']
            else:
                df[col] = (df[col] - stats['min']) / (stats['max'] - stats['min'])
        
        return df


class TabularProjector(nn.Module):
    """
    Project tabular features to 768-dim using MLP.
    Steps 38, R53.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 768):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)
    
    def project(self, features: np.ndarray) -> np.ndarray:
        """Project numpy features to 768-dim."""
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            projected = self.forward(tensor)
        return safe_embedding(projected.cpu().numpy().squeeze())


class ClinicalFeatureEngineer:
    """
    Engineer PHQ-8 sub-score features.
    Step 39.
    
    PHQ-8 Subscales:
    - Somatic: Items 3, 4, 5 (sleep, fatigue, appetite)
    - Cognitive: Items 1, 2, 6, 7, 8 (anhedonia, depression, guilt, concentration, psychomotor)
    """
    
    PHQ8_SOMATIC = [3, 4, 5]
    PHQ8_COGNITIVE = [1, 2, 6, 7, 8]
    
    def engineer(self, phq8_scores: Dict[int, int]) -> Dict:
        """
        Calculate PHQ-8 subscale scores.
        
        Args:
            phq8_scores: Dict mapping item number (1-8) to score (0-3)
            
        Returns:
            Dict with somatic_score, cognitive_score, total
        """
        somatic = sum(phq8_scores.get(i, 0) for i in self.PHQ8_SOMATIC)
        cognitive = sum(phq8_scores.get(i, 0) for i in self.PHQ8_COGNITIVE)
        total = sum(phq8_scores.values())
        
        return {
            'somatic_score': somatic,
            'cognitive_score': cognitive,
            'total_score': total,
            'somatic_ratio': somatic / (total + 1e-8),
            'cognitive_ratio': cognitive / (total + 1e-8)
        }


class QualityScorer:
    """
    Calculate quality scores for multimodal data.
    Steps 40, R59.
    """
    
    def __init__(self, config=None):
        cfg = config or CFG
        self.snr_min = cfg.AUDIO_SNR_MIN_DB
        self.clip_max = cfg.AUDIO_CLIPPING_MAX_RATIO
        self.vad_min = cfg.AUDIO_VAD_MIN_RATIO
        self.face_min = cfg.VIDEO_FACE_DETECTION_MIN_RATIO
    
    def score_audio(self, snr: float, clipping: float, vad_ratio: float) -> float:
        """Calculate audio quality score [0, 1]."""
        snr_score = 1.0 / (1.0 + np.exp(-(snr - self.snr_min) / 5.0))
        clip_score = max(0.0, 1.0 - clipping / self.clip_max)
        vad_score = min(1.0, vad_ratio / self.vad_min)
        
        return (snr_score + clip_score + vad_score) / 3.0
    
    def score_video(self, brightness_ok: float, blur_ok: float, 
                    face_detection_rate: float) -> float:
        """Calculate video quality score [0, 1]."""
        face_score = min(1.0, face_detection_rate / self.face_min)
        return (brightness_ok + blur_ok + face_score) / 3.0
    
    def score_multimodal(self, audio_quality: float, video_quality: float,
                         text_length: int = 100) -> Dict:
        """
        Calculate overall multimodal quality.
        
        Returns:
            Dict with individual and combined scores
        """
        text_score = min(1.0, text_length / 100)  # >100 words = good
        
        combined = (audio_quality + video_quality + text_score) / 3.0
        
        return {
            'audio_quality': audio_quality,
            'video_quality': video_quality,
            'text_quality': text_score,
            'overall_quality': combined
        }


class TabularPreprocessor:
    """Unified tabular preprocessing (Steps 35-40, R50-R53)."""
    
    def __init__(self, input_dim: int = 50):
        self.imputer = MissingValueImputer()
        self.encoder = CategoricalEncoder()
        self.normalizer = NumericalNormalizer()
        self.projector = TabularProjector(input_dim).to(DEVICE)
        self.clinical = ClinicalFeatureEngineer()
        self.quality = QualityScorer()
    
    def process(self, df: pd.DataFrame = None, features: np.ndarray = None,
                numeric_cols: List[str] = None, categorical_cols: List[str] = None) -> Dict:
        """
        Process tabular data.
        
        Returns:
            Dict with processed features and 768-dim embedding
        """
        if df is not None:
            df = self.imputer.fit_transform(df, numeric_cols, categorical_cols)
            
            if categorical_cols:
                df = self.encoder.fit(df, categorical_cols).transform(df)
            
            if numeric_cols:
                df = self.normalizer.fit(df, numeric_cols).transform(df)
            
            features = df.values.astype(np.float32)
        
        if features is None:
            return {'success': False, 'tabular_embedding': np.zeros(768)}
        
        if len(features.flatten()) != self.projector.projector[0].in_features:
            self.projector = TabularProjector(len(features.flatten())).to(DEVICE)
        
        embedding = self.projector.project(features.flatten())
        
        return {
            'success': True,
            'tabular_embedding': embedding,
            'feature_count': len(features.flatten())
        }
