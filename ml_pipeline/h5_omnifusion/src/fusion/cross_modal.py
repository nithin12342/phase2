"""
Cross-Modal Fusion Module
Implements R54-R59 from H5-OmniFusion specification.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import DEVICE, ensure_768_dim, safe_embedding


class TemporalAligner:
    """
    Align multimodal features to common temporal grid.
    Step R54.
    """
    
    def __init__(self, bin_size_sec: float = 0.5):
        self.bin_size = bin_size_sec
    
    def align(self, features: Dict[str, np.ndarray], 
              timestamps: Dict[str, np.ndarray],
              duration: float) -> Dict[str, np.ndarray]:
        """
        Align features to temporal bins.
        
        Args:
            features: Dict mapping modality -> feature array
            timestamps: Dict mapping modality -> timestamp array
            duration: Total duration in seconds
            
        Returns:
            Dict with aligned features per modality
        """
        num_bins = int(np.ceil(duration / self.bin_size))
        aligned = {}
        
        for modality, feat in features.items():
            ts = timestamps.get(modality)
            
            if ts is None or len(feat) == 0:
                aligned[modality] = np.tile(np.mean(feat, axis=0, keepdims=True), (num_bins, 1))
                continue
            
            bin_features = [[] for _ in range(num_bins)]
            
            for i, t in enumerate(ts):
                bin_idx = min(int(t / self.bin_size), num_bins - 1)
                if i < len(feat):
                    bin_features[bin_idx].append(feat[i])
            
            aligned_feat = []
            for bin_feat in bin_features:
                if bin_feat:
                    aligned_feat.append(np.mean(bin_feat, axis=0))
                else:
                    aligned_feat.append(np.zeros_like(feat[0]))
            
            aligned[modality] = np.array(aligned_feat)
        
        return aligned


class QualityGatedFusion(nn.Module):
    """
    Fuse modalities weighted by quality scores.
    ADV8.
    
    weighted = Σ(quality_i × embedding_i) / Σ(quality_i)
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.modality_weights = nn.Parameter(torch.ones(4))  # audio, text, video, face
    
    def forward(self, embeddings: Dict[str, torch.Tensor],
                qualities: Dict[str, float]) -> torch.Tensor:
        """
        Fuse embeddings weighted by quality.
        
        Args:
            embeddings: Dict of modality -> embedding tensor
            qualities: Dict of modality -> quality score [0,1]
            
        Returns:
            Fused 768-dim embedding
        """
        modality_order = ['audio', 'text', 'video', 'face']
        
        weighted_sum = torch.zeros(self.embed_dim, device=DEVICE)
        weight_sum = 0.0
        
        for i, modality in enumerate(modality_order):
            if modality in embeddings and modality in qualities:
                emb = embeddings[modality]
                if emb.dim() == 2:
                    emb = emb.squeeze(0)
                
                quality = qualities[modality]
                learned_weight = torch.softmax(self.modality_weights, dim=0)[i]
                
                combined_weight = quality * learned_weight.item()
                weighted_sum += combined_weight * emb
                weight_sum += combined_weight
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        return weighted_sum
    
    def fuse_numpy(self, embeddings: Dict[str, np.ndarray],
                   qualities: Dict[str, float]) -> np.ndarray:
        """Convenience method for numpy inputs."""
        torch_embeddings = {
            k: torch.tensor(v, dtype=torch.float32).to(DEVICE) 
            for k, v in embeddings.items()
        }
        
        with torch.no_grad():
            fused = self.forward(torch_embeddings, qualities)
        
        return safe_embedding(fused.cpu().numpy())


class ModalityImputer(nn.Module):
    """
    Impute missing modality embeddings using cross-modal prediction.
    ADV9.
    
    Critical for EATD-Corpus which has NO video.
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.audio_to_video = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
        self.text_to_video = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
        self.audio_text_to_face = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
    
    def impute_video(self, audio_emb: torch.Tensor = None,
                     text_emb: torch.Tensor = None) -> torch.Tensor:
        """Impute video embedding from audio and/or text."""
        if audio_emb is not None and text_emb is not None:
            pred1 = self.audio_to_video(audio_emb)
            pred2 = self.text_to_video(text_emb)
            return (pred1 + pred2) / 2
        elif audio_emb is not None:
            return self.audio_to_video(audio_emb)
        elif text_emb is not None:
            return self.text_to_video(text_emb)
        else:
            return torch.zeros(self.embed_dim, device=DEVICE)
    
    def impute_face(self, audio_emb: torch.Tensor,
                    text_emb: torch.Tensor) -> torch.Tensor:
        """Impute face embedding from audio and text."""
        combined = torch.cat([audio_emb, text_emb], dim=-1)
        return self.audio_text_to_face(combined)
    
    def impute_missing(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Impute any missing modalities.
        
        Args:
            embeddings: Dict with available modality embeddings
            
        Returns:
            Dict with all modalities (original + imputed)
        """
        result = dict(embeddings)
        
        audio = embeddings.get('audio')
        text = embeddings.get('text')
        
        audio_t = torch.tensor(audio, dtype=torch.float32).to(DEVICE) if audio is not None else None
        text_t = torch.tensor(text, dtype=torch.float32).to(DEVICE) if text is not None else None
        
        with torch.no_grad():
            if 'video' not in embeddings or embeddings['video'] is None:
                video = self.impute_video(audio_t, text_t)
                result['video'] = safe_embedding(video.cpu().numpy())
                result['video_imputed'] = True
            
            if 'face' not in embeddings or embeddings['face'] is None:
                if audio_t is not None and text_t is not None:
                    face = self.impute_face(audio_t, text_t)
                    result['face'] = safe_embedding(face.cpu().numpy())
                    result['face_imputed'] = True
        
        return result


class ConcatenationFusion:
    """
    Simple concatenation-based fusion with projection.
    """
    
    def __init__(self, num_modalities: int = 4, embed_dim: int = 768):
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        
        self.projector = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embed_dim)
        ).to(DEVICE)
    
    def fuse(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate and project embeddings."""
        modalities = ['audio', 'text', 'video', 'face']
        
        vectors = []
        for mod in modalities:
            if mod in embeddings and embeddings[mod] is not None:
                vectors.append(embeddings[mod])
            else:
                vectors.append(np.zeros(self.embed_dim))
        
        concatenated = np.concatenate(vectors)
        tensor = torch.tensor(concatenated, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            fused = self.projector(tensor)
        
        return safe_embedding(fused.cpu().numpy().squeeze())


class CrossModalProcessor:
    """Unified cross-modal processing (R54-R59, ADV8-9)."""
    
    def __init__(self):
        self.aligner = TemporalAligner()
        self.quality_fusion = QualityGatedFusion().to(DEVICE)
        self.imputer = ModalityImputer().to(DEVICE)
        self.concat_fusion = ConcatenationFusion()
    
    def process(self, embeddings: Dict[str, np.ndarray],
                qualities: Dict[str, float] = None,
                impute_missing: bool = True) -> Dict:
        """
        Process and fuse multimodal embeddings.
        
        Returns:
            Dict with fused embedding and fusion info
        """
        if impute_missing:
            embeddings = self.imputer.impute_missing(embeddings)
        
        if qualities is None:
            qualities = {k: 1.0 for k in embeddings.keys()}
        
        fused_quality = self.quality_fusion.fuse_numpy(embeddings, qualities)
        
        fused_concat = self.concat_fusion.fuse(embeddings)
        
        return {
            'fused_embedding_quality': fused_quality,
            'fused_embedding_concat': fused_concat,
            'modality_embeddings': embeddings,
            'qualities': qualities,
            'imputed': {k: embeddings.get(f'{k}_imputed', False) for k in ['video', 'face']}
        }
