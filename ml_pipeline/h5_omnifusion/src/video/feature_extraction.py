"""
Video Feature Extraction Module
Implements Steps 25-26 and R37-R38 from H5-OmniFusion specification.
"""
import numpy as np
import torch
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import (
    DEVICE, CV2_AVAILABLE, TRANSFORMERS_AVAILABLE, TIMM_AVAILABLE,
    ensure_768_dim, safe_embedding, clear_memory
)
from ..model_loader import MODEL_LOADER

if CV2_AVAILABLE:
    import cv2


class VideoMAEExtractor:
    """
    Extract 768-dim spatiotemporal video embeddings.
    Steps 25, R37.
    
    Uses MCG-NJU/videomae-base with ViT fallback.
    """
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
        self.processor = None
        self.is_vit_fallback = False
    
    def _ensure_loaded(self):
        """Lazy load model."""
        if self.model is None:
            self.model, self.processor = MODEL_LOADER.get_videomae()
            self.is_vit_fallback = self.processor is None
    
    def extract(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract video embedding from frames.
        
        Args:
            frames: Normalized frames (N, H, W, 3) or (N, 3, H, W)
            
        Returns:
            768-dim embedding
        """
        self._ensure_loaded()
        
        if self.model is None:
            return np.zeros(768, dtype=np.float32)
        
        try:
            if self.is_vit_fallback:
                return self._extract_vit(frames)
            else:
                return self._extract_videomae(frames)
        except Exception as e:
            print(f"Video embedding error: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def _extract_videomae(self, frames: np.ndarray) -> np.ndarray:
        """Extract using VideoMAE."""
        
        if frames.shape[1] == 3 and frames.shape[-1] != 3:
            frames = frames.transpose(0, 2, 3, 1)
        
        if self.processor is not None:
            inputs = self.processor(list(frames), return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        else:
            frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2).float()
            pixel_values = frames_tensor.unsqueeze(0).to(self.device)
        
        if hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
            pixel_values = pixel_values.half()
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
            if hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                embedding = outputs[0].mean(dim=1)
        
        return safe_embedding(embedding.cpu().float().numpy().squeeze())
    
    def _extract_vit(self, frames: np.ndarray) -> np.ndarray:
        """Fallback: Extract using ViT (per-frame then average)."""
        if frames.shape[1] == 3 and frames.shape[-1] != 3:
            frames = frames.transpose(0, 2, 3, 1)
        
        embeddings = []
        
        for frame in frames:
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).float()
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            
            if hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                frame_tensor = frame_tensor.half()
            
            with torch.no_grad():
                output = self.model.forward_features(frame_tensor)
                if output.dim() == 3:
                    emb = output[:, 0]  # CLS token
                else:
                    emb = output
                embeddings.append(emb.cpu().float())
        
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        
        result = safe_embedding(avg_embedding.numpy().squeeze())
        if len(result) != 768:
            result = ensure_768_dim(result).cpu().numpy().squeeze()
        
        return result


class OpticalFlowAnalyzer:
    """
    Analyze motion between frames using optical flow.
    Steps 26, R38.
    
    Uses Farneback or Lucas-Kanade optical flow.
    """
    
    def __init__(self):
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def analyze(self, frames: np.ndarray) -> Dict:
        """
        Compute optical flow statistics between frames.
        
        Args:
            frames: Array (N, H, W, 3) RGB frames
            
        Returns:
            Dict with flow magnitude mean, std, and motion trajectory
        """
        if not CV2_AVAILABLE or len(frames) < 2:
            return self._default_result()
        
        try:
            flow_magnitudes = []
            flow_directions = []
            
            for i in range(len(frames) - 1):
                prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, next_gray, None, **self.farneback_params
                )
                
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                flow_magnitudes.append(magnitude.mean())
                flow_directions.append(angle.mean())
            
            mags = np.array(flow_magnitudes)
            
            return {
                'flow_magnitude_mean': float(mags.mean()),
                'flow_magnitude_std': float(mags.std()),
                'flow_magnitude_max': float(mags.max()),
                'flow_magnitude_min': float(mags.min()),
                'motion_variability': float(np.std(np.diff(mags))) if len(mags) > 1 else 0,
                'frame_pairs': len(flow_magnitudes)
            }
            
        except Exception as e:
            print(f"Optical flow error: {e}")
            return self._default_result()
    
    def compute_motion_trajectory(self, frames: np.ndarray) -> Dict:
        """
        Track overall motion trajectory over time.
        
        Returns:
            Dict with motion over session thirds (for temporal trajectory)
        """
        if not CV2_AVAILABLE or len(frames) < 3:
            return {'motion_thirds': [0, 0, 0], 'motion_slope': 0}
        
        try:
            third_size = len(frames) // 3
            thirds = [
                frames[:third_size],
                frames[third_size:2*third_size],
                frames[2*third_size:]
            ]
            
            motion_per_third = []
            for third_frames in thirds:
                if len(third_frames) < 2:
                    motion_per_third.append(0)
                    continue
                
                third_flow = self.analyze(third_frames)
                motion_per_third.append(third_flow['flow_magnitude_mean'])
            
            slope = np.polyfit(range(3), motion_per_third, 1)[0] if len(motion_per_third) == 3 else 0
            
            return {
                'motion_thirds': motion_per_third,
                'motion_slope': float(slope)
            }
            
        except:
            return {'motion_thirds': [0, 0, 0], 'motion_slope': 0}
    
    def _default_result(self) -> Dict:
        return {
            'flow_magnitude_mean': 0,
            'flow_magnitude_std': 0,
            'flow_magnitude_max': 0,
            'flow_magnitude_min': 0,
            'motion_variability': 0,
            'frame_pairs': 0
        }


class VideoFeatureExtractor:
    """
    Unified video feature extraction (Steps 25-26, R37-R38).
    """
    
    def __init__(self):
        self.videomae = VideoMAEExtractor()
        self.optical_flow = OpticalFlowAnalyzer()
    
    def extract_all(self, frames: np.ndarray, normalized_frames: np.ndarray = None) -> Dict:
        """
        Extract all video features.
        
        Args:
            frames: Raw frames (N, H, W, 3) uint8
            normalized_frames: Normalized frames (N, H, W, 3) float32 (for VideoMAE)
            
        Returns:
            Dict with video_embedding and optical flow analysis
        """
        embedding_frames = normalized_frames if normalized_frames is not None else frames
        
        video_embedding = self.videomae.extract(embedding_frames)
        
        flow_analysis = self.optical_flow.analyze(frames)
        
        motion_trajectory = self.optical_flow.compute_motion_trajectory(frames)
        
        return {
            'video_embedding': video_embedding,  # 768-dim
            'optical_flow': flow_analysis,
            'motion_trajectory': motion_trajectory
        }
