"""
H5-OmniFusion Utility Module
Common utilities: safe imports, dimension projection, error handling, robust loaders.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import re
import gc
from typing import Optional, List, Tuple, Dict, Any, Union
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn


try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("WARNING: librosa not available")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False

try:
    import parselmouth
    from parselmouth.praat import call
    PRAAT_AVAILABLE = True
except ImportError:
    PRAAT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from snownlp import SnowNLP
    SNOWNLP_AVAILABLE = True
except ImportError:
    SNOWNLP_AVAILABLE = False

try:
    from transformers import (
        Wav2Vec2Model, Wav2Vec2FeatureExtractor,
        VideoMAEModel, VideoMAEImageProcessor,
        AutoModel, AutoTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()


class DimensionProjector(nn.Module):
    """
    Project embeddings from any dimension to target dimension (default 768).
    Used to ensure all modality embeddings are 768-dim for fusion.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if input_dim != output_dim:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
        else:
            self.projector = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


_PROJECTOR_CACHE: Dict[int, DimensionProjector] = {}

def ensure_768_dim(embedding: Union[np.ndarray, torch.Tensor], 
                   device: torch.device = DEVICE) -> torch.Tensor:
    """
    Ensure embedding is exactly 768-dimensional.
    Uses cached projectors for efficiency.
    """
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding, dtype=torch.float32)
    
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    
    input_dim = embedding.shape[-1]
    
    if input_dim == 768:
        return embedding.to(device)
    
    if input_dim not in _PROJECTOR_CACHE:
        _PROJECTOR_CACHE[input_dim] = DimensionProjector(input_dim, 768).to(device)
    
    with torch.no_grad():
        return _PROJECTOR_CACHE[input_dim](embedding.to(device))


def safe_embedding(embedding: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Ensure embedding has no NaN or Inf values.
    Replaces invalid values with zeros.
    """
    if torch.is_tensor(embedding):
        return torch.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        return np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)


def validate_embedding(embedding: Union[np.ndarray, torch.Tensor], 
                       expected_dim: int = 768) -> Tuple[bool, str]:
    """
    Validate that an embedding meets specifications.
    Returns (is_valid, error_message).
    """
    if embedding is None:
        return False, "Embedding is None"
    
    if torch.is_tensor(embedding):
        shape = embedding.shape
        has_nan = torch.isnan(embedding).any().item()
        has_inf = torch.isinf(embedding).any().item()
    else:
        shape = embedding.shape
        has_nan = np.isnan(embedding).any()
        has_inf = np.isinf(embedding).any()
    
    if shape[-1] != expected_dim:
        return False, f"Expected {expected_dim}-dim, got {shape[-1]}-dim"
    
    if has_nan:
        return False, "Contains NaN values"
    
    if has_inf:
        return False, "Contains Inf values"
    
    return True, "OK"


def robust_audio_load(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, bool]:
    """
    Load audio with multiple fallback methods.
    Returns (waveform, success_flag).
    """
    if not os.path.exists(audio_path):
        print(f"WARNING: Audio file not found: {audio_path}")
        return np.zeros(sr * 10), False  # 10 seconds of silence
    
    if LIBROSA_AVAILABLE:
        try:
            waveform, _ = librosa.load(audio_path, sr=sr)
            if len(waveform) > 0:
                return waveform, True
        except Exception as e:
            print(f"librosa failed: {e}")
    
    try:
        waveform, orig_sr = sf.read(audio_path)
        if orig_sr != sr and LIBROSA_AVAILABLE:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sr)
        return waveform, True
    except Exception as e:
        print(f"soundfile failed: {e}")
    
    print(f"WARNING: Could not load {audio_path}, returning silence")
    return np.zeros(sr * 10), False


def robust_video_load(video_path: str, num_frames: int = 16, 
                      target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, bool]:
    """
    Load video frames with error handling.
    Returns (frames_array, success_flag).
    """
    if not os.path.exists(video_path):
        print(f"WARNING: Video not found: {video_path}")
        return np.zeros((num_frames, 3, *target_size)), False
    
    if not CV2_AVAILABLE:
        print("WARNING: cv2 not available")
        return np.zeros((num_frames, 3, *target_size)), False
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("Video has no frames")
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
            else:
                frames.append(np.zeros((*target_size, 3), dtype=np.uint8))
        
        cap.release()
        frames_array = np.array(frames).transpose(0, 3, 1, 2)
        return frames_array, True
        
    except Exception as e:
        print(f"Video loading error: {e}")
        return np.zeros((num_frames, 3, *target_size)), False


def robust_transcript_load(transcript_path: str) -> Tuple[str, bool]:
    """
    Load transcript with multiple encoding fallbacks.
    Returns (text_content, success_flag).
    """
    if not os.path.exists(transcript_path):
        return "", False
    
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(transcript_path, 'r', encoding=encoding) as f:
                content = f.read()
            return content, True
        except:
            continue
    
    return "", False


def clean_transcript(text: str) -> str:
    """
    Clean transcript by removing timestamps, speaker tags, and annotations.
    """
    lines = text.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = re.sub(r'^\d+\.?\d*\s+\d+\.?\d*\s+', '', line)
        line = re.sub(r'^(ELLIE|Participant|Speaker\s*\d*)[:\s]+', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\[.*?\]', '', line)
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)


def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def memory_efficient_inference():
    """Context manager for memory-efficient inference."""
    try:
        yield
    finally:
        clear_memory()


def calculate_snr(waveform: np.ndarray, sr: int = 16000) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    if len(waveform) == 0:
        return 0.0
    
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-10:
        return 0.0
    
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    if len(waveform) < frame_length:
        return 20.0  # Default for very short audio
    
    frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length) if LIBROSA_AVAILABLE else np.array([waveform])
    frame_energy = np.sum(frames ** 2, axis=0)
    
    threshold = np.percentile(frame_energy, 10)
    noise_energy = np.mean(frame_energy[frame_energy <= threshold])
    signal_energy = np.mean(frame_energy)
    
    if noise_energy < 1e-10:
        return 40.0  # Very clean signal
    
    snr = 10 * np.log10(signal_energy / noise_energy)
    return max(0.0, min(snr, 50.0))  # Clamp to reasonable range


def calculate_clipping_ratio(waveform: np.ndarray, threshold: float = 0.99) -> float:
    """Calculate ratio of clipped samples."""
    if len(waveform) == 0:
        return 0.0
    clipped = np.sum(np.abs(waveform) >= threshold)
    return clipped / len(waveform)


def calculate_quality_score(snr: float, clipping: float, vad_ratio: float,
                            snr_min: float = 15.0, clip_max: float = 0.01, 
                            vad_min: float = 0.4) -> float:
    """
    Calculate overall audio quality score in [0, 1].
    Uses soft thresholds for gradual degradation.
    """
    snr_score = 1.0 / (1.0 + np.exp(-(snr - snr_min) / 5.0))
    
    clip_score = max(0.0, 1.0 - clipping / clip_max)
    
    vad_score = min(1.0, vad_ratio / vad_min)
    
    return (snr_score + clip_score + vad_score) / 3.0


print(f"Utils loaded. Device: {DEVICE}")
print(f"Available: librosa={LIBROSA_AVAILABLE}, opensmile={OPENSMILE_AVAILABLE}, "
      f"cv2={CV2_AVAILABLE}, transformers={TRANSFORMERS_AVAILABLE}")
