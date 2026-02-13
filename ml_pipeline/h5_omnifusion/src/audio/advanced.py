"""
Audio Advanced Features Module
Implements ADV1, ADV3, ADV5 and Step 11, R15 from H5-OmniFusion specification.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import DEVICE, LIBROSA_AVAILABLE

if LIBROSA_AVAILABLE:
    import librosa


class ResponseLatencyExtractor:
    """
    Measure precise millisecond gaps between interviewer and participant speech.
    ADV1: Quantifies psychomotor retardation - a key depression indicator.
    """
    
    def __init__(self):
        pass
    
    def extract_from_transcript(self, transcript_df) -> Dict:
        """
        Calculate response latencies from parsed transcript DataFrame.
        
        Args:
            transcript_df: DataFrame with start, end, speaker columns
            
        Returns:
            Dict with latencies, mean, std, max
        """
        if transcript_df is None or transcript_df.empty:
            return self._default_result()
        
        df = transcript_df.copy()
        latencies = []
        
        for i in range(1, len(df)):
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]
            
            prev_is_interviewer = any(
                label.lower() in prev_row['speaker'].lower() 
                for label in ['ellie', 'interviewer']
            )
            curr_is_participant = any(
                label.lower() in curr_row['speaker'].lower()
                for label in ['participant', 'man', 'woman', 'user']
            )
            
            if prev_is_interviewer and curr_is_participant:
                latency_ms = (curr_row['start'] - prev_row['end']) * 1000
                if latency_ms >= 0:  # Ignore overlapping speech
                    latencies.append(latency_ms)
        
        if not latencies:
            return self._default_result()
        
        return {
            'latencies': latencies,
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'count': len(latencies)
        }
    
    def _default_result(self) -> Dict:
        return {
            'latencies': [], 'mean_latency_ms': 0.0, 'std_latency_ms': 0.0,
            'max_latency_ms': 0.0, 'min_latency_ms': 0.0, 'count': 0
        }


class ProsodyFingerprint(nn.Module):
    """
    Generate learned 32-dim embedding of speech rhythm and pause distributions.
    ADV3: Captures temporal "shape" of depressive speech patterns.
    """
    
    def __init__(self, input_features: int = 20, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
    
    def extract_prosodic_features(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract prosodic features for fingerprinting.
        
        Features include:
        - Pause distribution statistics
        - Speaking rate variability  
        - Pitch contour statistics
        - Energy envelope patterns
        """
        features = []
        
        if not LIBROSA_AVAILABLE:
            return np.zeros(20, dtype=np.float32)
        
        try:
            intervals = librosa.effects.split(waveform, top_db=30)
            
            if len(intervals) > 1:
                pauses = []
                for i in range(1, len(intervals)):
                    pause = (intervals[i][0] - intervals[i-1][1]) / sr
                    if pause > 0.1:  # Min 100ms
                        pauses.append(pause)
                
                if pauses:
                    features.extend([
                        np.mean(pauses), np.std(pauses), 
                        np.max(pauses), len(pauses)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
                
                speech_durations = [(e - s) / sr for s, e in intervals]
                features.extend([
                    np.mean(speech_durations), np.std(speech_durations),
                    np.max(speech_durations), len(speech_durations)
                ])
            else:
                features.extend([0] * 8)
            
            rms = librosa.feature.rms(y=waveform)[0]
            features.extend([
                np.mean(rms), np.std(rms), np.max(rms) - np.min(rms)
            ])
            
            if len(rms) > 10:
                thirds = np.array_split(rms, 3)
                energy_trajectory = [np.mean(t) for t in thirds]
                features.extend([
                    energy_trajectory[1] - energy_trajectory[0],  # Mid - Start
                    energy_trajectory[2] - energy_trajectory[1]   # End - Mid
                ])
            else:
                features.extend([0, 0])
            
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            features.extend([
                tempo / 200.0,  # Normalize tempo
                np.std(np.diff(beats)) if len(beats) > 1 else 0,  # Beat regularity
                len(beats) / (len(waveform) / sr)  # Beats per second
            ])
            
            f0, voiced, _ = librosa.pyin(waveform, fmin=75, fmax=500, sr=sr)
            f0_voiced = f0[~np.isnan(f0)] if f0 is not None else []
            
            if len(f0_voiced) > 2:
                features.extend([
                    np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-8),  # CV
                    len(f0_voiced) / len(f0) if len(f0) > 0 else 0  # Voiced ratio
                ])
            else:
                features.extend([0, 0])
            
            features = np.array(features[:20], dtype=np.float32)
            if len(features) < 20:
                features = np.pad(features, (0, 20 - len(features)))
            
            return features
            
        except Exception as e:
            return np.zeros(20, dtype=np.float32)
    
    def forward(self, prosodic_features: torch.Tensor) -> torch.Tensor:
        """Generate fingerprint from prosodic features."""
        return self.encoder(prosodic_features)
    
    def generate_fingerprint(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Generate 32-dim prosodic fingerprint for audio.
        
        Returns:
            32-dim numpy array
        """
        features = self.extract_prosodic_features(waveform, sr)
        features_tensor = torch.tensor(features).unsqueeze(0).to(DEVICE)
        
        self.eval()
        with torch.no_grad():
            fingerprint = self.forward(features_tensor.float())
        
        return fingerprint.cpu().numpy().squeeze()


class SighDetector:
    """
    Detect sighs and analyze breath group patterns.
    ADV5: Breath patterns correlate with anxiety and depression.
    
    Sigh detection criteria:
    - Duration: 1-3 seconds
    - Frequency: Low-frequency dominance (<500Hz)
    - Envelope: Gradual decay pattern
    """
    
    def __init__(self,
                 min_duration: float = 1.0,
                 max_duration: float = 3.0,
                 max_freq: float = 500.0):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_freq = max_freq
    
    def detect_sighs(self, waveform: np.ndarray, sr: int = 16000) -> Dict:
        """
        Detect sighs in audio.
        
        Returns:
            Dict with sigh_count, sigh_times, breath_interval_std
        """
        if not LIBROSA_AVAILABLE:
            return self._default_result()
        
        try:
            sighs = []
            
            intervals = librosa.effects.split(waveform, top_db=25)
            
            for start, end in intervals:
                duration = (end - start) / sr
                
                if not (self.min_duration <= duration <= self.max_duration):
                    continue
                
                segment = waveform[start:end]
                
                if self._is_low_frequency_dominant(segment, sr):
                    if self._has_decay_envelope(segment):
                        sighs.append({
                            'start_sec': start / sr,
                            'end_sec': end / sr,
                            'duration': duration
                        })
            
            breath_intervals = []
            for i in range(1, len(intervals)):
                gap = (intervals[i][0] - intervals[i-1][1]) / sr
                if 0.3 < gap < 5.0:  # Reasonable breath interval range
                    breath_intervals.append(gap)
            
            return {
                'sigh_count': len(sighs),
                'sigh_times': sighs,
                'sigh_rate': len(sighs) / (len(waveform) / sr / 60),  # per minute
                'breath_intervals': breath_intervals,
                'breath_interval_mean': float(np.mean(breath_intervals)) if breath_intervals else 0,
                'breath_interval_std': float(np.std(breath_intervals)) if breath_intervals else 0
            }
            
        except Exception as e:
            return self._default_result()
    
    def _is_low_frequency_dominant(self, segment: np.ndarray, sr: int) -> bool:
        """Check if segment has low-frequency dominance (<500Hz)."""
        try:
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            mean_centroid = np.mean(centroid)
            return mean_centroid < self.max_freq
        except:
            return False
    
    def _has_decay_envelope(self, segment: np.ndarray) -> bool:
        """Check if segment has gradual decay pattern."""
        try:
            third = len(segment) // 3
            if third < 10:
                return False
            
            rms_first = np.sqrt(np.mean(segment[:third] ** 2))
            rms_last = np.sqrt(np.mean(segment[-third:] ** 2))
            
            return rms_last < rms_first * 0.7
        except:
            return False
    
    def _default_result(self) -> Dict:
        return {
            'sigh_count': 0, 'sigh_times': [], 'sigh_rate': 0,
            'breath_intervals': [], 'breath_interval_mean': 0, 'breath_interval_std': 0
        }


class AdvancedAudioExtractor:
    """
    Combined advanced audio feature extraction (ADV1, ADV3, ADV5).
    """
    
    def __init__(self):
        self.response_latency = ResponseLatencyExtractor()
        self.prosody_fingerprint = ProsodyFingerprint().to(DEVICE)
        self.sigh_detector = SighDetector()
    
    def extract_all(self, waveform: np.ndarray, sr: int = 16000,
                    transcript_df=None) -> Dict:
        """
        Extract all advanced audio features.
        
        Returns:
            Dict with response_latency, prosody_fingerprint, sigh_detection
        """
        latency_features = self.response_latency.extract_from_transcript(transcript_df)
        
        fingerprint = self.prosody_fingerprint.generate_fingerprint(waveform, sr)
        
        sigh_features = self.sigh_detector.detect_sighs(waveform, sr)
        
        return {
            'response_latency': latency_features,
            'prosody_fingerprint': fingerprint,  # 32-dim
            'sigh_detection': sigh_features
        }
