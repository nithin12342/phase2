"""
Audio Preprocessing Module
Implements Steps 1-8 and R1-R9 from H5-OmniFusion specification.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import (
    LIBROSA_AVAILABLE, NOISEREDUCE_AVAILABLE,
    robust_audio_load, safe_embedding
)

if LIBROSA_AVAILABLE:
    import librosa

if NOISEREDUCE_AVAILABLE:
    import noisereduce as nr

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False


class AudioLoader:
    """
    Load and resample audio to 16kHz for Wav2Vec2 compatibility.
    Steps 1, R1-R2.
    """
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def load(self, audio_path: str) -> Tuple[np.ndarray, int, bool]:
        """
        Load audio file and resample to target sample rate.
        
        Returns:
            waveform: Audio samples
            sr: Sample rate (always target_sr)
            success: True if loaded successfully
        """
        waveform, success = robust_audio_load(audio_path, self.target_sr)
        return waveform, self.target_sr, success
    
    def load_and_validate(self, audio_path: str) -> Dict:
        """Load audio with validation metadata."""
        waveform, sr, success = self.load(audio_path)
        
        return {
            'waveform': waveform,
            'sr': sr,
            'success': success,
            'duration_sec': len(waveform) / sr if success else 0,
            'num_samples': len(waveform)
        }


class StereoToMono:
    """
    Convert multi-channel audio to mono.
    Step 2, R3.
    """
    
    def process(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert stereo to mono by averaging channels.
        
        Args:
            waveform: Input audio (may be stereo or mono)
            
        Returns:
            Mono waveform
        """
        if waveform.ndim > 1:
            return np.mean(waveform, axis=0).astype(np.float32)
        return waveform.astype(np.float32)


class PeakNormalizer:
    """
    Normalize audio amplitude to [-1, 1] range.
    Step 4, R5.
    """
    
    def process(self, waveform: np.ndarray) -> np.ndarray:
        """
        Scale waveform so peak amplitude is 1.0.
        
        Args:
            waveform: Input audio
            
        Returns:
            Peak-normalized waveform
        """
        peak = np.max(np.abs(waveform))
        if peak < 1e-8:
            return waveform
        return (waveform / (peak + 1e-8)).astype(np.float32)


class LoudnessNormalizer:
    """
    Normalize audio loudness to -23 LUFS (EBU R128 standard).
    Step 5, R6.
    """
    
    def __init__(self, target_lufs: float = -23.0):
        self.target_lufs = target_lufs
        self.meter = None
        
        if PYLOUDNORM_AVAILABLE:
            self.meter = pyln.Meter(16000)  # 16kHz sample rate
    
    def process(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Normalize to target LUFS.
        
        Args:
            waveform: Input audio
            sr: Sample rate
            
        Returns:
            LUFS-normalized waveform
        """
        if not PYLOUDNORM_AVAILABLE:
            return self._rms_normalize(waveform)
        
        try:
            loudness = self.meter.integrated_loudness(waveform)
            
            if np.isinf(loudness) or np.isnan(loudness):
                return self._rms_normalize(waveform)
            
            normalized = pyln.normalize.loudness(waveform, loudness, self.target_lufs)
            return normalized.astype(np.float32)
            
        except Exception:
            return self._rms_normalize(waveform)
    
    def _rms_normalize(self, waveform: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """Fallback RMS normalization."""
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms < 1e-8:
            return waveform
        return (waveform * target_rms / rms).astype(np.float32)


class NoiseReducer:
    """
    Reduce background noise using spectral gating.
    Step 6, R7.
    """
    
    def __init__(self, prop_decrease: float = 0.8, stationary: bool = True):
        self.prop_decrease = prop_decrease
        self.stationary = stationary
    
    def process(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Apply spectral noise reduction.
        
        Args:
            waveform: Noisy audio
            sr: Sample rate
            
        Returns:
            Denoised audio
        """
        if not NOISEREDUCE_AVAILABLE:
            return waveform
        
        try:
            reduced = nr.reduce_noise(
                y=waveform,
                sr=sr,
                prop_decrease=self.prop_decrease,
                stationary=self.stationary
            )
            return reduced.astype(np.float32)
        except Exception:
            return waveform


class VADProcessor:
    """
    Detect speech regions in audio.
    Step 7, R8.
    
    Uses librosa.effects.split or Silero VAD (if available).
    """
    
    def __init__(self, top_db: int = 30, min_silence_len_ms: int = 200):
        self.top_db = top_db
        self.min_silence_len = min_silence_len_ms / 1000.0  # Convert to seconds
    
    def detect(self, waveform: np.ndarray, sr: int = 16000) -> List[Tuple[int, int]]:
        """
        Detect speech segments.
        
        Returns:
            List of (start_sample, end_sample) tuples
        """
        if not LIBROSA_AVAILABLE:
            return [(0, len(waveform))]
        
        try:
            intervals = librosa.effects.split(waveform, top_db=self.top_db)
            return [(int(start), int(end)) for start, end in intervals]
        except Exception:
            return [(0, len(waveform))]
    
    def extract_speech(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract only speech portions of audio.
        
        Returns:
            Concatenated speech segments
        """
        segments = self.detect(waveform, sr)
        if not segments:
            return waveform
        
        speech_parts = [waveform[start:end] for start, end in segments]
        return np.concatenate(speech_parts) if speech_parts else waveform
    
    def get_vad_ratio(self, waveform: np.ndarray, sr: int = 16000) -> float:
        """
        Calculate ratio of speech to total audio duration.
        
        Returns:
            VAD ratio in [0, 1]
        """
        segments = self.detect(waveform, sr)
        speech_samples = sum(end - start for start, end in segments)
        return speech_samples / len(waveform) if len(waveform) > 0 else 0.0


class Segmenter:
    """
    Segment audio into fixed-length windows with overlap.
    Step 8, R9.
    """
    
    def __init__(self, window_sec: float = 10.0, overlap: float = 0.5, sr: int = 16000):
        self.window_sec = window_sec
        self.overlap = overlap
        self.sr = sr
        
        self.window_samples = int(window_sec * sr)
        self.hop_samples = int(self.window_samples * (1 - overlap))
    
    def segment(self, waveform: np.ndarray) -> List[np.ndarray]:
        """
        Divide audio into overlapping segments.
        
        Returns:
            List of segment arrays
        """
        segments = []
        
        for start in range(0, len(waveform) - self.window_samples + 1, self.hop_samples):
            segment = waveform[start:start + self.window_samples]
            segments.append(segment)
        
        if len(waveform) > self.window_samples:
            remaining = len(waveform) % self.hop_samples
            if remaining > 0:
                last_start = len(waveform) - self.window_samples
                if last_start >= 0:
                    segments.append(waveform[last_start:])
        elif len(waveform) > 0:
            padded = np.zeros(self.window_samples)
            padded[:len(waveform)] = waveform
            segments.append(padded)
        
        return segments
    
    def segment_with_timestamps(self, waveform: np.ndarray) -> List[Dict]:
        """
        Segment with timestamp metadata.
        
        Returns:
            List of dicts with 'audio', 'start_sec', 'end_sec'
        """
        result = []
        
        for start in range(0, len(waveform) - self.window_samples + 1, self.hop_samples):
            end = start + self.window_samples
            result.append({
                'audio': waveform[start:end],
                'start_sec': start / self.sr,
                'end_sec': end / self.sr
            })
        
        return result


class AudioPreprocessor:
    """
    Unified audio preprocessing pipeline combining Steps 1-8, R1-R9.
    """
    
    def __init__(self, config=None):
        self.config = config or CFG
        
        self.loader = AudioLoader(self.config.SAMPLE_RATE)
        self.stereo_to_mono = StereoToMono()
        self.peak_normalizer = PeakNormalizer()
        self.loudness_normalizer = LoudnessNormalizer(self.config.TARGET_LUFS)
        self.noise_reducer = NoiseReducer(self.config.NOISE_PROP_DECREASE)
        self.vad = VADProcessor(self.config.VAD_TOP_DB)
        self.segmenter = Segmenter(
            self.config.WINDOW_SEC,
            self.config.OVERLAP,
            self.config.SAMPLE_RATE
        )
    
    def process(self, audio_path: str, extract_speech_only: bool = True) -> Dict:
        """
        Run complete audio preprocessing pipeline.
        
        Args:
            audio_path: Path to audio file
            extract_speech_only: Whether to extract participant speech only
            
        Returns:
            Dict with processed audio, segments, and metadata
        """
        load_result = self.loader.load_and_validate(audio_path)
        if not load_result['success']:
            return self._failure_result("Load failed")
        
        waveform = load_result['waveform']
        sr = load_result['sr']
        
        waveform = self.stereo_to_mono.process(waveform)
        
        waveform = self.peak_normalizer.process(waveform)
        
        waveform = self.loudness_normalizer.process(waveform, sr)
        
        waveform = self.noise_reducer.process(waveform, sr)
        
        vad_ratio = self.vad.get_vad_ratio(waveform, sr)
        if extract_speech_only:
            waveform = self.vad.extract_speech(waveform, sr)
        
        segments = self.segmenter.segment(waveform)
        
        return {
            'success': True,
            'waveform': waveform,
            'sr': sr,
            'segments': segments,
            'num_segments': len(segments),
            'duration_sec': len(waveform) / sr,
            'vad_ratio': vad_ratio,
            'load_info': load_result
        }
    
    def _failure_result(self, error: str) -> Dict:
        """Return standardized failure result."""
        return {
            'success': False,
            'error': error,
            'waveform': np.zeros(self.config.SAMPLE_RATE * 10),
            'sr': self.config.SAMPLE_RATE,
            'segments': [],
            'num_segments': 0,
            'duration_sec': 0,
            'vad_ratio': 0.0
        }
