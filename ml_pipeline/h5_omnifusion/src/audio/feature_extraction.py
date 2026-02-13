"""
Audio Feature Extraction Module
Implements Steps 9-11 and R10-R17 from H5-OmniFusion specification.
"""
import numpy as np
import torch
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import (
    DEVICE, LIBROSA_AVAILABLE, PRAAT_AVAILABLE, OPENSMILE_AVAILABLE,
    ensure_768_dim, safe_embedding, clear_memory
)
from ..model_loader import MODEL_LOADER

if LIBROSA_AVAILABLE:
    import librosa

if PRAAT_AVAILABLE:
    import parselmouth
    from parselmouth.praat import call


class Wav2Vec2Extractor:
    """
    Extract 768-dim contextual audio embeddings using Wav2Vec2.
    Steps 9, R10.
    """
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
        self.processor = None
    
    def _ensure_loaded(self):
        """Lazy load model."""
        if self.model is None:
            self.model, self.processor = MODEL_LOADER.get_wav2vec2()
    
    def extract(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract Wav2Vec2 embeddings with mean pooling.
        
        Args:
            waveform: Audio samples (mono, 16kHz)
            sr: Sample rate
            
        Returns:
            768-dim embedding as numpy array
        """
        self._ensure_loaded()
        
        if self.model is None or self.processor is None:
            return self._fallback_embedding()
        
        try:
            inputs = self.processor(
                waveform,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            if next(self.model.parameters()).dtype == torch.float16:
                input_values = input_values.half()
            
            with torch.no_grad():
                outputs = self.model(input_values)
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            return safe_embedding(embedding.cpu().float().numpy().squeeze())
            
        except Exception as e:
            print(f"Wav2Vec2 extraction error: {e}")
            return self._fallback_embedding()
    
    def extract_batch(self, segments: List[np.ndarray], sr: int = 16000) -> np.ndarray:
        """Extract embeddings for multiple segments and average."""
        if not segments:
            return self._fallback_embedding()
        
        embeddings = [self.extract(seg, sr) for seg in segments]
        return np.mean(embeddings, axis=0)
    
    def _fallback_embedding(self) -> np.ndarray:
        """Return zero embedding as fallback."""
        return np.zeros(768, dtype=np.float32)


class EGeMAPSExtractor:
    """
    Extract 88 acoustic features using OpenSMILE eGeMAPSv02.
    Steps 10, R11.
    
    Features include pitch, jitter, shimmer, HNR, MFCCs, loudness, spectral features.
    """
    
    def __init__(self):
        self.smile = None
        self.projector = None
    
    def _ensure_loaded(self):
        """Lazy load OpenSMILE."""
        if self.smile is None:
            self.smile, _ = MODEL_LOADER.get_opensmile()
        
        if self.projector is None and self.smile is not None:
            from ..utils import DimensionProjector
            self.projector = DimensionProjector(88, 768).to(DEVICE)
    
    def extract(self, audio_path: str = None, waveform: np.ndarray = None, 
                sr: int = 16000) -> np.ndarray:
        """
        Extract eGeMAPS features.
        
        Args:
            audio_path: Path to audio file (preferred)
            waveform: Audio samples (if no path)
            sr: Sample rate
            
        Returns:
            768-dim projected embedding
        """
        self._ensure_loaded()
        
        if self.smile is None:
            return self._fallback_with_librosa(waveform, sr)
        
        try:
            if audio_path:
                features = self.smile.process_file(audio_path)
            else:
                features = self.smile.process_signal(waveform, sr)
            
            features_np = features.values.flatten().astype(np.float32)
            
            features_np = np.nan_to_num(features_np, nan=0.0)
            
            features_tensor = torch.tensor(features_np).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                projected = self.projector(features_tensor)
            
            return safe_embedding(projected.cpu().numpy().squeeze())
            
        except Exception as e:
            print(f"eGeMAPS error: {e}")
            return self._fallback_with_librosa(waveform, sr)
    
    def _fallback_with_librosa(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Fallback feature extraction using librosa."""
        if waveform is None or not LIBROSA_AVAILABLE:
            return np.zeros(768, dtype=np.float32)
        
        try:
            features = []
            
            mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
            features.extend(np.mean(mfcc, axis=1))
            
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
            features.append(np.mean(spectral_centroid))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
            features.append(np.mean(spectral_rolloff))
            
            zcr = librosa.feature.zero_crossing_rate(waveform)
            features.append(np.mean(zcr))
            
            rms = librosa.feature.rms(y=waveform)
            features.append(np.mean(rms))
            
            features = np.array(features, dtype=np.float32)
            padded = np.zeros(768, dtype=np.float32)
            padded[:len(features)] = features
            
            return padded
            
        except Exception:
            return np.zeros(768, dtype=np.float32)


class PitchAnalyzer:
    """
    Analyze fundamental frequency (F0) characteristics.
    Step R12.
    
    Uses Praat via parselmouth for accurate pitch tracking.
    """
    
    def __init__(self, f0_min: float = 75.0, f0_max: float = 500.0):
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> Dict:
        """
        Extract pitch statistics.
        
        Returns:
            Dict with f0_mean, f0_std, f0_range, f0_slope
        """
        if not PRAAT_AVAILABLE:
            return self._fallback_analysis(waveform, sr)
        
        try:
            sound = parselmouth.Sound(waveform, sr)
            
            pitch = call(sound, "To Pitch", 0.0, self.f0_min, self.f0_max)
            
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values > 0]  # Remove unvoiced
            
            if len(f0_values) < 2:
                return self._default_result()
            
            return {
                'f0_mean': float(np.mean(f0_values)),
                'f0_std': float(np.std(f0_values)),
                'f0_min': float(np.min(f0_values)),
                'f0_max': float(np.max(f0_values)),
                'f0_range': float(np.max(f0_values) - np.min(f0_values)),
                'f0_slope': float(np.polyfit(range(len(f0_values)), f0_values, 1)[0]),
                'voiced_ratio': len(f0_values) / len(pitch.selected_array['frequency'])
            }
            
        except Exception as e:
            print(f"Pitch analysis error: {e}")
            return self._fallback_analysis(waveform, sr)
    
    def _fallback_analysis(self, waveform: np.ndarray, sr: int) -> Dict:
        """Fallback using librosa."""
        if not LIBROSA_AVAILABLE:
            return self._default_result()
        
        try:
            f0, voiced_flag, _ = librosa.pyin(
                waveform, fmin=self.f0_min, fmax=self.f0_max, sr=sr
            )
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) < 2:
                return self._default_result()
            
            return {
                'f0_mean': float(np.mean(f0_voiced)),
                'f0_std': float(np.std(f0_voiced)),
                'f0_min': float(np.min(f0_voiced)),
                'f0_max': float(np.max(f0_voiced)),
                'f0_range': float(np.max(f0_voiced) - np.min(f0_voiced)),
                'f0_slope': float(np.polyfit(range(len(f0_voiced)), f0_voiced, 1)[0]),
                'voiced_ratio': len(f0_voiced) / len(f0)
            }
        except:
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {
            'f0_mean': 0.0, 'f0_std': 0.0, 'f0_min': 0.0, 'f0_max': 0.0,
            'f0_range': 0.0, 'f0_slope': 0.0, 'voiced_ratio': 0.0
        }


class JitterShimmerAnalyzer:
    """
    Analyze voice quality through jitter (pitch perturbation) and shimmer (amplitude perturbation).
    Step R13.
    """
    
    def __init__(self, f0_min: float = 75.0, f0_max: float = 500.0):
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> Dict:
        """
        Extract jitter and shimmer values.
        
        Returns:
            Dict with jitter_local, jitter_rap, shimmer_local, shimmer_apq
        """
        if not PRAAT_AVAILABLE:
            return self._default_result()
        
        try:
            sound = parselmouth.Sound(waveform, sr)
            
            point_process = call(sound, "To PointProcess (periodic, cc)",
                                self.f0_min, self.f0_max)
            
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 
                               0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0,
                             0.0001, 0.02, 1.3)
            
            shimmer_local = call([sound, point_process], "Get shimmer (local)",
                                0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq = call([sound, point_process], "Get shimmer (apq3)",
                              0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            return {
                'jitter_local': float(jitter_local) if not np.isnan(jitter_local) else 0.0,
                'jitter_rap': float(jitter_rap) if not np.isnan(jitter_rap) else 0.0,
                'shimmer_local': float(shimmer_local) if not np.isnan(shimmer_local) else 0.0,
                'shimmer_apq': float(shimmer_apq) if not np.isnan(shimmer_apq) else 0.0
            }
            
        except Exception as e:
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {
            'jitter_local': 0.0, 'jitter_rap': 0.0,
            'shimmer_local': 0.0, 'shimmer_apq': 0.0
        }


class FormantExtractor:
    """
    Extract formant frequencies (F1-F4) representing vocal tract resonances.
    Step R14.
    """
    
    def __init__(self, max_formant: float = 5500.0, num_formants: int = 4):
        self.max_formant = max_formant
        self.num_formants = num_formants
    
    def extract(self, waveform: np.ndarray, sr: int = 16000) -> Dict:
        """
        Extract mean formant frequencies.
        
        Returns:
            Dict with f1_mean, f2_mean, f3_mean, f4_mean
        """
        if not PRAAT_AVAILABLE:
            return self._default_result()
        
        try:
            sound = parselmouth.Sound(waveform, sr)
            formants = call(sound, "To Formant (burg)", 0.0, self.num_formants, 
                           self.max_formant, 0.025, 50.0)
            
            result = {}
            for i in range(1, self.num_formants + 1):
                values = [call(formants, "Get value at time", i, t, "Hertz", "Linear")
                         for t in np.linspace(0, sound.duration, 50)]
                values = [v for v in values if not np.isnan(v) and v > 0]
                result[f'f{i}_mean'] = float(np.mean(values)) if values else 0.0
            
            return result
            
        except Exception:
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {'f1_mean': 0.0, 'f2_mean': 0.0, 'f3_mean': 0.0, 'f4_mean': 0.0}


class PauseAnalyzer:
    """
    Analyze pause patterns in speech.
    Step R16.
    """
    
    def __init__(self, min_pause_ms: float = 200, energy_threshold_db: float = -30):
        self.min_pause_sec = min_pause_ms / 1000.0
        self.energy_threshold_db = energy_threshold_db
    
    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> Dict:
        """
        Detect and analyze pauses.
        
        Returns:
            Dict with pause_count, pause_duration_mean, pause_duration_max, pause_ratio
        """
        if not LIBROSA_AVAILABLE:
            return self._default_result()
        
        try:
            intervals = librosa.effects.split(
                waveform, top_db=abs(self.energy_threshold_db)
            )
            
            if len(intervals) < 2:
                return self._default_result()
            
            pauses = []
            for i in range(1, len(intervals)):
                pause_start = intervals[i-1][1]
                pause_end = intervals[i][0]
                pause_duration = (pause_end - pause_start) / sr
                
                if pause_duration >= self.min_pause_sec:
                    pauses.append(pause_duration)
            
            if not pauses:
                return self._default_result()
            
            total_duration = len(waveform) / sr
            
            return {
                'pause_count': len(pauses),
                'pause_duration_mean': float(np.mean(pauses)),
                'pause_duration_max': float(np.max(pauses)),
                'pause_duration_total': float(np.sum(pauses)),
                'pause_ratio': float(np.sum(pauses) / total_duration)
            }
            
        except Exception:
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {
            'pause_count': 0, 'pause_duration_mean': 0.0,
            'pause_duration_max': 0.0, 'pause_duration_total': 0.0, 'pause_ratio': 0.0
        }


class SpeakingRateAnalyzer:
    """
    Analyze speaking rate metrics.
    Step R17.
    """
    
    def analyze(self, waveform: np.ndarray, sr: int = 16000, 
                word_count: int = None) -> Dict:
        """
        Estimate speaking rate from audio.
        
        Returns:
            Dict with speaking_rate, articulation_rate, phonation_ratio
        """
        if not LIBROSA_AVAILABLE:
            return self._default_result()
        
        try:
            rms = librosa.feature.rms(y=waveform, hop_length=512)[0]
            
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rms, distance=8, height=np.median(rms) * 0.5)
            syllable_count = len(peaks)
            
            intervals = librosa.effects.split(waveform, top_db=30)
            speech_duration = sum((e - s) / sr for s, e in intervals)
            total_duration = len(waveform) / sr
            
            if speech_duration < 0.1:
                return self._default_result()
            
            return {
                'speaking_rate': syllable_count / total_duration,  # syl/sec (total)
                'articulation_rate': syllable_count / speech_duration,  # syl/sec (speech only)
                'phonation_ratio': speech_duration / total_duration,
                'estimated_syllables': syllable_count
            }
            
        except Exception:
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {
            'speaking_rate': 0.0, 'articulation_rate': 0.0,
            'phonation_ratio': 0.0, 'estimated_syllables': 0
        }


class AudioFeatureExtractor:
    """
    Unified audio feature extraction combining Steps 9-11, R10-R17.
    """
    
    def __init__(self):
        self.wav2vec2 = Wav2Vec2Extractor()
        self.egemaps = EGeMAPSExtractor()
        self.pitch = PitchAnalyzer()
        self.jitter_shimmer = JitterShimmerAnalyzer()
        self.formants = FormantExtractor()
        self.pauses = PauseAnalyzer()
        self.speaking_rate = SpeakingRateAnalyzer()
    
    def extract_all(self, waveform: np.ndarray, sr: int = 16000,
                    audio_path: str = None) -> Dict:
        """
        Extract all audio features.
        
        Returns:
            Dict with all embeddings and prosodic features
        """
        wav2vec2_emb = self.wav2vec2.extract(waveform, sr)
        egemaps_emb = self.egemaps.extract(audio_path, waveform, sr)
        
        pitch_features = self.pitch.analyze(waveform, sr)
        jitter_shimmer_features = self.jitter_shimmer.analyze(waveform, sr)
        formant_features = self.formants.extract(waveform, sr)
        pause_features = self.pauses.analyze(waveform, sr)
        rate_features = self.speaking_rate.analyze(waveform, sr)
        
        return {
            'wav2vec2_embedding': wav2vec2_emb,
            'egemaps_embedding': egemaps_emb,
            
            'pitch': pitch_features,
            'jitter_shimmer': jitter_shimmer_features,
            'formants': formant_features,
            'pauses': pause_features,
            'speaking_rate': rate_features
        }
