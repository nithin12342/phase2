"""
H5-OmniFusion Audio Pipeline Enhancements
Steps 1-11 from 40-Step Production Pipeline + Advanced Innovations
"""

import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
from typing import Dict, List, Tuple, Optional

class LoudnessNormalizer:
    """EBU R128 LUFS loudness normalization to -23 LUFS."""
    
    def __init__(self, target_lufs: float = -23.0, sr: int = 16000):
        self.target_lufs = target_lufs
        self.sr = sr
    
    def _compute_lufs(self, waveform: np.ndarray) -> float:
        """Compute integrated loudness in LUFS (simplified K-weighting)."""
        b, a = scipy.signal.butter(2, 1500 / (self.sr / 2), btype='high')
        filtered = scipy.signal.filtfilt(b, a, waveform)
        ms = np.mean(filtered ** 2)
        if ms < 1e-10:
            return -70.0
        return -0.691 + 10 * np.log10(ms)
    
    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to target LUFS."""
        current_lufs = self._compute_lufs(waveform)
        if current_lufs > -70:
            gain_db = self.target_lufs - current_lufs
            gain = 10 ** (gain_db / 20)
            return waveform * gain
        return waveform


class PeakNormalizer:
    """Normalize peak amplitude to [-1, 1] range."""
    
    @staticmethod
    def normalize(waveform: np.ndarray) -> np.ndarray:
        max_amp = np.max(np.abs(waveform))
        if max_amp > 0:
            return waveform / max_amp
        return waveform


class NoiseReducer:
    """Spectral gating noise reduction."""
    
    def __init__(self, sr: int = 16000, prop_decrease: float = 0.8):
        self.sr = sr
        self.prop_decrease = prop_decrease
    
    def reduce(self, waveform: np.ndarray) -> np.ndarray:
        """Apply spectral gating for noise reduction."""
        try:
            import noisereduce as nr
            noise_sample = waveform[:int(self.sr * 0.5)]
            return nr.reduce_noise(y=waveform, sr=self.sr, y_noise=noise_sample, 
                                   prop_decrease=self.prop_decrease)
        except ImportError:
            return self._simple_spectral_subtraction(waveform)
    
    def _simple_spectral_subtraction(self, waveform: np.ndarray) -> np.ndarray:
        """Simple spectral subtraction fallback."""
        stft = librosa.stft(waveform)
        mag, phase = np.abs(stft), np.angle(stft)
        noise_floor = np.mean(mag[:, :int(0.5 * self.sr / 512)], axis=1, keepdims=True)
        mag_clean = np.maximum(mag - noise_floor * 0.5, 0)
        return librosa.istft(mag_clean * np.exp(1j * phase))


class VoiceActivityDetector:
    """Detect voiced speech regions."""
    
    def __init__(self, sr: int = 16000, top_db: int = 30):
        self.sr = sr
        self.top_db = top_db
    
    def detect(self, waveform: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of (start, end) sample indices for voiced regions."""
        intervals = librosa.effects.split(waveform, top_db=self.top_db,
                                          frame_length=2048, hop_length=512)
        return [(int(s), int(e)) for s, e in intervals]
    
    def extract_voiced(self, waveform: np.ndarray) -> np.ndarray:
        """Extract only voiced segments concatenated."""
        intervals = self.detect(waveform)
        voiced = [waveform[s:e] for s, e in intervals]
        return np.concatenate(voiced) if voiced else waveform


class AudioSegmenter:
    """Segment audio into overlapping windows."""
    
    def __init__(self, sr: int = 16000, window_sec: float = 10.0, overlap: float = 0.5):
        self.sr = sr
        self.window_samples = int(window_sec * sr)
        self.hop_samples = int(self.window_samples * (1 - overlap))
    
    def segment(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Return list of audio segments."""
        segments = []
        for start in range(0, len(waveform) - self.window_samples + 1, self.hop_samples):
            segments.append(waveform[start:start + self.window_samples])
        if len(waveform) > self.window_samples and len(waveform) % self.hop_samples != 0:
            segments.append(waveform[-self.window_samples:])
        if not segments:
            segments.append(waveform)
        return segments


class BreathIntervalAnalyzer:
    """Analyze breath patterns - respiratory irregularity biomarker."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def extract(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract breath interval features."""
        features = {
            'breath_interval_mean': 0.0,
            'breath_interval_std': 0.0,
            'breath_interval_cv': 0.0,
            'breath_count': 0
        }
        
        try:
            hop = int(self.sr * 0.05)  # 50ms
            rms = librosa.feature.rms(y=waveform, hop_length=hop)[0]
            
            smoothed = scipy.ndimage.gaussian_filter1d(rms, sigma=3)
            troughs, _ = scipy.signal.find_peaks(-smoothed, distance=20)
            
            if len(troughs) > 1:
                intervals = np.diff(troughs) * 0.05  # Convert to seconds
                features['breath_interval_mean'] = float(np.mean(intervals))
                features['breath_interval_std'] = float(np.std(intervals))
                features['breath_interval_cv'] = float(np.std(intervals) / (np.mean(intervals) + 1e-8))
                features['breath_count'] = len(troughs)
        except Exception:
            pass
        
        return features


class SighDetector:
    """Detect sighing patterns - depression biomarker."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.min_duration = 1.0  # seconds
        self.max_duration = 3.0  # seconds
        self.max_freq = 500  # Hz dominant
    
    def detect(self, waveform: np.ndarray) -> Dict[str, float]:
        """Detect sighs based on duration, frequency, and energy decay."""
        features = {'sigh_count': 0, 'sigh_rate_per_min': 0.0, 'sigh_total_duration': 0.0}
        
        try:
            hop = int(self.sr * 0.1)
            rms = librosa.feature.rms(y=waveform, hop_length=hop)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.sr, hop_length=hop)[0]
            
            low_freq = spectral_centroid < self.max_freq
            threshold_low = np.percentile(rms, 10)
            threshold_high = np.percentile(rms, 40)
            candidates = (rms > threshold_low) & (rms < threshold_high) & low_freq
            
            labeled, n = scipy.ndimage.label(candidates)
            sighs = 0
            total_dur = 0.0
            
            for i in range(1, n + 1):
                duration = np.sum(labeled == i) * 0.1
                if self.min_duration <= duration <= self.max_duration:
                    sighs += 1
                    total_dur += duration
            
            duration_min = len(waveform) / self.sr / 60
            features['sigh_count'] = sighs
            features['sigh_rate_per_min'] = sighs / duration_min if duration_min > 0 else 0
            features['sigh_total_duration'] = total_dur
        except Exception:
            pass
        
        return features


class AudioQualityChecker:
    """Check audio quality against thresholds from documentation."""
    
    SNR_MIN = 15  # dB
    CLIPPING_MAX = 0.01  # 1%
    VOICE_ACTIVITY_MIN = 0.40  # 40%
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def check(self, waveform: np.ndarray) -> Dict[str, float]:
        """Return quality metrics and pass/fail flags."""
        signal_power = np.mean(waveform ** 2)
        noise_region = waveform[np.abs(waveform) < 0.01]
        noise_power = np.mean(noise_region ** 2) + 1e-10
        snr = 10 * np.log10(signal_power / noise_power)
        
        clipping_rate = np.mean(np.abs(waveform) > 0.99)
        
        vad = VoiceActivityDetector(self.sr)
        intervals = vad.detect(waveform)
        voiced_samples = sum(e - s for s, e in intervals)
        voice_ratio = voiced_samples / len(waveform)
        
        return {
            'snr_db': float(snr),
            'snr_pass': snr >= self.SNR_MIN,
            'clipping_rate': float(clipping_rate),
            'clipping_pass': clipping_rate <= self.CLIPPING_MAX,
            'voice_activity_ratio': float(voice_ratio),
            'voice_activity_pass': voice_ratio >= self.VOICE_ACTIVITY_MIN,
            'overall_quality_score': float(min(1.0, (snr / 30) * (1 - clipping_rate) * voice_ratio))
        }


class PauseAnalyzer:
    """Analyze pause patterns - key depression biomarker."""
    
    PAUSE_MIN_MS = 200  # Minimum pause duration
    PAUSE_ENERGY_THRESHOLD_DB = -30
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def analyze(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract pause features."""
        hop = int(self.sr * 0.01)  # 10ms frames
        rms = librosa.feature.rms(y=waveform, hop_length=hop)[0]
        rms_db = 20 * np.log10(rms + 1e-10)
        
        silent = rms_db < self.PAUSE_ENERGY_THRESHOLD_DB
        min_frames = int(self.PAUSE_MIN_MS / 10)
        
        pauses = []
        count = 0
        for is_silent in silent:
            if is_silent:
                count += 1
            else:
                if count >= min_frames:
                    pauses.append(count * 0.01)  # Convert to seconds
                count = 0
        if count >= min_frames:
            pauses.append(count * 0.01)
        
        duration = len(waveform) / self.sr
        
        return {
            'pause_count': len(pauses),
            'pause_duration_mean': float(np.mean(pauses)) if pauses else 0,
            'pause_duration_std': float(np.std(pauses)) if pauses else 0,
            'pause_duration_max': float(np.max(pauses)) if pauses else 0,
            'pause_total': float(sum(pauses)),
            'pause_ratio': float(sum(pauses) / duration) if duration > 0 else 0
        }


class ProsodicFingerprint:
    """Generate learned 32-dim embedding of speech rhythm and pause distributions.
    
    Captures temporal "shape" of speech patterns as depression biomarker.
    Reference: implementation_plan.md ADV3 specification.
    """
    
    def __init__(self, sr: int = 16000, output_dim: int = 32):
        self.sr = sr
        self.output_dim = output_dim
        self.pause_analyzer = PauseAnalyzer(sr)
        self.breath_analyzer = BreathIntervalAnalyzer(sr)
        self.sigh_detector = SighDetector(sr)
    
    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Extract 32-dimensional prosodic fingerprint embedding.
        
        Args:
            waveform: Audio waveform array at self.sr sample rate
            
        Returns:
            32-dim numpy array representing prosodic fingerprint
        """
        features = []
        
        pause_feats = self.pause_analyzer.analyze(waveform)
        features.extend([
            pause_feats.get('pause_count', 0) / 50,  # Normalized
            pause_feats.get('pause_duration_mean', 0) / 2,
            pause_feats.get('pause_duration_std', 0) / 1,
            pause_feats.get('pause_duration_max', 0) / 5,
            pause_feats.get('pause_ratio', 0),
            min(pause_feats.get('pause_total', 0) / 30, 1),
            0.0,  # Reserved
            0.0   # Reserved
        ])
        
        breath_feats = self.breath_analyzer.extract(waveform)
        features.extend([
            breath_feats.get('breath_interval_mean', 0) / 5,
            breath_feats.get('breath_interval_std', 0) / 2,
            breath_feats.get('breath_interval_cv', 0),
            breath_feats.get('breath_count', 0) / 30,
            0.0, 0.0, 0.0, 0.0  # Reserved
        ])
        
        sigh_feats = self.sigh_detector.detect(waveform)
        features.extend([
            sigh_feats.get('sigh_count', 0) / 10,
            sigh_feats.get('sigh_rate_per_min', 0) / 5,
            sigh_feats.get('sigh_total_duration', 0) / 10,
            0.0  # Reserved
        ])
        
        try:
            hop = int(self.sr * 0.02)
            rms = librosa.feature.rms(y=waveform, hop_length=hop)[0]
            
            peaks, _ = scipy.signal.find_peaks(rms, distance=5)
            duration_sec = len(waveform) / self.sr
            speaking_rate = len(peaks) / duration_sec if duration_sec > 0 else 0
            
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            energy_range = np.max(rms) - np.min(rms)
            
            vad = VoiceActivityDetector(self.sr)
            intervals = vad.detect(waveform)
            voiced_samples = sum(e - s for s, e in intervals)
            phonation_ratio = voiced_samples / len(waveform) if len(waveform) > 0 else 0
            
            features.extend([
                min(speaking_rate / 10, 1),
                min(energy_mean * 10, 1),
                min(energy_std * 10, 1),
                min(energy_range, 1),
                phonation_ratio,
                0.0, 0.0, 0.0  # Reserved
            ])
        except Exception as e:
            print(f"Warning [ProsodicFingerprint/SpeakingRate]: {e}")
            features.extend([0.0] * 8)
        
        try:
            autocorr = np.correlate(rms, rms, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-8)
            
            peaks_ac, _ = scipy.signal.find_peaks(autocorr, distance=5)
            rhythm_regularity = autocorr[peaks_ac[0]] if len(peaks_ac) > 0 else 0
            
            features.extend([
                rhythm_regularity,
                np.std(np.diff(peaks_ac)) / 10 if len(peaks_ac) > 1 else 0,
                0.0, 0.0
            ])
        except Exception as e:
            print(f"Warning [ProsodicFingerprint/Rhythm]: {e}")
            features.extend([0.0] * 4)
        
        fingerprint = np.array(features[:self.output_dim], dtype=np.float32)
        if len(fingerprint) < self.output_dim:
            fingerprint = np.pad(fingerprint, (0, self.output_dim - len(fingerprint)))
        
        fingerprint = np.clip(fingerprint, 0, 1)
        
        return fingerprint
    
    def extract_dict(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract fingerprint as dictionary for compatibility."""
        fp = self.extract(waveform)
        return {f'prosodic_fp_{i}': float(v) for i, v in enumerate(fp)}
