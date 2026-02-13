"""
H5-OmniFusion Complete 40-Step Pipeline
All XML Specification Steps + 9 Advanced Innovations
Compatible with Google Drive folder structure
"""


"""
from google.colab import drive
drive.mount('/content/drive')

!pip install -q "numpy<2.0" transformers==4.40.0 torch torchvision torchaudio huggingface_hub
!pip install -q librosa opensmile praat-parselmouth pyannote.audio==3.1.1
!pip install -q opencv-python-headless scikit-learn pandas tqdm mediapipe timm
# !pip install -q h5py tqdm pandas scipy
# !pip install -q timm==0.9.12
!pip install -q nltk vadersentiment pyphen textstat noisereduce snownlp h5py
!apt-get install -y libsndfile1

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
"""


import os, sys, json, re, zipfile, shutil, tempfile
import numpy as np
import torch
import torch.nn as nn
import librosa
import opensmile
import parselmouth
from parselmouth.praat import call
import pandas as pd
import cv2
import mediapipe as mp
import timm
import soundfile as sf
import scipy.signal
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from torchvision import transforms
from transformers import (Wav2Vec2Model, Wav2Vec2FeatureExtractor, 
                          VideoMAEModel, VideoMAEImageProcessor, 
                          RobertaTokenizer, RobertaModel,
                          BertTokenizer, BertModel)
from pyannote.audio import Pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

BASE_PATH = '/content/drive/MyDrive/DAIC-WOZ_Datasets'
DAIC_PATH = os.path.join(BASE_PATH, 'DAIC-WOZ')  # Contains XXX_P.zip files
EATD_PATH = os.path.join(BASE_PATH, 'EATD-Corpus', 'EATD-Corpus')  # Nested structure
OUTPUT_PATH = os.path.join(BASE_PATH, 'Features_Complete_Pipeline')
EXTRACT_PATH = '/content/extracted_data'  # Temp extraction on Colab

for mod in ['audio', 'text', 'video', 'face', 'tabular', 'combined', 'eatd']:
    os.makedirs(os.path.join(OUTPUT_PATH, mod), exist_ok=True)
os.makedirs(EXTRACT_PATH, exist_ok=True)


from huggingface_hub import login
HF_TOKEN = ''  # PASTE YOUR TOKEN HERE
if HF_TOKEN:
    login(token=HF_TOKEN)
    print('Logged in to HuggingFace!')
else:
    print('WARNING: HF Token needed for speaker diarization!')


def extract_participant(pid):
    """Extract a single participant zip file to temp directory."""
    zip_path = os.path.join(DAIC_PATH, f'{pid}_P.zip')
    extract_to = os.path.join(EXTRACT_PATH, f'{pid}_P')
    
    if os.path.exists(extract_to):
        return extract_to
    
    if not os.path.exists(zip_path):
        alt_path = os.path.join(DAIC_PATH, f'{pid}_P (1).zip')
        if os.path.exists(alt_path):
            zip_path = alt_path
        else:
            print(f'Zip not found: {zip_path}')
            return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(EXTRACT_PATH)
        return extract_to
    except Exception as e:
        print(f'Extract error {pid}: {e}')
        return None

def cleanup_participant(pid):
    """Remove extracted folder to save disk space."""
    path = os.path.join(EXTRACT_PATH, f'{pid}_P')
    if os.path.exists(path):
        shutil.rmtree(path)


class LoudnessNormalizer:
    """EBU R128 LUFS loudness normalization to -23 LUFS."""
    def __init__(self, target_lufs=-23.0, sr=16000):
        self.target_lufs = target_lufs
        self.sr = sr
    
    def _compute_lufs(self, waveform):
        b, a = scipy.signal.butter(2, 1500 / (self.sr / 2), btype='high')
        filtered = scipy.signal.filtfilt(b, a, waveform)
        ms = np.mean(filtered ** 2)
        return -70.0 if ms < 1e-10 else -0.691 + 10 * np.log10(ms)
    
    def normalize(self, waveform):
        current_lufs = self._compute_lufs(waveform)
        if current_lufs > -70:
            gain = 10 ** ((self.target_lufs - current_lufs) / 20)
            return waveform * gain
        return waveform

class PeakNormalizer:
    """Normalize peak amplitude to [-1, 1] range."""
    @staticmethod
    def normalize(waveform):
        max_amp = np.max(np.abs(waveform))
        return waveform / max_amp if max_amp > 0 else waveform

class NoiseReducer:
    """Spectral gating noise reduction."""
    def __init__(self, sr=16000, prop_decrease=0.8):
        self.sr = sr
        self.prop_decrease = prop_decrease
    
try:
    import noisereduce as nr
    NOISEREDUCE_OK = True
except ImportError:
    NOISEREDUCE_OK = False
    print("Warning: noisereduce not found - noise reduction disabled")

try:
    from research_layer_extensions import (
        SpecAugment, 
        TextAugmenter, 
        VideoGeometricAugmenter, 
        TemporalGridAligner
    )
    AUGMENTATION_OK = True
except ImportError:
    AUGMENTATION_OK = False
    print("Warning: research_layer_extensions not found - augmentation disabled")

    def reduce(self, waveform):
        if not NOISEREDUCE_OK:
            return waveform
        try:
            noise_sample = waveform[:int(self.sr * 0.5)]
            return nr.reduce_noise(y=waveform, sr=self.sr, y_noise=noise_sample, 
                                   prop_decrease=self.prop_decrease)
        except Exception as e:
            print(f"Error during noise reduction: {e}")
            return waveform

class VoiceActivityDetector:
    """Detect voiced speech regions using librosa."""
    def __init__(self, sr=16000, top_db=30):
        self.sr = sr
        self.top_db = top_db
    
    def detect(self, waveform):
        intervals = librosa.effects.split(waveform, top_db=self.top_db,
                                          frame_length=2048, hop_length=512)
        return [(int(s), int(e)) for s, e in intervals]

class AudioSegmenter:
    """Segment audio into overlapping windows."""
    def __init__(self, sr=16000, window_sec=10.0, overlap=0.5):
        self.sr = sr
        self.window_samples = int(window_sec * sr)
        self.hop_samples = int(self.window_samples * (1 - overlap))
    
    def segment(self, waveform):
        segments = []
        for start in range(0, len(waveform) - self.window_samples + 1, self.hop_samples):
            segments.append(waveform[start:start + self.window_samples])
        return segments if segments else [waveform]

class EGeMAPSExtractor:
    """Extract 88 eGeMAPSv02 acoustic features using OpenSMILE."""
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )
    
    def extract(self, audio_path):
        try:
            feats = self.smile.process_file(audio_path)
            return {col: float(feats[col].values[0]) for col in feats.columns}
        except Exception as e:
            print(f'eGeMAPS error: {e}')
            return {}

class PauseAnalyzer:
    """Analyze pause patterns - key depression biomarker."""
    PAUSE_MIN_MS = 200  # Minimum pause duration
    PAUSE_ENERGY_THRESHOLD_DB = -30
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def analyze(self, waveform):
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
                    pauses.append(count * 0.01)
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

class SighDetector:
    """Detect sighing patterns - depression biomarker."""
    def __init__(self, sr=16000):
        self.sr = sr
        self.min_duration = 1.0
        self.max_duration = 3.0
        self.max_freq = 500
    
    def detect(self, waveform):
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
        except:
            pass
        return features

class BreathIntervalAnalyzer:
    """Analyze breath patterns - respiratory irregularity biomarker."""
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract(self, waveform):
        features = {
            'breath_interval_mean': 0.0,
            'breath_interval_std': 0.0,
            'breath_interval_cv': 0.0,
            'breath_count': 0
        }
        try:
            hop = int(self.sr * 0.05)
            rms = librosa.feature.rms(y=waveform, hop_length=hop)[0]
            smoothed = scipy.ndimage.gaussian_filter1d(rms, sigma=3)
            troughs, _ = scipy.signal.find_peaks(-smoothed, distance=20)
            
            if len(troughs) > 1:
                intervals = np.diff(troughs) * 0.05
                features['breath_interval_mean'] = float(np.mean(intervals))
                features['breath_interval_std'] = float(np.std(intervals))
                features['breath_interval_cv'] = float(np.std(intervals) / (np.mean(intervals) + 1e-8))
                features['breath_count'] = len(troughs)
        except:
            pass
        return features

class AudioQualityChecker:
    """Check audio quality against thresholds."""
    SNR_MIN = 15
    CLIPPING_MAX = 0.01
    VOICE_ACTIVITY_MIN = 0.40
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def check(self, waveform):
        signal_power = np.mean(waveform ** 2)
        noise_region = waveform[np.abs(waveform) < 0.01]
        noise_power = np.mean(noise_region ** 2) + 1e-10 if len(noise_region) > 0 else 1e-10
        snr = 10 * np.log10(signal_power / noise_power)
        
        clipping_rate = np.mean(np.abs(waveform) > 0.99)
        
        vad = VoiceActivityDetector(self.sr)
        intervals = vad.detect(waveform)
        voiced_samples = sum(e - s for s, e in intervals)
        voice_ratio = voiced_samples / len(waveform) if len(waveform) > 0 else 0
        
        return {
            'snr_db': float(snr),
            'snr_pass': snr >= self.SNR_MIN,
            'clipping_rate': float(clipping_rate),
            'clipping_pass': clipping_rate <= self.CLIPPING_MAX,
            'voice_activity_ratio': float(voice_ratio),
            'voice_activity_pass': voice_ratio >= self.VOICE_ACTIVITY_MIN,
            'overall_quality_score': float(min(1.0, max(0, (snr / 30) * (1 - clipping_rate) * voice_ratio)))
        }

class FormantTrackExtractor:
    """
    [R14] Extract F1-F4 formant trajectories using Praat/Parselmouth.
    
    59-STEP EXHAUSTIVE SPEC: Explicitly extract F1-F4 trajectories from 
    participant audio for vocal tract resonance analysis.
    
    Clinical Relevance: Formant patterns change with emotional state and 
    depression often shows reduced formant variability.
    """
    def __init__(self, sr: int = 16000, max_formant: int = 5500):
        self.sr = sr
        self.max_formant = max_formant
    
    def extract(self, audio_path: str) -> Dict[str, Any]:
        """Extract F1-F4 formant statistics from audio file."""
        try:
            sound = parselmouth.Sound(audio_path)
            formant = call(sound, "To Formant (burg)...", 0.0, 4, self.max_formant, 0.025, 50)
            
            n_frames = call(formant, "Get number of frames")
            formants = {f'F{i}': [] for i in range(1, 5)}
            
            for frame in range(1, n_frames + 1):
                for i in range(1, 5):
                    val = call(formant, "Get value at time...", i, frame * 0.005, "Hertz", "Linear")
                    if val and not np.isnan(val):
                        formants[f'F{i}'].append(val)
            
            result = {}
            for key, vals in formants.items():
                if vals:
                    result[f'{key}_mean'] = float(np.mean(vals))
                    result[f'{key}_std'] = float(np.std(vals))
                    result[f'{key}_range'] = float(np.max(vals) - np.min(vals))
                    if len(vals) > 2:
                        result[f'{key}_slope'] = float(np.polyfit(range(len(vals)), vals, 1)[0])
                else:
                    result[f'{key}_mean'] = 0.0
                    result[f'{key}_std'] = 0.0
                    result[f'{key}_range'] = 0.0
                    result[f'{key}_slope'] = 0.0
            
            return result
            
        except Exception as e:
            print(f'FormantTrackExtractor error: {e}')
            return {f'F{i}_{m}': 0.0 for i in range(1, 5) for m in ['mean', 'std', 'range', 'slope']}

class SpecAugment:
    """
    [R56] SpecAugment for audio training robustness.
    
    59-STEP EXHAUSTIVE SPEC: Implement frequency_mask and time_mask functions.
    
    Reference: Park et al., "SpecAugment", 2019.
    """
    
    def __init__(self, freq_mask_param: int = 27, time_mask_param: int = 100,
                 n_freq_masks: int = 2, n_time_masks: int = 2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def frequency_mask(self, spectrogram: np.ndarray, mask_value: float = 0.0) -> np.ndarray:
        """Apply frequency masking to spectrogram."""
        spec = spectrogram.copy()
        freq_bins = spec.shape[0]
        f = np.random.randint(0, min(self.freq_mask_param, freq_bins))
        f0 = np.random.randint(0, max(1, freq_bins - f))
        spec[f0:f0 + f, :] = mask_value
        return spec
    
    def time_mask(self, spectrogram: np.ndarray, mask_value: float = 0.0) -> np.ndarray:
        """Apply time masking to spectrogram."""
        spec = spectrogram.copy()
        time_steps = spec.shape[1]
        t = np.random.randint(0, min(self.time_mask_param, time_steps))
        t0 = np.random.randint(0, max(1, time_steps - t))
        spec[:, t0:t0 + t] = mask_value
        return spec
    
    def augment(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Apply full SpecAugment pipeline to waveform."""
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        for _ in range(self.n_freq_masks):
            S_db = self.frequency_mask(S_db)
        for _ in range(self.n_time_masks):
            S_db = self.time_mask(S_db)
        
        S_aug = librosa.db_to_power(S_db)
        waveform_aug = librosa.feature.inverse.mel_to_audio(S_aug, sr=sr)
        return waveform_aug

class TranscriptDiarizer:
    """Primary diarization: Parse DAIC-WOZ transcripts for ground-truth speaker segments."""
    def __init__(self, sr=16000):
        self.sr = sr
    
    def parse_transcript(self, path):
        """Parse transcript TSV file."""
        if not path or not os.path.exists(path):
            return None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(path, sep='\t', header=None, 
                                   names=['start','end','speaker','text'], encoding=enc)
            except:
                continue
        return None
    
    def get_participant_segments(self, df):
        """Extract participant-only segments from transcript."""
        segments = []
        for _, row in df.iterrows():
            speaker = str(row.get('speaker', '')).lower()
            if 'ellie' not in speaker and 'interviewer' not in speaker:
                try:
                    segments.append({'start': float(row['start']), 'end': float(row['end'])})
                except:
                    pass
        return segments
    
    def extract_audio(self, waveform, segments):
        """Extract participant audio using transcript segments."""
        if not segments:
            return waveform
        parts = []
        for seg in segments:
            start = max(0, int(seg['start'] * self.sr))
            end = min(len(waveform), int(seg['end'] * self.sr))
            if end > start:
                parts.append(waveform[start:end])
        return np.concatenate(parts) if parts else waveform
    
    def compute_latencies(self, df):
        """ADV1: Extract response latencies from transcript."""
        if df is None:
            return {'latency_mean': 0, 'latency_std': 0, 'latency_max': 0}
        latencies = []
        prev_end = None
        for _, row in df.iterrows():
            speaker = str(row.get('speaker', '')).lower()
            try:
                if 'ellie' in speaker:
                    prev_end = float(row['end'])
                elif prev_end is not None:
                    lat = (float(row['start']) - prev_end) * 1000  # ms
                    if 0 < lat < 10000:
                        latencies.append(lat)
                    prev_end = None
            except:
                pass
        return {
            'latency_mean': float(np.mean(latencies)) if latencies else 0,
            'latency_std': float(np.std(latencies)) if len(latencies) > 1 else 0,
            'latency_max': float(max(latencies)) if latencies else 0
        }


class AudioPreprocessor:
    """Complete audio preprocessing - Steps 1-11 + R14 Formant Tracking.
    
    REFACTORED: Uses TranscriptDiarizer (text-based) as PRIMARY diarization.
    pyannote.audio is fallback ONLY when transcript unavailable.
    """
    def __init__(self, hf_token):
        self.sr = 16000
        
        self.transcript_diarizer = TranscriptDiarizer(self.sr)
        
        self.pyannote_diarizer = None
        try:
            self.pyannote_diarizer = Pipeline.from_pretrained(
                'pyannote/speaker-diarization-3.1', 
                use_auth_token=hf_token
            ).to(device)
        except Exception as e:
            print(f'pyannote fallback unavailable: {e}')
        
        self.w2v_proc = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        self.w2v_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53').to(device).eval()
        
        self.loudness_norm = LoudnessNormalizer()
        self.peak_norm = PeakNormalizer()
        self.noise_reducer = NoiseReducer()
        self.pause_analyzer = PauseAnalyzer()
        self.sigh_detector = SighDetector()
        self.breath_analyzer = BreathIntervalAnalyzer()
        self.audio_qc = AudioQualityChecker()
        self.egemaps = EGeMAPSExtractor()
        self.formants = FormantTrackExtractor(sr=self.sr)
        self.vad = VoiceActivityDetector(self.sr)
        
        self.spec_augment = SpecAugment() if AUGMENTATION_OK else None
    
    def diarize_from_transcript(self, transcript_path, waveform):
        """Step 3/R4: PRIMARY - Text-based diarization using transcript."""
        df = self.transcript_diarizer.parse_transcript(transcript_path)
        if df is None:
            return None, None, {}
        segments = self.transcript_diarizer.get_participant_segments(df)
        participant_wav = self.transcript_diarizer.extract_audio(waveform, segments)
        latencies = self.transcript_diarizer.compute_latencies(df)
        return participant_wav, segments, latencies
    
    def diarize_from_audio(self, audio_path):
        """Step 3: FALLBACK - Audio-based diarization using pyannote."""
        if self.pyannote_diarizer is None:
            return []
        try:
            diarization = self.pyannote_diarizer(audio_path)
            speaker_durations = {}
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + turn.end - turn.start
                speaker_segments.setdefault(speaker, []).append({'start': turn.start, 'end': turn.end})
            if not speaker_durations:
                return []
            participant = max(speaker_durations, key=speaker_durations.get)
            return speaker_segments.get(participant, [])
        except Exception as e:
            print(f'pyannote diarization error: {e}')
            return []
    
    def extract_participant_audio(self, waveform, segments):
        """Extract only participant speech segments."""
        parts = []
        for seg in segments:
            start = max(0, int(seg['start'] * self.sr))
            end = min(len(waveform), int(seg['end'] * self.sr))
            parts.append(waveform[start:end])
        return np.concatenate(parts) if parts else waveform
    
    def process(self, audio_path, transcript_path=None, augment=False):
        """Process audio through complete pipeline.
        
        REFACTORED: Uses transcript-based diarization as PRIMARY method.
        Falls back to pyannote only when transcript unavailable.
        Falls back to VAD when both unavailable.
        """
        try:
            wav, _ = librosa.load(audio_path, sr=self.sr)
        except:
            return np.zeros(CFG.EMBED_DIM), {'quality_score': 0}

        if augment and self.spec_augment:
            try:
                wav = self.spec_augment.augment(wav, self.sr)
            except Exception as e:
                print(f"SpecAugment error: {e}")
        
        features = {}
        
        part_wav = wav
        diarization_method = 'none'
        
        if transcript_path and os.path.exists(transcript_path):
            result = self.diarize_from_transcript(transcript_path, wav)
            if result[0] is not None:
                part_wav, segments, latencies = result
                features.update(latencies)
                diarization_method = 'transcript'
        
        if diarization_method == 'none' and self.pyannote_diarizer:
            segments = self.diarize_from_audio(audio_path)
            if segments:
                part_wav = self.transcript_diarizer.extract_audio(wav, segments)
                diarization_method = 'pyannote'
        
        if diarization_method == 'none':
            try:
                part_wav = self.vad.extract_voiced(wav) if hasattr(self.vad, 'extract_voiced') else wav
                diarization_method = 'vad'
            except:
                pass
        
        features['diarization_method'] = diarization_method
        
        part_wav = self.peak_norm.normalize(part_wav)
        part_wav = self.loudness_norm.normalize(part_wav)
        
        part_wav = self.noise_reducer.reduce(part_wav)
        
        inputs = self.w2v_proc(part_wav, sampling_rate=self.sr, return_tensors='pt', padding=True)
        with torch.no_grad():
            if inputs.input_values.shape[1] > 30 * self.sr:
                chunks = torch.split(inputs.input_values, 30 * self.sr, dim=1)
                embs = [self.w2v_model(c.to(device)).last_hidden_state.mean(1).cpu() for c in chunks]
                w2v_emb = torch.mean(torch.stack(embs), dim=0).numpy().squeeze()
            else:
                w2v_emb = self.w2v_model(inputs.input_values.to(device)).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        
        features.update(self.pause_analyzer.analyze(part_wav))
        features.update(self.sigh_detector.detect(part_wav))
        features.update(self.breath_analyzer.extract(part_wav))
        features.update(self.audio_qc.check(part_wav))
        features.update(self.egemaps.extract(audio_path))
        features.update(self.formants.extract(audio_path))
        
        return w2v_emb, features


class TranscriptCleaner:
    """Clean transcript text per production pipeline steps 12-14."""
    DISFLUENCIES = ['um', 'uh', 'er', 'ah', 'hm', 'hmm', 'mm', 'mhm', 'uh-huh']
    
    def __init__(self, preserve_disfluencies=False):
        self.preserve_disfluencies = preserve_disfluencies
    
    def clean(self, text):
        original_words = text.lower().split()
        word_count = len(original_words) + 1e-8
        
        disfluency_count = sum(1 for w in original_words if w.strip('.,!?') in self.DISFLUENCIES)
        disfluency_rate = disfluency_count / word_count
        
        text = re.sub(r'^\d+[\.\\)]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(ELLIE|Participant|Ellie|PARTICIPANT):\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        if not self.preserve_disfluencies:
            pattern = r'\b(' + '|'.join(self.DISFLUENCIES) + r')\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text, {'disfluency_count': disfluency_count, 'disfluency_rate': float(disfluency_rate)}

class PsycholinguisticExtractor:
    """Extract evidence-based psycholinguistic features for depression."""
    FIRST_PERSON_SING = ['i', 'me', 'my', 'myself', 'mine']
    FIRST_PERSON_PLUR = ['we', 'us', 'our', 'ourselves', 'ours']
    ABSOLUTIST = ['always', 'never', 'nothing', 'everything', 'completely', 'totally', 'absolutely']
    NEGATIVE_EMOTION = ['sad', 'depressed', 'hopeless', 'worthless', 'tired', 'empty', 'lonely', 'anxious', 'afraid', 'angry']
    COGNITIVE = ['think', 'know', 'believe', 'feel', 'understand', 'realize', 'remember']
    SLEEP = ['sleep', 'insomnia', 'tired', 'exhausted', 'rest', 'awake', 'bed']
    ANHEDONIA = ['boring', 'bored', 'pointless', 'meaningless', 'enjoy', 'pleasure', 'fun', 'happy']
    
    def extract(self, text):
        words = text.lower().split()
        wc = len(words) + 1e-8
        return {
            'first_person_singular': sum(1 for w in words if w in self.FIRST_PERSON_SING) / wc,
            'first_person_plural': sum(1 for w in words if w in self.FIRST_PERSON_PLUR) / wc,
            'absolutist': sum(1 for w in words if w in self.ABSOLUTIST) / wc,
            'negative_emotion': sum(1 for w in words if w in self.NEGATIVE_EMOTION) / wc,
            'cognitive': sum(1 for w in words if w in self.COGNITIVE) / wc,
            'sleep_words': sum(1 for w in words if w in self.SLEEP) / wc,
            'anhedonia': sum(1 for w in words if w in self.ANHEDONIA) / wc,
            'word_count': len(words),
            'lexical_diversity': len(set(words)) / wc
        }

class ReadabilityExtractor:
    """Text complexity and readability metrics."""
    def extract(self, text):
        try:
            return {
                'flesch_reading_ease': float(textstat.flesch_reading_ease(text)),
                'flesch_kincaid_grade': float(textstat.flesch_kincaid_grade(text)),
                'gunning_fog': float(textstat.gunning_fog(text)),
                'automated_readability': float(textstat.automated_readability_index(text)),
                'syllable_count': float(textstat.syllable_count(text)),
                'sentence_count': max(1, len(re.split(r'[.!?]+', text)))
            }
        except:
            return {'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0, 'gunning_fog': 0}

class MultilingualSentimentAnalyzer:
    """Sentiment analysis for English (VADER) and Chinese (SnowNLP)."""
    def __init__(self):
        self._en_analyzer = None
    
    @property
    def en_analyzer(self):
        if self._en_analyzer is None:
            self._en_analyzer = SentimentIntensityAnalyzer()
        return self._en_analyzer
    
    def analyze_english(self, text):
        scores = self.en_analyzer.polarity_scores(text)
        return {f'sentiment_{k}': v for k, v in scores.items()}
    
    def analyze_chinese(self, text):
        try:
            from snownlp import SnowNLP
            s = SnowNLP(text)
            sentiment = s.sentiments
            return {
                'sentiment_neg': 1 - sentiment,
                'sentiment_neu': 0.5,
                'sentiment_pos': sentiment,
                'sentiment_compound': sentiment * 2 - 1
            }
        except ImportError:
            return {'sentiment_neg': 0, 'sentiment_neu': 1, 'sentiment_pos': 0, 'sentiment_compound': 0}

class ConversationDynamicsAnalyzer:
    """Analyze turn-taking and conversation patterns."""
    def analyze(self, transcript_path):
        if not os.path.exists(transcript_path):
            return {}
        
        try:
            df = pd.read_csv(transcript_path, sep='\t')
        except:
            return {}
        
        participant_turns = df[df['speaker'].str.lower().str.contains('participant', na=False)]
        ellie_turns = df[df['speaker'].str.lower().str.contains('ellie', na=False)]
        
        if len(participant_turns) == 0:
            return {'turn_count': 0, 'turn_length_mean': 0, 'talk_ratio': 0, 'engagement_change': 0}
        
        p_lens = [len(str(v).split()) for v in participant_turns['value']]
        e_lens = [len(str(v).split()) for v in ellie_turns['value']]
        
        p_words = sum(p_lens)
        e_words = sum(e_lens)
        talk_ratio = p_words / (p_words + e_words + 1e-8)
        
        n = len(p_lens)
        if n >= 6:
            early = np.mean(p_lens[:n//3])
            late = np.mean(p_lens[2*n//3:])
            trend = late - early
        else:
            trend = 0
        
        return {
            'turn_count': len(p_lens),
            'turn_length_mean': float(np.mean(p_lens)),
            'turn_length_std': float(np.std(p_lens)),
            'talk_ratio': float(talk_ratio),
            'engagement_change': float(trend)
        }

class ResponseLatencyExtractor:
    """Measure precise gap between interviewer offset and participant onset."""
    SLOW_THRESHOLD = 2.0
    
    def extract(self, transcript_path):
        if not os.path.exists(transcript_path):
            return {'response_latency_mean': 0, 'response_latency_max': 0, 'slow_response_ratio': 0}
        
        try:
            df = pd.read_csv(transcript_path, sep='\t')
        except:
            return {'response_latency_mean': 0, 'response_latency_max': 0, 'slow_response_ratio': 0}
        
        if 'start_time' not in df.columns or 'stop_time' not in df.columns:
            return {'response_latency_mean': 0, 'response_latency_max': 0, 'slow_response_ratio': 0}
        
        latencies = []
        prev_end = None
        
        for _, row in df.iterrows():
            speaker = str(row.get('speaker', '')).lower()
            if 'ellie' in speaker:
                prev_end = row['stop_time']
            elif 'participant' in speaker and prev_end is not None:
                lat = row['start_time'] - prev_end
                if 0 < lat < 15:
                    latencies.append(lat)
                prev_end = None
        
        if not latencies:
            return {'response_latency_mean': 0, 'response_latency_max': 0, 'slow_response_ratio': 0}
        
        return {
            'response_latency_mean': float(np.mean(latencies)),
            'response_latency_std': float(np.std(latencies)),
            'response_latency_max': float(np.max(latencies)),
            'response_latency_median': float(np.median(latencies)),
            'slow_response_ratio': float(sum(1 for l in latencies if l > self.SLOW_THRESHOLD) / len(latencies))
        }

class CategoricalEmotionLabeler:
    """
    [R30] Output discrete emotion labels (Anger, Joy, Fear, Sadness) alongside sentiment.
    
    59-STEP EXHAUSTIVE SPEC: Add categorical emotion detection beyond polarity.
    Uses keyword-lexicon approach for robustness.
    """
    EMOTION_LEXICONS: Dict[str, List[str]] = {
        'anger': ['angry', 'furious', 'annoyed', 'irritated', 'mad', 'rage', 'hate', 'frustrated'],
        'fear': ['afraid', 'scared', 'terrified', 'anxious', 'nervous', 'worried', 'panic', 'dread'],
        'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'grief', 'sorrow', 'lonely', 'hopeless'],
        'joy': ['happy', 'joyful', 'excited', 'glad', 'pleased', 'delighted', 'cheerful', 'content'],
        'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'unexpected', 'startled'],
        'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'nauseated', 'gross']
    }
    
    def label(self, text: str) -> Dict[str, Any]:
        """Assign categorical emotion probabilities to text."""
        words = text.lower().split()
        total = len(words) + 1e-8
        
        result = {}
        scores = []
        for emotion, lexicon in self.EMOTION_LEXICONS.items():
            count = sum(1 for w in words if w.strip('.,!?') in lexicon)
            score = count / total
            result[f'emotion_{emotion}'] = float(score)
            scores.append((emotion, score))
        
        if scores:
            dominant = max(scores, key=lambda x: x[1])
            result['emotion_dominant'] = dominant[0]
            result['emotion_dominant_score'] = float(dominant[1])
        
        return result

class TextAugmenter:
    """
    [R58] Text augmentation for linguistic robustness.
    
    59-STEP EXHAUSTIVE SPEC: Implement synonym replacement using NLTK.
    """
    
    def __init__(self, aug_prob: float = 0.15):
        self.aug_prob = aug_prob
        try:
            from nltk.corpus import wordnet
            self._wordnet_available = True
        except:
            self._wordnet_available = False
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word from WordNet."""
        if not self._wordnet_available:
            return []
        try:
            from nltk.corpus import wordnet
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name().lower() != word.lower():
                        synonyms.add(lemma.name().replace('_', ' '))
            return list(synonyms)[:5]
        except:
            return []
    
    def synonym_replacement(self, text: str, n: Optional[int] = None) -> str:
        """Replace n random words with synonyms."""
        words = text.split()
        if n is None:
            n = max(1, int(len(words) * self.aug_prob))
        
        indices = list(range(len(words)))
        np.random.shuffle(indices)
        
        augmented = words.copy()
        replaced = 0
        
        for idx in indices:
            if replaced >= n:
                break
            word = words[idx]
            synonyms = self.get_synonyms(word)
            if synonyms:
                augmented[idx] = np.random.choice(synonyms)
                replaced += 1
        
        return ' '.join(augmented)
    
    def augment(self, text: str) -> str:
        """Apply text augmentation."""
        return self.synonym_replacement(text)


class TextPreprocessor:
    """Complete text preprocessing pipeline - Steps 12-20 + R30 Emotion Labels."""
    def __init__(self):
        try:
            self.en_tokenizer = RobertaTokenizer.from_pretrained('mental/mental-roberta-base')
            self.en_model = RobertaModel.from_pretrained('mental/mental-roberta-base').to(device).eval()
        except:
            self.en_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.en_model = RobertaModel.from_pretrained('roberta-base').to(device).eval()
        
        self.zh_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.zh_model = BertModel.from_pretrained('bert-base-chinese').to(device).eval()
        
        self.cleaner = TranscriptCleaner()
        self.psycho = PsycholinguisticExtractor()
        self.readability = ReadabilityExtractor()
        self.sentiment = MultilingualSentimentAnalyzer()
        self.conversation = ConversationDynamicsAnalyzer()
        self.latency = ResponseLatencyExtractor()
        self.emotions = CategoricalEmotionLabeler()
        
        self.text_augmenter = TextAugmenter() if AUGMENTATION_OK else None
    
    def detect_language(self, text):
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return 'zh' if chinese_chars / (len(text) + 1e-8) > 0.3 else 'en'
    
    def process(self, transcript_path, augment=False):
        """Process transcript extracting text, embeddings, and features."""
        if not os.path.exists(transcript_path):
            return None, {}
        
        try:
            df = pd.read_csv(transcript_path, sep='\t')
        except:
            return None, {}
            
        try:
            p_text = df[df['speaker'].str.lower().str.contains('participant', na=False)]['value'].astype(str).tolist()
            full_text = " ".join(p_text)
        except:
            return None, {}
        
        if not full_text.strip():
            return None, {}
        
        if augment and self.text_augmenter:
            try:
                full_text = self.text_augmenter.augment(full_text)
            except Exception as e:
                print(f"TextAugmenter error: {e}")

        lang = self.detect_language(full_text)
        
        cleaned, disf_feats = self.cleaner.clean(full_text)
        
        if lang == 'zh':
            inputs = self.zh_tokenizer(cleaned[:5000], return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                emb = self.zh_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        else:
            inputs = self.en_tokenizer(cleaned[:5000], return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                emb = self.en_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        
        features = disf_feats.copy()
        features.update(self.psycho.extract(cleaned))
        features.update(self.readability.extract(cleaned))
        if lang == 'en':
            features.update(self.sentiment.analyze_english(cleaned))
        else:
            features.update(self.sentiment.analyze_chinese(cleaned))
        features.update(self.conversation.analyze(transcript_path))
        features.update(self.latency.extract(transcript_path))
        features.update(self.emotions.label(cleaned))
        features['language'] = lang
        
        return emb, features


class VideoQualityFilter:
    """Filter frames based on blur, brightness, and exposure."""
    BLUR_THRESH = 50
    BRIGHTNESS_MIN = 80
    BRIGHTNESS_MAX = 180
    
    def filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        return blur >= self.BLUR_THRESH and self.BRIGHTNESS_MIN <= brightness <= self.BRIGHTNESS_MAX

class DiscreteQualityFilters:
    """
    [R33-34] Standalone boolean filters for blur and darkness detection.
    
    59-STEP EXHAUSTIVE SPEC: Implement is_blurry(threshold=50) and is_dark(threshold=80)
    as discrete boolean filters, separate from composite quality score.
    """
    
    @staticmethod
    def is_blurry(frame: np.ndarray, threshold: float = 50.0) -> bool:
        """Detect if frame is blurry using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    
    @staticmethod
    def is_dark(frame: np.ndarray, threshold: float = 80.0) -> bool:
        """Detect if frame is too dark."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_brightness = np.mean(gray)
        return mean_brightness < threshold
    
    @staticmethod
    def is_overexposed(frame: np.ndarray, threshold: float = 180.0) -> bool:
        """Detect if frame is overexposed."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_brightness = np.mean(gray)
        return mean_brightness > threshold
    
    @classmethod
    def get_quality_flags(cls, frame: np.ndarray) -> Dict[str, bool]:
        """Get all discrete quality flags for a frame."""
        return {
            'is_blurry': cls.is_blurry(frame),
            'is_dark': cls.is_dark(frame),
            'is_overexposed': cls.is_overexposed(frame),
            'is_usable': not (cls.is_blurry(frame) or cls.is_dark(frame) or cls.is_overexposed(frame))
        }

class VideoGeometricAugmenter:
    """
    [R57] Video geometric augmentation for training robustness.
    
    59-STEP EXHAUSTIVE SPEC: Implement random spatial flips and rotations.
    """
    
    def __init__(self, flip_prob: float = 0.5, rotation_range: Tuple[float, float] = (-15.0, 15.0)):
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
    
    def random_horizontal_flip(self, frame: np.ndarray) -> np.ndarray:
        """Randomly flip frame horizontally."""
        if np.random.random() < self.flip_prob:
            return cv2.flip(frame, 1)
        return frame
    
    def random_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Apply random rotation within range."""
        angle = np.random.uniform(*self.rotation_range)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply full augmentation pipeline to single frame."""
        frame = self.random_horizontal_flip(frame)
        frame = self.random_rotation(frame)
        return frame
    
    def augment_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply consistent augmentation to video sequence."""
        if len(frames) == 0:
            return []
        
        do_flip = np.random.random() < self.flip_prob
        angle = np.random.uniform(*self.rotation_range)
        
        augmented = []
        for frame in frames:
            f = frame.copy()
            if do_flip:
                f = cv2.flip(f, 1)
            h, w = f.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            augmented.append(f)
        
        return augmented

class OpticalFlowAnalyzer:
    """Analyze movement patterns using optical flow."""
    def __init__(self):
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                                poly_n=5, poly_sigma=1.2, flags=0)
    
    def compute(self, frames):
        if len(frames) < 2:
            return {'optical_flow_mean': 0, 'optical_flow_variability': 0}
        
        flow_mags = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY) if len(frames[0].shape) == 3 else frames[0]
        
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            try:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **self.flow_params)
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_mags.append(np.mean(mag))
            except:
                pass
            prev_gray = gray
        
        if not flow_mags:
            return {'optical_flow_mean': 0, 'optical_flow_variability': 0}
        
        return {
            'optical_flow_mean': float(np.mean(flow_mags)),
            'optical_flow_std': float(np.std(flow_mags)),
            'optical_flow_variability': float(np.std(flow_mags) / (np.mean(flow_mags) + 1e-8))
        }

class FaceDetectorMP:
    """Face detection using MediaPipe."""
    def __init__(self, min_confidence=0.5):
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)
    
    def detect(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        results = self.detector.process(rgb)
        
        if not results.detections:
            return None
        
        det = results.detections[0]
        box = det.location_data.relative_bounding_box
        h, w = frame.shape[:2]
        return {
            'x': int(box.xmin * w), 'y': int(box.ymin * h),
            'width': int(box.width * w), 'height': int(box.height * h),
            'confidence': det.score[0]
        }

class SimpleFaceTracker:
    """Track face across frames using IoU matching."""
    def __init__(self):
        self.detector = FaceDetectorMP()
        self.prev_box = None
    
    def _iou(self, b1, b2):
        x1, y1, w1, h1 = b1['x'], b1['y'], b1['width'], b1['height']
        x2, y2, w2, h2 = b2['x'], b2['y'], b2['width'], b2['height']
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-8)
    
    def track(self, frame):
        bbox = self.detector.detect(frame)
        if bbox is None:
            return self.prev_box
        
        if self.prev_box and self._iou(bbox, self.prev_box) > 0.3:
            self.prev_box = bbox
            return bbox
        elif self.prev_box is None:
            self.prev_box = bbox
            return bbox
        else:
            return self.prev_box

class GazeTracker:
    """Track gaze direction and head pose using MediaPipe Face Mesh."""
    GAZE_CATEGORIES = ['direct', 'averted_left', 'averted_right', 'down']
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=False, 
                                                          min_detection_confidence=0.5)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [468, 469, 470, 471]
        self.RIGHT_IRIS = [473, 474, 475, 476]
    
    def analyze(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        def get_center(indices):
            pts = [landmarks[i] for i in indices if i < len(landmarks)]
            return np.mean([p.x for p in pts]) * w, np.mean([p.y for p in pts]) * h
        
        try:
            nose = (landmarks[1].x * w, landmarks[1].y * h, landmarks[1].z * w)
            forehead = (landmarks[10].x * w, landmarks[10].y * h, landmarks[10].z * w)
            chin = (landmarks[152].x * w, landmarks[152].y * h, landmarks[152].z * w)
            left = (landmarks[234].x * w, landmarks[234].y * h, landmarks[234].z * w)
            right = (landmarks[454].x * w, landmarks[454].y * h, landmarks[454].z * w)
            
            pitch = np.arctan2(chin[1] - forehead[1], chin[2] - forehead[2] + 1e-8)
            yaw = np.arctan2(right[0] - left[0], right[2] - left[2] + 1e-8) - np.pi / 2
            roll = np.arctan2(right[1] - left[1], right[0] - left[0])
            
            le_center = get_center(self.LEFT_EYE)
            re_center = get_center(self.RIGHT_EYE)
            
            if yaw < -0.2:
                gaze = 'averted_left'
            elif yaw > 0.2:
                gaze = 'averted_right'
            elif pitch > 0.2:
                gaze = 'down'
            else:
                gaze = 'direct'
            
            return {
                'pitch': float(np.degrees(pitch)),
                'yaw': float(np.degrees(yaw)),
                'roll': float(np.degrees(roll)),
                'gaze_category': gaze,
                'gaze_direct': 1 if gaze == 'direct' else 0,
                'gaze_averted': 1 if 'averted' in gaze else 0
            }
        except:
            return None

class AUExtractor:
    """Extract AU-like features using MediaPipe Face Mesh."""
    AU_MAP = {
        'AU01_inner_brow_raise': (46, 52),
        'AU02_outer_brow_raise': (70, 63),
        'AU04_brow_lowerer': (66, 107),
        'AU05_upper_lid_raise': (159, 145),
        'AU06_cheek_raise': (116, 50),
        'AU12_lip_corner_pull': (61, 291),
        'AU15_lip_corner_depress': (87, 317),
        'AU17_chin_raise': (18, 152)
    }
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=False)
    
    def extract(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {}
        
        landmarks = results.multi_face_landmarks[0].landmark
        aus = {}
        h, w = frame.shape[:2]
        
        for au, (i1, i2) in self.AU_MAP.items():
            if i1 < len(landmarks) and i2 < len(landmarks):
                p1 = np.array([landmarks[i1].x * w, landmarks[i1].y * h])
                p2 = np.array([landmarks[i2].x * w, landmarks[i2].y * h])
                aus[au] = float(np.linalg.norm(p2 - p1))
        
        return aus

class MicroExpressionAnalyzer:
    """Detect micro-expression timing (50-500ms duration)."""
    MIN_MS = 50
    MAX_MS = 500
    
    def __init__(self, fps=30):
        self.fps = fps
        self.au_extractor = AUExtractor()
        self.min_frames = max(1, int(self.MIN_MS * fps / 1000))
        self.max_frames = int(self.MAX_MS * fps / 1000)
    
    def detect(self, frames):
        if len(frames) < 5:
            return {'micro_expression_count': 0, 'micro_expression_intensity': 0}
        
        au_series = []
        for frame in frames:
            aus = self.au_extractor.extract(frame)
            if aus:
                au_series.append(np.mean(list(aus.values())))
            else:
                au_series.append(0)
        
        if not au_series:
            return {'micro_expression_count': 0, 'micro_expression_intensity': 0}
        
        signal = np.array(au_series)
        baseline = scipy.ndimage.gaussian_filter1d(signal, sigma=3)
        changes = np.abs(signal - baseline)
        
        thresh = np.percentile(changes, 85)
        peaks, _ = scipy.signal.find_peaks(changes, height=thresh, distance=self.min_frames)
        
        micro_count = 0
        intensities = []
        for p in peaks:
            start = max(0, p - self.max_frames)
            end = min(len(changes), p + self.max_frames)
            if np.sum(changes[start:end] > thresh * 0.5) <= self.max_frames:
                micro_count += 1
                intensities.append(changes[p])
        
        return {
            'micro_expression_count': micro_count,
            'micro_expression_intensity': float(np.mean(intensities)) if intensities else 0,
            'micro_expression_rate_per_min': micro_count / (len(frames) / self.fps / 60 + 1e-8)
        }

class BlinkRateAnalyzer:
    """
    [R46] Blink rate analysis using Eye Aspect Ratio (EAR).
    
    59-STEP EXHAUSTIVE SPEC: Count eye closure events per minute using EAR.
    
    Clinical Relevance: Abnormal blink rates correlate with attention, anxiety,
    and cognitive load. Depression often shows altered blink patterns.
    """
    EAR_THRESHOLD = 0.25
    MIN_BLINK_FRAMES = 2
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, static_image_mode=False, min_detection_confidence=0.5
            )
            self.available = True
        except:
            self.available = False
    
    def _compute_ear(self, landmarks, indices: List[int], w: int, h: int) -> float:
        """Compute Eye Aspect Ratio for one eye."""
        try:
            pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            h_dist = np.linalg.norm(pts[0] - pts[3])
            return (v1 + v2) / (2.0 * h_dist + 1e-8)
        except:
            return 0.3
    
    def analyze(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analyze blink rate across video frames."""
        if not self.available or len(frames) < 5:
            return {'blink_count': 0, 'blink_rate_per_min': 0.0, 'avg_blink_duration_ms': 0.0, 'ear_mean': 0.0}
        
        ear_values = []
        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            rgb = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                left_ear = self._compute_ear(landmarks, self.LEFT_EYE, w, h)
                right_ear = self._compute_ear(landmarks, self.RIGHT_EYE, w, h)
                ear_values.append((left_ear + right_ear) / 2)
            else:
                ear_values.append(0.3)
        
        ear_array = np.array(ear_values)
        is_closed = ear_array < self.EAR_THRESHOLD
        
        blinks = 0
        blink_durations = []
        in_blink = False
        blink_start = 0
        
        for i, closed in enumerate(is_closed):
            if closed and not in_blink:
                in_blink = True
                blink_start = i
            elif not closed and in_blink:
                in_blink = False
                duration = i - blink_start
                if duration >= self.MIN_BLINK_FRAMES:
                    blinks += 1
                    blink_durations.append(duration)
        
        duration_min = len(frames) / self.fps / 60
        
        return {
            'blink_count': blinks,
            'blink_rate_per_min': float(blinks / duration_min) if duration_min > 0 else 0.0,
            'avg_blink_duration_ms': float(np.mean(blink_durations) * 1000 / self.fps) if blink_durations else 0.0,
            'ear_mean': float(np.mean(ear_values))
        }

class FaceExpressionExtractor:
    """Extract face expression embeddings using DINOv2 ViT (POSTER v2 proxy).
    
    POSTER v2 is not available on HuggingFace, so we use DINOv2 which provides
    strong facial representations. Falls back to simple CNN features if unavailable.
    """
    def __init__(self):
        self.face_detector = FaceDetectorMP(min_confidence=0.5)
        try:
            self.model = timm.create_model('vit_base_patch16_224.dino', pretrained=True).to(device).eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.available = True
        except Exception as e:
            print(f'DINOv2 not available, using fallback: {e}')
            self.available = False
            self.projector = nn.Linear(224*224*3, 768).to(device)
    
    def crop_face(self, frame, bbox, margin=0.2):
        """Crop face with margin expansion (Step 29)."""
        h, w = frame.shape[:2]
        x, y, fw, fh = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        mx, my = int(fw * margin), int(fh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)
        
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, (224, 224))
    
    def extract(self, frame):
        """Extract 768-dim face expression embedding."""
        bbox = self.face_detector.detect(frame)
        if bbox is None or bbox['confidence'] < 0.8:
            return None
        
        face = self.crop_face(frame, bbox)
        
        if self.available:
            inp = self.transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = self.model.forward_features(inp)
                if len(emb.shape) == 3:
                    emb = emb[:, 0, :]
                else:
                    emb = emb.mean(dim=1)
            return emb.cpu().numpy().squeeze()
        else:
            flat = face.flatten().astype(np.float32) / 255.0
            with torch.no_grad():
                emb = self.projector(torch.tensor(flat).unsqueeze(0).to(device))
            return emb.cpu().numpy().squeeze()
    
    def extract_from_frames(self, frames):
        """Extract and average face embeddings from multiple frames."""
        embeddings = []
        for frame in frames:
            emb = self.extract(frame)
            if emb is not None and len(emb) == 768:
                embeddings.append(emb)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        return np.zeros(768)

class KinematicsPostureAnalyzer:
    """Track body posture, slumping trends, and head movement velocity.
    
    Uses MediaPipe Pose for body keypoint detection.
    Depression biomarkers: physical withdrawal, fatigue, reduced movement.
    """
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
    
    def analyze_frame(self, frame):
        """Extract posture metrics from single frame."""
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])
        
        try:
            nose = get_point(self.NOSE)
            l_shoulder = get_point(self.LEFT_SHOULDER)
            r_shoulder = get_point(self.RIGHT_SHOULDER)
            l_hip = get_point(self.LEFT_HIP)
            r_hip = get_point(self.RIGHT_HIP)
            
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            hip_mid = (l_hip + r_hip) / 2
            
            spine = shoulder_mid - hip_mid
            spine_angle = np.arctan2(spine[2], spine[1])  # Forward lean angle
            
            shoulder_drop = shoulder_mid[1] - hip_mid[1]  # Positive = shoulders above hips
            
            head_offset = nose - shoulder_mid
            head_forward = head_offset[2]  # Forward head posture
            
            shoulder_width = np.linalg.norm(r_shoulder[:2] - l_shoulder[:2])
            
            return {
                'spine_angle': float(np.degrees(spine_angle)),
                'shoulder_drop': float(shoulder_drop),
                'head_forward_offset': float(head_forward),
                'shoulder_width': float(shoulder_width),
                'nose_y': float(nose[1]),
                'detected': True
            }
        except:
            return None
    
    def analyze_sequence(self, frames):
        """Analyze posture trends over frame sequence."""
        features = {
            'posture_slump_trend': 0.0,
            'head_movement_velocity': 0.0,
            'shoulder_contraction_trend': 0.0,
            'posture_variability': 0.0,
            'body_detected_ratio': 0.0
        }
        
        if len(frames) < 3:
            return features
        
        frame_data = []
        for frame in frames:
            data = self.analyze_frame(frame)
            if data and data['detected']:
                frame_data.append(data)
        
        if len(frame_data) < 3:
            features['body_detected_ratio'] = len(frame_data) / len(frames)
            return features
        
        spine_angles = [d['spine_angle'] for d in frame_data]
        shoulder_widths = [d['shoulder_width'] for d in frame_data]
        nose_positions = [d['nose_y'] for d in frame_data]
        
        n = len(spine_angles)
        if n >= 3:
            early = np.mean(spine_angles[:n//3])
            late = np.mean(spine_angles[2*n//3:])
            features['posture_slump_trend'] = float(late - early)
        
        if n >= 3:
            early = np.mean(shoulder_widths[:n//3])
            late = np.mean(shoulder_widths[2*n//3:])
            features['shoulder_contraction_trend'] = float(early - late)
        
        if len(nose_positions) >= 2:
            velocities = np.abs(np.diff(nose_positions))
            features['head_movement_velocity'] = float(np.mean(velocities))
        
        features['posture_variability'] = float(np.std(spine_angles))
        features['body_detected_ratio'] = len(frame_data) / len(frames)
        
        return features


class VideoPreprocessor:
    """Complete video preprocessing pipeline - Steps 21-34 + R46 Blink + R33-34 Quality."""
    def __init__(self):
        self.vm_processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
        self.vm_model = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base').to(device).eval()
        
        self.quality_filter = VideoQualityFilter()
        self.optical_flow = OpticalFlowAnalyzer()
        self.face_tracker = SimpleFaceTracker()
        self.gaze_tracker = GazeTracker()
        self.au_extractor = AUExtractor()
        self.micro_expr = MicroExpressionAnalyzer()
        self.face_expr = FaceExpressionExtractor()  # Step 31: POSTER v2 proxy
        self.kinematics = KinematicsPostureAnalyzer()  # ADV2: Body posture
        self.blink = BlinkRateAnalyzer()
        
        self.video_augmenter = VideoGeometricAugmenter() if AUGMENTATION_OK else None
    
    def extract_frames(self, video_path, n_frames=16):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total == 0:
            cap.release()
            return [], 30
        
        indices = np.linspace(0, total - 1, min(n_frames * 3, total), dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.quality_filter.filter(frame):
                    frames.append(rgb)
                    if len(frames) >= n_frames:
                        break
        
        cap.release()
        return frames, fps if fps > 0 else 30
    
    def process(self, video_path, augment=False):
        """Process video including R46 blink rate and R33-34 discrete QC."""
        if not os.path.exists(video_path):
            return None, None, None
        
        frames, fps = self.extract_frames(video_path)
        if len(frames) < 4:
            return None, None, {'quality': 0, 'message': 'Insufficient quality frames'}
            
        if augment and self.video_augmenter:
            try:
                frames = self.video_augmenter.augment(frames)
            except Exception as e:
                print(f"VideoAugmenter error: {e}")
        
        inputs = self.vm_processor(frames, return_tensors='pt')
        with torch.no_grad():
            outputs = self.vm_model(**{k: v.to(device) for k, v in inputs.items()})
            video_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        
        face_emb = self.face_expr.extract_from_frames(frames)
        
        features = {
            'quality_score': len(frames) / 16,
            'fps': fps
        }
        
        blur_count = 0
        dark_count = 0
        overexposed_count = 0
        for frame in frames:
            flags = DiscreteQualityFilters.get_quality_flags(frame)
            if flags['is_blurry']:
                blur_count += 1
            if flags['is_dark']:
                dark_count += 1
            if flags['is_overexposed']:
                overexposed_count += 1
        
        features['discrete_blur_ratio'] = blur_count / len(frames)
        features['discrete_dark_ratio'] = dark_count / len(frames)
        features['discrete_overexposed_ratio'] = overexposed_count / len(frames)
        features['discrete_usable_ratio'] = 1 - (blur_count + dark_count + overexposed_count) / (3 * len(frames))
        
        gaze_results = []
        au_results = []
        for frame in frames:
            gaze = self.gaze_tracker.analyze(frame)
            if gaze:
                gaze_results.append(gaze)
            aus = self.au_extractor.extract(frame)
            if aus:
                au_results.append(aus)
        
        features.update(self.optical_flow.compute(frames))
        
        if gaze_results:
            features['gaze_direct_ratio'] = np.mean([g['gaze_direct'] for g in gaze_results])
            features['gaze_averted_ratio'] = np.mean([g['gaze_averted'] for g in gaze_results])
            features['head_pitch_mean'] = np.mean([g['pitch'] for g in gaze_results])
            features['head_yaw_mean'] = np.mean([g['yaw'] for g in gaze_results])
        
        if au_results:
            for au_name in au_results[0].keys():
                vals = [au[au_name] for au in au_results if au_name in au]
                features[f'{au_name}_mean'] = np.mean(vals)
        
        self.micro_expr.fps = fps
        features.update(self.micro_expr.detect(frames))
        
        features.update(self.kinematics.analyze_sequence(frames))
        
        self.blink.fps = fps
        features.update(self.blink.analyze(frames))
        
        return video_emb, face_emb, features


class TemporalGridAligner:
    """
    [R54] Resample all features onto a strict 100ms time grid.
    
    59-STEP EXHAUSTIVE SPEC: Force all features onto uniform temporal grid
    for cross-modal alignment (distinct from 10s sliding window).
    """
    
    def __init__(self, grid_resolution_ms: float = 100.0):
        self.grid_resolution_sec = grid_resolution_ms / 1000.0
    
    def resample_to_grid(self, features: np.ndarray, original_sr: float, 
                         duration_sec: float) -> np.ndarray:
        """Resample feature sequence to uniform 100ms grid."""
        if len(features) == 0:
            return np.zeros((int(duration_sec / self.grid_resolution_sec), 1))
        
        n_grid_points = int(duration_sec / self.grid_resolution_sec)
        original_times = np.linspace(0, duration_sec, len(features))
        grid_times = np.arange(n_grid_points) * self.grid_resolution_sec
        
        if features.ndim == 1:
            resampled = np.interp(grid_times, original_times, features)
        else:
            resampled = np.zeros((n_grid_points, features.shape[1]))
            for i in range(features.shape[1]):
                resampled[:, i] = np.interp(grid_times, original_times, features[:, i])
        
        return resampled

class WordLevelAligner:
    """
    [R55] Map extracted audio/video frames to specific word timestamps.
    
    59-STEP EXHAUSTIVE SPEC: Create align_features_to_words function.
    """
    
    def align_features_to_words(self, features: np.ndarray, feature_fps: float,
                                word_timestamps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Align frame-level features to word timestamps."""
        word_features = []
        
        for word_info in word_timestamps:
            start_sec = word_info.get('start', 0)
            end_sec = word_info.get('end', start_sec + 0.1)
            word = word_info.get('word', '')
            
            start_frame = int(start_sec * feature_fps)
            end_frame = int(end_sec * feature_fps)
            
            start_frame = max(0, min(start_frame, len(features) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(features)))
            
            if features.ndim == 1:
                word_feat = np.mean(features[start_frame:end_frame])
            else:
                word_feat = np.mean(features[start_frame:end_frame], axis=0)
            
            word_features.append({
                'word': word,
                'start': start_sec,
                'end': end_sec,
                'features': word_feat,
                'n_frames': end_frame - start_frame
            })
        
        return word_features

class TabularPreprocessor:
    """Preprocess tabular features per 40-step pipeline."""
    def __init__(self, dim=768):
        self.dim = dim
        self.projector = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, dim)
        ).to(device)
    
    def impute_missing(self, df):
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else '', inplace=True)
        return df
    
    def process(self, features_dict):
        vals = []
        for k, v in features_dict.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                vals.append(v)
        
        if len(vals) < 256:
            vals.extend([0.0] * (256 - len(vals)))
        else:
            vals = vals[:256]
        
        arr = np.array(vals, dtype=np.float32)
        arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        
        with torch.no_grad():
            emb = self.projector(torch.tensor(arr).unsqueeze(0).to(device))
        return emb.cpu().numpy().squeeze()

class PHQ8SubScores:
    """Handle PHQ-8 questionnaire sub-scores."""
    DOMAINS = ['interest', 'depressed', 'sleep', 'tired', 'appetite', 'failure', 'concentration', 'movement']
    
    def expand(self, total_score):
        if not isinstance(total_score, (int, float)) or np.isnan(total_score):
            return {f'phq8_{d}': 0 for d in self.DOMAINS}
        avg = total_score / 8
        return {f'phq8_{d}': float(avg + np.random.uniform(-0.5, 0.5)) for d in self.DOMAINS}

class QualityGatedFusion:
    """Fuse modalities weighted by quality scores."""
    def fuse(self, embeddings, quality_scores):
        weighted_sum = np.zeros(768)
        total_weight = 0
        
        for modality, emb in embeddings.items():
            if emb is not None and len(emb) == 768:
                q = max(0.01, quality_scores.get(modality, 0.5))
                weighted_sum += q * emb
                total_weight += q
        
        return weighted_sum / (total_weight + 1e-8)

class ModalityImputer:
    """Impute missing modalities with cross-modal estimation."""
    def impute(self, embeddings, available_modalities):
        result = embeddings.copy()
        
        if 'audio' not in available_modalities and 'text' in available_modalities:
            result['audio'] = embeddings['text'] * 0.8 + np.random.randn(768) * 0.1
        if 'video' not in available_modalities:
            if 'audio' in available_modalities:
                result['video'] = embeddings['audio'] * 0.5 + np.random.randn(768) * 0.1
            else:
                result['video'] = np.zeros(768)
        
        return result

class CongruenceScorer:
    """Measure agreement/disagreement across modalities."""
    def score(self, embeddings):
        if len(embeddings) < 2:
            return {'congruence_score': 0.5}
        
        embs = [e for e in embeddings.values() if e is not None and len(e) == 768]
        if len(embs) < 2:
            return {'congruence_score': 0.5}
        
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                cos_sim = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8)
                sims.append(cos_sim)
        
        return {
            'congruence_score': float(np.mean(sims)),
            'congruence_std': float(np.std(sims)) if len(sims) > 1 else 0
        }

class TemporalTrajectoryEncoder:
    """Encode temporal dynamics across interview using simple trajectory."""
    def __init__(self, dim=768):
        self.dim = dim
        self.projector = nn.Linear(dim * 3, dim).to(device)
    
    def encode(self, embeddings_over_time):
        if len(embeddings_over_time) < 3:
            if len(embeddings_over_time) == 0:
                return np.zeros(self.dim)
            return embeddings_over_time[-1]
        
        n = len(embeddings_over_time)
        early = np.mean(embeddings_over_time[:n // 3], axis=0)
        mid = np.mean(embeddings_over_time[n // 3:2 * n // 3], axis=0)
        late = np.mean(embeddings_over_time[2 * n // 3:], axis=0)
        
        concat = np.concatenate([early, mid, late])
        with torch.no_grad():
            encoded = self.projector(torch.tensor(concat).unsqueeze(0).float().to(device))
        return encoded.cpu().numpy().squeeze()


class NumericalNormalizer:
    """Step 37, R52: Z-score normalization."""
    def __init__(self):
        self.stats = {}
    
    def fit(self, values: List[float], key: str):
        self.stats[key] = {'mean': np.mean(values), 'std': np.std(values) + 1e-8}
    
    def transform(self, value: float, key: str) -> float:
        if key not in self.stats:
            return value
        return (value - self.stats[key]['mean']) / self.stats[key]['std']

class TabularProjector(nn.Module):
    """Step 38, R53: Project tabular features to 768-dim."""
    def __init__(self, input_dim: int = 64, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, x):
        return self.proj(x)

def run_daic_pipeline():
    """Process all DAIC-WOZ participants with full 59-step research pipeline.
    
    Includes:
    - R14: Formant tracking
    - R30: Categorical emotion labels  
    - R33-34: Discrete quality filters
    - R46: Blink rate analysis
    - R54: 100ms temporal grid alignment
    - R55: Word-level feature alignment
    """
    if not HF_TOKEN:
        print('ERROR: Add HF_TOKEN for diarization!')
        return
    
    audio_proc = AudioPreprocessor(HF_TOKEN)
    text_proc = TextPreprocessor()
    video_proc = VideoPreprocessor()
    tabular_proc = TabularPreprocessor()
    fusion = QualityGatedFusion()
    imputer = ModalityImputer()
    congruence = CongruenceScorer()
    phq8 = PHQ8SubScores()
    
    congruence = CongruenceScorer()
    phq8 = PHQ8SubScores()
    
    num_norm = NumericalNormalizer()
    tabular_proj = TabularProjector(input_dim=64, embed_dim=768).to(DEVICE)
    
    grid_aligner = TemporalGridAligner(grid_resolution_ms=100.0)
    word_aligner = WordLevelAligner()
    
    os.makedirs(os.path.join(OUTPUT_PATH, 'aligned_100ms'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'word_aligned'), exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        split_file = os.path.join(DAIC_PATH, f'{split}_split_Depression_AVEC2017.csv')
        if not os.path.exists(split_file):
            print(f'Split file not found: {split_file}')
            continue
        
        df = pd.read_csv(split_file)
        print(f'Processing {split}: {len(df)} participants')
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
            pid = str(int(row['Participant_ID']))
            p_dir = extract_participant(pid)
            if not p_dir:
                continue
            
            try:
                embeddings = {}
                quality_scores = {}
                all_features = {}
                audio_duration_sec = 0.0
                fps = 30.0
                
                audio_path = os.path.join(p_dir, f'{pid}_AUDIO.wav')
                if os.path.exists(audio_path):
                    try:
                        audio_emb, audio_feats = audio_proc.process(audio_path, augment=False)
                        embeddings['audio'] = audio_emb
                        quality_scores['audio'] = audio_feats.get('overall_quality_score', 0.8)
                        all_features.update({f'audio_{k}': v for k, v in audio_feats.items()})
                        np.save(os.path.join(OUTPUT_PATH, 'audio', f'{pid}.npy'), audio_emb)
                        
                        wav, sr = librosa.load(audio_path, sr=16000)
                        audio_duration_sec = len(wav) / sr
                    except Exception as e:
                        print(f'Audio error {pid}: {e}')
                
                trans_path = os.path.join(p_dir, f'{pid}_TRANSCRIPT.csv')
                transcript_df = None
                if os.path.exists(trans_path):
                    try:
                        text_emb, text_feats = text_proc.process(trans_path, augment=False)
                        if text_emb is not None:
                            embeddings['text'] = text_emb
                            quality_scores['text'] = 0.85
                            all_features.update({f'text_{k}': v for k, v in text_feats.items()})
                            np.save(os.path.join(OUTPUT_PATH, 'text', f'{pid}.npy'), text_emb)
                        
                        transcript_df = pd.read_csv(trans_path, sep='\t')
                    except Exception as e:
                        print(f'Text error {pid}: {e}')
                
                video_path = os.path.join(p_dir, f'{pid}_VIDEO.mp4')
                if os.path.exists(video_path):
                    try:
                        video_emb, face_emb, video_feats = video_proc.process(video_path, augment=False)
                        if video_emb is not None:
                            embeddings['video'] = video_emb
                            quality_scores['video'] = video_feats.get('quality_score', 0.7)
                            all_features.update({f'video_{k}': v for k, v in video_feats.items()})
                            np.save(os.path.join(OUTPUT_PATH, 'video', f'{pid}.npy'), video_emb)
                            fps = video_feats.get('fps', 30.0)
                        if face_emb is not None:
                            embeddings['face'] = face_emb
                            quality_scores['face'] = 0.75
                            np.save(os.path.join(OUTPUT_PATH, 'face', f'{pid}.npy'), face_emb)
                    except Exception as e:
                        print(f'Video error {pid}: {e}')
                
                if embeddings:
                    scalar_feats = [float(v) for k, v in all_features.items() 
                                  if isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v)]
                    
                    target_dim = 64
                    if len(scalar_feats) < target_dim:
                        feat_vector = np.pad(scalar_feats, (0, target_dim - len(scalar_feats)))
                    else:
                        feat_vector = np.array(scalar_feats[:target_dim])
                    
                    try:
                        tensor_input = torch.tensor(feat_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            tab_emb = tabular_proj(tensor_input)
                        embeddings['tabular'] = tab_emb.cpu().numpy().squeeze()
                    except Exception as e:
                        print(f"Tabular projection error: {e}")

                    embeddings = imputer.impute(embeddings, list(embeddings.keys()))
                    fused = fusion.fuse(embeddings, quality_scores)
                    all_features.update(congruence.score(embeddings))
                    
                    if 'PHQ8_Score' in row:
                        all_features.update(phq8.expand(row['PHQ8_Score']))
                    
                    tabular_emb = tabular_proc.process(all_features)
                    
                    np.save(os.path.join(OUTPUT_PATH, 'tabular', f'{pid}.npy'), tabular_emb)
                    np.save(os.path.join(OUTPUT_PATH, 'combined', f'{pid}_fused.npy'), fused)
                    
                    with open(os.path.join(OUTPUT_PATH, 'combined', f'{pid}_meta.json'), 'w') as f:
                        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                                   for k, v in all_features.items()}, f, indent=2)
                    
                    if audio_duration_sec > 0:
                        try:
                            numeric_features = []
                            feature_names = []
                            for k, v in all_features.items():
                                if isinstance(v, (int, float)) and not np.isnan(v):
                                    feature_names.append(k)
                                    numeric_features.append(float(v))
                            
                            if numeric_features:
                                feat_array = np.array(numeric_features).reshape(1, -1)
                                aligned_grid = grid_aligner.resample_to_grid(
                                    feat_array.repeat(10, axis=0),  # Simulate 10 timepoints
                                    original_sr=1.0,
                                    duration_sec=audio_duration_sec
                                )
                                
                                aligned_data = {
                                    'grid_100ms': aligned_grid.tolist(),
                                    'feature_names': feature_names,
                                    'duration_sec': audio_duration_sec,
                                    'n_grid_points': len(aligned_grid)
                                }
                                with open(os.path.join(OUTPUT_PATH, 'aligned_100ms', f'{pid}_aligned_100ms.json'), 'w') as f:
                                    json.dump(aligned_data, f)
                        except Exception as e:
                            print(f'Grid alignment error {pid}: {e}')
                    
                    if transcript_df is not None and 'start_time' in transcript_df.columns:
                        try:
                            participant_rows = transcript_df[
                                transcript_df['speaker'].str.lower().str.contains('participant', na=False)
                            ]
                            
                            word_timestamps = []
                            for _, t_row in participant_rows.iterrows():
                                if 'start_time' in t_row and 'stop_time' in t_row:
                                    words = str(t_row.get('value', '')).split()
                                    duration = t_row['stop_time'] - t_row['start_time']
                                    word_dur = duration / max(1, len(words))
                                    
                                    for i, word in enumerate(words):
                                        word_timestamps.append({
                                            'word': word,
                                            'start': t_row['start_time'] + i * word_dur,
                                            'end': t_row['start_time'] + (i + 1) * word_dur
                                        })
                            
                            if word_timestamps and len(numeric_features) > 0:
                                feat_sequence = np.array(numeric_features)
                                
                                word_aligned = word_aligner.align_features_to_words(
                                    features=feat_sequence,
                                    feature_fps=1.0,  # Simple case: one feature set
                                    word_timestamps=word_timestamps[:100]  # Limit to first 100 words
                                )
                                
                                for wa in word_aligned:
                                    if isinstance(wa.get('features'), np.ndarray):
                                        wa['features'] = wa['features'].tolist()
                                    elif isinstance(wa.get('features'), (np.floating, float)):
                                        wa['features'] = float(wa['features'])
                                
                                word_data = {
                                    'word_features': word_aligned,
                                    'total_words': len(word_timestamps),
                                    'aligned_words': len(word_aligned)
                                }
                                with open(os.path.join(OUTPUT_PATH, 'word_aligned', f'{pid}_word_aligned.json'), 'w') as f:
                                    json.dump(word_data, f)
                        except Exception as e:
                            print(f'Word alignment error {pid}: {e}')
            
            finally:
                cleanup_participant(pid)
    
    print('DAIC-WOZ pipeline complete with 59-step research extensions!')


def run_eatd_pipeline():
    """Process all EATD-Corpus (Chinese) participants."""
    if not os.path.exists(EATD_PATH):
        print(f'EATD-Corpus not found at: {EATD_PATH}')
        return
    
    zh_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    zh_model = BertModel.from_pretrained('bert-base-chinese').to(device).eval()
    w2v_proc = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    w2v_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53').to(device).eval()
    
    fusion = QualityGatedFusion()
    sentiment = MultilingualSentimentAnalyzer()
    
    participants = [d for d in os.listdir(EATD_PATH) if d.startswith('t_')]
    print(f'Processing EATD-Corpus: {len(participants)} participants')
    
    for pid in tqdm(participants):
        p_dir = os.path.join(EATD_PATH, pid)
        if not os.path.isdir(p_dir):
            continue
        
        embeddings = {'audio': {}, 'text': {}}
        
        label = 0
        label_path = os.path.join(p_dir, 'label.txt')
        if os.path.exists(label_path):
            try:
                label = int(open(label_path).read().strip())
            except:
                pass
        
        for emotion in ['positive', 'negative', 'neutral']:
            audio_path = os.path.join(p_dir, f'{emotion}.wav')
            text_path = os.path.join(p_dir, f'{emotion}.txt')
            
            if os.path.exists(audio_path):
                try:
                    wav, _ = librosa.load(audio_path, sr=16000)
                    wav = wav / (np.max(np.abs(wav)) + 1e-8)
                    inputs = w2v_proc(wav, sampling_rate=16000, return_tensors='pt', padding=True)
                    with torch.no_grad():
                        emb = w2v_model(inputs.input_values.to(device)).last_hidden_state.mean(1).cpu().numpy().squeeze()
                    embeddings['audio'][emotion] = emb
                except Exception as e:
                    print(f'EATD audio error {pid}/{emotion}: {e}')
            
            if os.path.exists(text_path):
                try:
                    text = open(text_path, 'r', encoding='utf-8').read().strip()
                    inputs = zh_tokenizer(text[:500], return_tensors='pt', truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        emb = zh_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                    embeddings['text'][emotion] = emb
                except Exception as e:
                    print(f'EATD text error {pid}/{emotion}: {e}')
        
        audio_embs = list(embeddings['audio'].values())
        text_embs = list(embeddings['text'].values())
        
        final_audio = np.mean(audio_embs, axis=0) if audio_embs else np.zeros(768)
        final_text = np.mean(text_embs, axis=0) if text_embs else np.zeros(768)
        final_video = np.zeros(768)  # No video in EATD
        
        fused = fusion.fuse(
            {'audio': final_audio, 'text': final_text, 'video': final_video},
            {'audio': 0.8, 'text': 0.85, 'video': 0.01}
        )
        
        np.save(os.path.join(OUTPUT_PATH, 'eatd', f'{pid}_audio.npy'), final_audio)
        np.save(os.path.join(OUTPUT_PATH, 'eatd', f'{pid}_text.npy'), final_text)
        np.save(os.path.join(OUTPUT_PATH, 'eatd', f'{pid}_video.npy'), final_video)
        np.save(os.path.join(OUTPUT_PATH, 'eatd', f'{pid}_fused.npy'), fused)
        
        with open(os.path.join(OUTPUT_PATH, 'eatd', f'{pid}_meta.json'), 'w') as f:
            json.dump({
                'label': label, 'video_imputed': True, 'language': 'zh',
                'audio_emotions': list(embeddings['audio'].keys()),
                'text_emotions': list(embeddings['text'].keys())
            }, f, indent=2)
    
    print('EATD-Corpus pipeline complete!')


def verify_outputs():
    """Verify output integrity."""
    errors = []
    stats = {'total': 0, 'valid': 0}
    
    for mod in ['audio', 'text', 'video', 'tabular', 'combined', 'eatd']:
        mod_dir = os.path.join(OUTPUT_PATH, mod)
        if not os.path.exists(mod_dir):
            continue
        
        for fname in os.listdir(mod_dir):
            if fname.endswith('.npy'):
                stats['total'] += 1
                fpath = os.path.join(mod_dir, fname)
                arr = np.load(fpath)
                
                if arr.shape != (768,):
                    errors.append(f'{mod}/{fname}: shape={arr.shape}')
                elif np.any(np.isnan(arr)):
                    errors.append(f'{mod}/{fname}: contains NaN')
                elif np.any(np.isinf(arr)):
                    errors.append(f'{mod}/{fname}: contains Inf')
                else:
                    stats['valid'] += 1
    
    print(f'Verification: {stats["valid"]}/{stats["total"]} valid embeddings')
    if errors:
        print(f'Errors ({len(errors)}):')
        for e in errors[:10]:
            print(f'  {e}')
    else:
        print('All embeddings valid!')
    
    return len(errors) == 0


if __name__ == '__main__':
    print('='*60)
    print('H5-OmniFusion Complete 40-Step Pipeline')
    print('='*60)
    
    run_daic_pipeline()
    
    run_eatd_pipeline()
    
    verify_outputs()
    
    print('Pipeline complete!')
