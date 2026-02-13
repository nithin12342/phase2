"""
H5-OmniFusion Complete Preprocessing Pipeline
Version: 3.0 | Target: F1 > 0.80, AUC > 0.85, MAE < 3.0
Complete 40-Step Production + 59-Step Research + 9 ADV Innovations
"""


import os, sys, re, gc, zipfile, shutil, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import h5py
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, CUDA: {torch.cuda.is_available()}")

try:
    import librosa
    import soundfile as sf
    LIBROSA_OK = True
except: LIBROSA_OK = False

try:
    import opensmile
    OPENSMILE_OK = True
except: OPENSMILE_OK = False

try:
    import parselmouth
    from parselmouth.praat import call
    PRAAT_OK = True
except: PRAAT_OK = False

try:
    import noisereduce as nr
    NOISEREDUCE_OK = True
except: NOISEREDUCE_OK = False

try:
    import cv2
    CV2_OK = True
except: CV2_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except: MEDIAPIPE_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except: VADER_OK = False

try:
    from snownlp import SnowNLP
    SNOWNLP_OK = True
except: SNOWNLP_OK = False

try:
    import timm
    TIMM_OK = True
except: TIMM_OK = False

try:
    from transformers import (Wav2Vec2Model, Wav2Vec2FeatureExtractor,
                              VideoMAEModel, VideoMAEImageProcessor,
                              AutoModel, AutoTokenizer)
    TRANSFORMERS_OK = True
except: TRANSFORMERS_OK = False

import scipy.signal
from PIL import Image
from torchvision import transforms

print(f"Libraries: librosa={LIBROSA_OK}, opensmile={OPENSMILE_OK}, praat={PRAAT_OK}")
print(f"Libraries: cv2={CV2_OK}, mediapipe={MEDIAPIPE_OK}, transformers={TRANSFORMERS_OK}")


try:
    from audio_enhancements import ProsodicFingerprint
    ADV3_OK = True
except ImportError:
    ADV3_OK = False
    print("Warning: ProsodicFingerprint (ADV3) not found in audio_enhancements")

try:
    from research_layer_extensions import FormantTrackExtractor, SpecAugment
    R14_R56_OK = True
except ImportError:
    R14_R56_OK = False
    print("Warning: FormantTrackExtractor (R14) / SpecAugment (R56) not found")

@dataclass
class Config:
    DAIC_WOZ_PATH: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets'
    EXTENDED_DAIC_WOZ_PATH: str = '/content/drive/MyDrive/Extended_DAIC-WOZ'  # Extended DAIC-WOZ
    EATD_CORPUS_PATH: str = '/content/drive/MyDrive/EATD-Corpus'
    PRETRAINED_PATH: str = '/content/drive/MyDrive/pretrained_models'
    OUTPUT_PATH: str = '/content/drive/MyDrive/h5_features'
    TEMP_PATH: str = '/content/temp_extract'
    
    SAMPLE_RATE: int = 16000
    WINDOW_SEC: float = 10.0
    OVERLAP: float = 0.5
    TARGET_LUFS: float = -23.0
    TARGET_FPS: int = 5
    NUM_FRAMES: int = 16
    FRAME_SIZE: Tuple[int, int] = (224, 224)
    BLUR_THRESHOLD: float = 50.0
    BRIGHTNESS_MIN: int = 80
    BRIGHTNESS_MAX: int = 180
    EMBED_DIM: int = 768

CFG = Config()
os.makedirs(CFG.TEMP_PATH, exist_ok=True)
os.makedirs(CFG.OUTPUT_PATH, exist_ok=True)

class ModelLoader:
    def __init__(self, device=DEVICE):
        self.device = device
        self.models = {}
        self.processors = {}
    
    def load_wav2vec2(self):
        path = f"{CFG.PRETRAINED_PATH}/audio/wav2vec2-large-xlsr-53"
        try:
            if os.path.exists(path):
                self.models['wav2vec2'] = Wav2Vec2Model.from_pretrained(path).to(self.device).eval()
                self.processors['wav2vec2'] = Wav2Vec2FeatureExtractor.from_pretrained(path)
                print(f"Loaded Wav2Vec2 from {path}")
                return True
            else:
                print(f"❌ LOCAL path not found: {path} (Strict Mode)")
        except Exception as e:
            print(f"Wav2Vec2 failed: {e}")
        return False
    
    def load_text_encoder(self, lang='english'):
        if lang == 'chinese':
            local = f"{CFG.PRETRAINED_PATH}/text/chinese-roberta-wwm-ext"
        else:
            local = f"{CFG.PRETRAINED_PATH}/text/mental-roberta-base"
        
        try:
            if os.path.exists(local):
                self.models[f'text_{lang}'] = AutoModel.from_pretrained(local).to(self.device).eval()
                self.processors[f'text_{lang}'] = AutoTokenizer.from_pretrained(local)
                print(f"Loaded {lang} text encoder from {local}")
                return True
            else:
                print(f"❌ LOCAL path not found: {local} (Strict Mode)")
        except Exception as e:
            print(f"{lang} encoder failed: {e}")
        return False
    
    def load_video_encoder(self):
        path = f"{CFG.PRETRAINED_PATH}/video/videomae-base"
        try:
            if os.path.exists(path):
                self.models['video'] = VideoMAEModel.from_pretrained(path).to(self.device).eval()
                self.processors['video'] = VideoMAEImageProcessor.from_pretrained(path)
                print(f"Loaded VideoMAE from {path}")
                return True
            else:
                 print(f"❌ LOCAL path not found: {path} (Strict Mode)")
        except Exception as e:
            print(f"VideoMAE failed: {e}")
            if TIMM_OK:
                print("⚠️ Trying ViT fallback (might download if not cached)...")
                try:
                    self.models['video'] = timm.create_model('vit_base_patch16_224', pretrained=True).to(self.device).eval()
                    return True
                except: pass
        return False
    
    def load_face_encoder(self):
        path = f"{CFG.PRETRAINED_PATH}/face/dinov2-base"
        try:
             if os.path.exists(path):
                 self.models['face'] = AutoModel.from_pretrained(path).to(self.device).eval()
                 print(f"Loaded Face Encoder from {path}")
                 return True
        except: pass

        if TIMM_OK:
            try:
                print("⚠️ Trying ViT face fallback (might download if not cached)...")
                self.models['face'] = timm.create_model('vit_base_patch16_224', pretrained=True).to(self.device).eval()
                return True
            except: pass
        return False

class TranscriptDiarizer:
    """R4: Speaker diarization using transcripts."""
    def __init__(self, sr=16000):
        self.sr = sr
    
    def parse_transcript(self, path: str) -> Optional[pd.DataFrame]:
        if not path or not os.path.exists(path): return None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(path, sep='\t', header=None, names=['start','end','speaker','text'], encoding=enc)
            except: continue
        return None
    
    def get_participant_segments(self, df: pd.DataFrame) -> List[Tuple[float,float]]:
        segs = []
        for _, r in df.iterrows():
            if 'ellie' not in str(r.get('speaker','')).lower():
                try: segs.append((float(r['start']), float(r['end'])))
                except: pass
        return segs
    
    def extract_audio(self, wav: np.ndarray, segs: List[Tuple[float,float]]) -> np.ndarray:
        if not segs: return wav
        parts = [wav[int(s*self.sr):int(e*self.sr)] for s,e in segs if int(e*self.sr)<=len(wav)]
        return np.concatenate(parts) if parts else wav
    
    def compute_latencies(self, df: pd.DataFrame) -> Dict:
        """ADV1: Response latency extraction."""
        if df is None: return {'latency_mean':0,'latency_std':0,'latency_max':0}
        lats, prev_end = [], None
        for _, r in df.iterrows():
            spk = str(r.get('speaker','')).lower()
            try:
                if 'ellie' in spk: prev_end = float(r['end'])
                elif prev_end is not None:
                    lat = (float(r['start']) - prev_end) * 1000
                    if 0 < lat < 10000: lats.append(lat)
                    prev_end = None
            except: pass
        return {'latency_mean':np.mean(lats) if lats else 0, 'latency_std':np.std(lats) if len(lats)>1 else 0, 'latency_max':max(lats) if lats else 0}

class LoudnessNormalizer:
    """Step 5, R6: LUFS normalization."""
    def __init__(self, target=-23.0, sr=16000):
        self.target, self.sr = target, sr
    def normalize(self, wav: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(wav**2)) + 1e-8
        return wav * (10**(self.target/20) / rms)

class PeakNormalizer:
    """Step 4, R5: Peak normalization."""
    @staticmethod
    def normalize(wav: np.ndarray) -> np.ndarray:
        return wav / (np.max(np.abs(wav)) + 1e-8)

class NoiseReducer:
    """Step 6, R7: Noise reduction."""
    def __init__(self, sr=16000, prop=0.8):
        self.sr, self.prop = sr, prop
    def reduce(self, wav: np.ndarray) -> np.ndarray:
        if NOISEREDUCE_OK:
            try: return nr.reduce_noise(y=wav, sr=self.sr, prop_decrease=self.prop)
            except: pass
        return wav

class VoiceActivityDetector:
    """Step 7, R8: VAD."""
    def __init__(self, sr=16000, top_db=30):
        self.sr, self.top_db = sr, top_db
    def detect(self, wav: np.ndarray) -> List[Tuple[int,int]]:
        if LIBROSA_OK:
            return [(int(s),int(e)) for s,e in librosa.effects.split(wav, top_db=self.top_db)]
        return [(0, len(wav))]

class AudioSegmenter:
    """Step 8, R9: Segmentation."""
    def __init__(self, sr=16000, win=10.0, overlap=0.5):
        self.win, self.hop = int(win*sr), int(win*sr*(1-overlap))
    def segment(self, wav: np.ndarray) -> List[np.ndarray]:
        return [wav[i:i+self.win] for i in range(0, max(1, len(wav)-self.win+1), self.hop)] or [wav]

class EGeMAPSExtractor:
    """Step 10, R11: eGeMAPS features."""
    def __init__(self):
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals) if OPENSMILE_OK else None
    def extract(self, path: str) -> np.ndarray:
        if self.smile and os.path.exists(path):
            try: return self.smile.process_file(path).values.flatten()
            except: pass
        return np.zeros(88)

class PauseAnalyzer:
    """Step 11, R16: Pause analysis."""
    def __init__(self, sr=16000):
        self.sr = sr
    def analyze(self, wav: np.ndarray) -> Dict:
        if not LIBROSA_OK: return {'pause_count':0,'pause_mean':0,'pause_ratio':0}
        ints = librosa.effects.split(wav, top_db=30)
        pauses = [(ints[i][0]-ints[i-1][1])/self.sr*1000 for i in range(1,len(ints)) if ints[i][0]-ints[i-1][1]>self.sr*0.2]
        total = len(wav)/self.sr
        return {'pause_count':len(pauses), 'pause_mean':np.mean(pauses) if pauses else 0, 'pause_ratio':sum(pauses)/1000/total if total>0 else 0}

class SpeakingRateAnalyzer:
    """R17: Speaking rate."""
    def __init__(self, sr=16000):
        self.sr = sr
    def analyze(self, wav: np.ndarray) -> Dict:
        if not LIBROSA_OK: return {'speaking_rate':0,'phonation_ratio':0}
        rms = librosa.feature.rms(y=wav)[0]
        peaks = len(scipy.signal.find_peaks(rms, distance=5)[0])
        dur = len(wav)/self.sr
        ints = librosa.effects.split(wav, top_db=30)
        speech = sum(e-s for s,e in ints)/self.sr
        return {'speaking_rate':peaks/max(dur,0.1), 'phonation_ratio':speech/max(dur,0.1)}

class SighDetector:
    """Step 11, ADV5: Sigh detection."""
    def __init__(self, sr=16000):
        self.sr = sr
    def detect(self, wav: np.ndarray) -> Dict:
        if not LIBROSA_OK: return {'sigh_count':0}
        ints = librosa.effects.split(wav, top_db=25)
        sighs = 0
        for s,e in ints:
            dur = (e-s)/self.sr
            if 1.0 <= dur <= 3.0:
                seg = wav[s:e]
                spec = np.abs(np.fft.rfft(seg))
                freqs = np.fft.rfftfreq(len(seg), 1/self.sr)
                if np.sum(spec[freqs<500])/(np.sum(spec)+1e-8) > 0.6: sighs += 1
        return {'sigh_count':sighs}

class BreathIntervalAnalyzer:
    """R15, ADV5: Breath analysis."""
    def __init__(self, sr=16000):
        self.sr = sr
    def analyze(self, wav: np.ndarray) -> Dict:
        if not LIBROSA_OK: return {'breath_groups':0,'breath_interval_std':0}
        ints = librosa.effects.split(wav, top_db=25)
        intervals = [(ints[i][0]-ints[i-1][1])/self.sr for i in range(1,len(ints)) if 0.3<(ints[i][0]-ints[i-1][1])/self.sr<2.0]
        return {'breath_groups':len(ints), 'breath_interval_std':np.std(intervals) if len(intervals)>1 else 0}

class FormantExtractor:
    """R14: Formants F1-F4."""
    def __init__(self, sr=16000):
        self.sr = sr
    def extract(self, wav: np.ndarray) -> Dict:
        if not PRAAT_OK: return {f'f{i}_mean':0 for i in range(1,5)}
        try:
            snd = parselmouth.Sound(wav, self.sr)
            fmt = call(snd, 'To Formant (burg)', 0.0, 5, 5500, 0.025, 50)
            return {f'f{i}_mean':call(fmt,'Get mean',i,0,0,'hertz') for i in range(1,5)}
        except: return {f'f{i}_mean':0 for i in range(1,5)}

class PitchAnalyzer:
    """R12: F0 pitch."""
    def __init__(self, sr=16000):
        self.sr = sr
    def analyze(self, wav: np.ndarray) -> Dict:
        if not PRAAT_OK: return {'f0_mean':0,'f0_std':0,'f0_range':0}
        try:
            snd = parselmouth.Sound(wav, self.sr)
            pitch = call(snd, 'To Pitch', 0.0, 75, 500)
            f0 = pitch.selected_array['frequency']
            f0 = f0[f0>0]
            return {'f0_mean':np.mean(f0),'f0_std':np.std(f0),'f0_range':np.max(f0)-np.min(f0)} if len(f0)>0 else {'f0_mean':0,'f0_std':0,'f0_range':0}
        except: return {'f0_mean':0,'f0_std':0,'f0_range':0}

class JitterShimmerAnalyzer:
    """R13: Voice quality."""
    def __init__(self, sr=16000):
        self.sr = sr
    def analyze(self, wav: np.ndarray) -> Dict:
        if not PRAAT_OK: return {'jitter':0,'shimmer':0}
        try:
            snd = parselmouth.Sound(wav, self.sr)
            pp = call(snd, 'To PointProcess (periodic, cc)', 75, 500)
            jit = call(pp, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
            shim = call([snd,pp], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
            return {'jitter':jit if not np.isnan(jit) else 0, 'shimmer':shim if not np.isnan(shim) else 0}
        except: return {'jitter':0,'shimmer':0}

class AudioQualityChecker:
    """Step 40, R59: Quality scoring."""
    def __init__(self, sr=16000):
        self.sr = sr
    def check(self, wav: np.ndarray) -> Dict:
        rms = np.sqrt(np.mean(wav**2)) + 1e-8
        noise_est = np.percentile(np.abs(wav), 10) + 1e-8
        snr = 20*np.log10(rms/noise_est)
        clip = np.mean(np.abs(wav)>0.99)
        va = np.mean(np.abs(wav)>0.01)
        score = min(1.0, max(0.0, (snr-5)/25)) * (1-clip) * min(1.0, va/0.4)
        return {'snr':snr, 'clipping':clip, 'voice_activity':va, 'quality_score':score}

class AudioPreprocessor:
    """Complete audio pipeline: Steps 1-11, R1-R17, ADV1,3,5."""
    def __init__(self, models: ModelLoader):
        self.models = models
        self.sr = CFG.SAMPLE_RATE
        self.diarizer = TranscriptDiarizer(self.sr)
        self.loudness = LoudnessNormalizer(CFG.TARGET_LUFS, self.sr)
        self.noise = NoiseReducer(self.sr)
        self.vad = VoiceActivityDetector(self.sr)
        self.segmenter = AudioSegmenter(self.sr)
        self.egemaps = EGeMAPSExtractor()
        self.pause = PauseAnalyzer(self.sr)
        self.rate = SpeakingRateAnalyzer(self.sr)
        self.sigh = SighDetector(self.sr)
        self.breath = BreathIntervalAnalyzer(self.sr)
        self.formant = FormantExtractor(self.sr)
        self.pitch = PitchAnalyzer(self.sr)
        self.jitshim = JitterShimmerAnalyzer(self.sr)
        self.quality = AudioQualityChecker(self.sr)
        
        self.audio_adapter = nn.Linear(1024, 768).to(DEVICE)
        self.audio_adapter.weight.data.normal_(0, 0.01)  # Initialize safely
        
        self.proj = nn.Linear(88, CFG.EMBED_DIM).to(DEVICE)
        
        if ADV3_OK:
            self.prosody = ProsodicFingerprint(self.sr)
        else:
            self.prosody = None
        
        if R14_R56_OK:
            self.formant_tracker = FormantTrackExtractor(sr=self.sr)
            self.spec_augment = SpecAugment()
        else:
            self.formant_tracker = None
            self.spec_augment = None
            
        try:
            self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                    model='silero_vad',
                                                    force_reload=False,
                                                    onnx=False)
            (self.get_speech_timestamps, _, _, _, _) = utils
            self.silero_model.to(DEVICE)
            print("Loaded Silero VAD (Non-Proximal Restoration)")
            self.SILERO_OK = True
        except Exception as e:
            print(f"Silero VAD failed: {e}. Falling back to Energy VAD.")
            self.SILERO_OK = False

    def load_audio(self, path: str) -> np.ndarray:
        if LIBROSA_OK:
            wav, _ = librosa.load(path, sr=self.sr, mono=True)
            return wav
        return np.zeros(self.sr*10)
    
    @torch.no_grad()
    def get_wav2vec2_embedding(self, wav: np.ndarray) -> np.ndarray:

        """Step 9: Extract Wav2Vec2 embedding. Projects 1024-dim to 768-dim if needed."""
        if 'wav2vec2' not in self.models.models: return np.zeros(CFG.EMBED_DIM)
        try:
            proc = self.models.processors['wav2vec2']
            model = self.models.models['wav2vec2']
            inputs = proc(wav, sampling_rate=self.sr, return_tensors='pt', padding=True)
            
            model_dtype = next(model.parameters()).dtype
            inputs = {k: v.to(device=DEVICE, dtype=model_dtype) if v.dtype.is_floating_point else v.to(DEVICE) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)  # Shape: [1, 1024] or [1, 768]
            
            if emb.shape[-1] == 1024:
                emb = self.audio_adapter(emb.float())  # Ensure float for linear layer
            
            return emb.float().cpu().numpy().flatten()  # Now guaranteed 768-dim
        except Exception as e:
            print(f"Wav2Vec2 Error: {e}")
            return np.zeros(CFG.EMBED_DIM)

    
    def process_audio(self, audio_path: str, transcript_path: str = None, augment: bool = False) -> Dict:
        result = {'audio_embedding': np.zeros(CFG.EMBED_DIM), 'quality_score': 0.0}
        wav = self.load_audio(audio_path)
        if len(wav) < self.sr: return result
        
        raw_max = np.max(np.abs(wav)) if len(wav) > 0 else 0
        peak_db_before = 20 * np.log10(raw_max + 1e-9)
        noise_floor = np.percentile(np.abs(wav), 10) if len(wav) > 0 else 0
        noise_floor_db = 20 * np.log10(noise_floor + 1e-9)
        
        try:
            if hasattr(self, 'SILERO_OK') and self.SILERO_OK:
                wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                vad_segs_dict = self.get_speech_timestamps(wav_t, self.silero_model, sampling_rate=self.sr)
                speech_dur = sum((d['end'] - d['start']) for d in vad_segs_dict)
                vad_ratio = speech_dur / len(wav) if len(wav) > 0 else 0
                vad_segs = [(d['start'], d['end']) for d in vad_segs_dict] # For usage below
            else:
                vad_segs = self.vad.detect(wav)
                speech_dur = sum(e-s for s,e in vad_segs)
                vad_ratio = speech_dur / len(wav) if len(wav) > 0 else 0
        except:
            vad_ratio = 0.0
            vad_segs = []

        if augment and self.spec_augment:
            try:
                wav = self.spec_augment.augment(wav, self.sr)
                result['augmentation_applied'] = True
            except Exception as e:
                print(f"SpecAugment error: {e}")

        
        if transcript_path and os.path.exists(transcript_path):
            df = self.diarizer.parse_transcript(transcript_path)
            if df is not None:
                segs = self.diarizer.get_participant_segments(df)
                wav = self.diarizer.extract_audio(wav, segs)
                result.update(self.diarizer.compute_latencies(df))
        else:
            try:
                if hasattr(self, 'SILERO_OK') and self.SILERO_OK and vad_segs:
                     speech_parts = [wav[int(s):int(e)] for s,e in vad_segs] # Silero returns INT samples usually, but let's Ensure
                     if speech_parts:
                         wav = np.concatenate(speech_parts)
                     result['diarization_fallback'] = True
                else:
                    voiced_regions = self.vad.detect(wav)
                    if voiced_regions:
                        wav = self.vad.extract_voiced(wav)
                    result['diarization_fallback'] = True
            except:
                result['diarization_fallback'] = True
        
        if len(wav) < self.sr: return result

        wav = PeakNormalizer.normalize(wav)
        peak_db_after = 20 * np.log10(np.max(np.abs(wav)) + 1e-9)

        wav = self.loudness.normalize(wav)
        wav = self.noise.reduce(wav)
        
        qc = self.quality.check(wav)
        result['quality_score'] = qc['quality_score']
        result.update(qc)
        result['audio_embedding'] = self.get_wav2vec2_embedding(wav)
        
        egemaps = self.egemaps.extract(audio_path)
        with torch.no_grad():
            result['egemaps_features'] = egemaps
            result['egemaps_embedding'] = self.proj(torch.tensor(egemaps, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        
        result.update(self.pause.analyze(wav))
        result.update(self.rate.analyze(wav))
        result.update(self.sigh.detect(wav))
        result.update(self.breath.analyze(wav))
        result.update(self.formant.extract(wav))
        result.update(self.pitch.analyze(wav))
        result.update(self.jitshim.analyze(wav))
        
        
        if self.prosody:
            result.update(self.prosody.extract_dict(wav))
        
        if self.formant_tracker:
            result.update(self.formant_tracker.extract(audio_path))
            
        prosodic_features = {}
        prosodic_features.update(result.get('pause_features', {}))
        for key in ['pause_count', 'pause_mean', 'pause_ratio', 'speaking_rate', 'phonation_ratio', 'sigh_count', 'breath_groups', 'breath_interval_std', 'f0_mean', 'f0_std', 'f0_range', 'jitter', 'shimmer']:
            if key in result: prosodic_features[key] = result[key]
        for i in range(1, 5):
            if f'f{i}_mean' in result: prosodic_features[f'f{i}_mean'] = result[f'f{i}_mean']
        result['prosodic_features'] = prosodic_features

        result['metadata'] = {
            'audio_sr_detected': self.sr,
            'peak_db_before': float(peak_db_before),
            'peak_db_after': float(peak_db_after),
            'noise_floor_db': float(noise_floor_db),
            'vad_speech_ratio': float(vad_ratio)
        }
            
        return result

print("Audio Pipeline loaded: 15 classes + AudioPreprocessor")
