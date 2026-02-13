
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
import tarfile
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DRIVE_PATH = '/content/drive/MyDrive'
DATA_PATH = f'{DRIVE_PATH}/DAIC-WOZ_Datasets'
OUTPUT_PATH = f'{DATA_PATH}/Features_SOTA_2025'
TEMP_PATH = '/content/temp_extract'
FEATURE_DIM = 768
FORCE_REEXTRACT = True  # Set True to re-extract all features

for subdir in ['audio', 'text', 'video', 'image', 'tabular', 'combined']:
    os.makedirs(f'{OUTPUT_PATH}/{subdir}', exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)

print(f"✓ Data path: {DATA_PATH}")
print(f"✓ Output path: {OUTPUT_PATH}")
print(f"✓ Force re-extract: {FORCE_REEXTRACT}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def get_participant_ids():
    """Get all participant IDs from both datasets."""
    pids = set()
    
    ext_data = Path(DATA_PATH) / 'Extended-DAIC-WOZ' / 'data'
    if ext_data.exists():
        for f in ext_data.glob('*_P.tar.gz'):
            try:
                pid = int(f.stem.split('_')[0])
                pids.add(pid)
            except:
                pass
    
    daic_folder = Path(DATA_PATH) / 'DAIC-WOZ'
    if daic_folder.exists():
        for f in daic_folder.glob('*_P.zip'):
            try:
                pid = int(f.stem.split('_')[0])
                pids.add(pid)
            except:
                pass
    
    return sorted(list(pids))

def load_models():
    """Load all SOTA models."""
    global wav2vec_processor, wav2vec_model, wav2vec_projection
    global mental_tokenizer, mental_model
    global videomae_processor, videomae_model
    global dino_processor, dino_model
    global tabular_projection, smile
    
    from transformers import (
        Wav2Vec2Processor, Wav2Vec2Model,
        AutoTokenizer, AutoModel,
        VideoMAEImageProcessor, VideoMAEModel,
        AutoImageProcessor
    )
    import torch.nn as nn
    
    print("\n" + "="*60)
    print("Loading SOTA models...")
    print("="*60)
    
    print("[1/5] Loading Wav2Vec2-Large-XLSR-53...")
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(device)
    wav2vec_model.eval()
    wav2vec_projection = nn.Linear(1024, FEATURE_DIM).to(device)
    print(f"✓ Wav2Vec2-Large-XLSR-53 loaded (1024 -> {FEATURE_DIM})")
    
    print("\n[2/5] Loading MentalRoBERTa...")
    try:
        mental_tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base")
        mental_model = AutoModel.from_pretrained("mental/mental-roberta-base").to(device)
        mental_model.eval()
        print(f"✓ MentalRoBERTa loaded (output dim: {FEATURE_DIM})")
    except:
        from transformers import RobertaTokenizer, RobertaModel
        mental_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        mental_model = RobertaModel.from_pretrained("roberta-base").to(device)
        mental_model.eval()
        print(f"✓ RoBERTa-base loaded as fallback (output dim: {FEATURE_DIM})")
    
    print("\n[3/5] Loading VideoMAE-Base...")
    videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    videomae_model.eval()
    print(f"✓ VideoMAE-Base loaded (output dim: {FEATURE_DIM})")
    
    print("\n[4/5] Loading DINOv2-Base...")
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    dino_model.eval()
    print(f"✓ DINOv2-Base loaded (output dim: {FEATURE_DIM})")
    
    print("\n[5/5] Setting up Tabular projection...")
    tabular_projection = nn.Linear(100, FEATURE_DIM).to(device)
    print(f"✓ Tabular projection ready (100 -> {FEATURE_DIM})")
    
    print("\n[Bonus] Loading eGeMAPSS...")
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print("✓ eGeMAPSS extractor ready (88 acoustic features)")
    
    print("\n" + "="*60)
    print("✓ All SOTA models loaded successfully!")
    print("="*60)

import librosa
import cv2
from PIL import Image

@torch.no_grad()
def extract_audio_features(audio_path, sample_rate=16000):
    """Extract audio features: Wav2Vec2-XLSR + eGeMAPSS -> 768-dim."""
    try:
        waveform, sr = librosa.load(audio_path, sr=sample_rate)
        
        max_samples = sample_rate * 30
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        inputs = wav2vec_processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        ).to(device)
        
        outputs = wav2vec_model(**inputs)
        wav2vec_feat = outputs.last_hidden_state.mean(dim=1)  # [1, 1024]
        wav2vec_feat = wav2vec_projection(wav2vec_feat)  # [1, 768]
        
        try:
            egemaps = smile.process_file(str(audio_path))
            egemaps_feat = egemaps.values.flatten()  # (88,)
            egemaps_feat = (egemaps_feat - egemaps_feat.mean()) / (egemaps_feat.std() + 1e-8)
        except:
            egemaps_feat = np.zeros(88)
        
        final_feat = wav2vec_feat.cpu().numpy().squeeze()  # (768,)
        final_feat[-88:] = final_feat[-88:] * 0.5 + egemaps_feat * 0.5
        
        return final_feat.astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)

@torch.no_grad()
def extract_text_features(text):
    """Extract text features using MentalRoBERTa -> 768-dim."""
    try:
        if not text or len(text.strip()) == 0:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        
        inputs = mental_tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        outputs = mental_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return features.cpu().numpy().squeeze().astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)

@torch.no_grad()
def extract_video_features(video_path, num_frames=16):
    """Extract video features using VideoMAE -> 768-dim."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        inputs = videomae_processor(list(frames), return_tensors="pt").to(device)
        outputs = videomae_model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)
        
        return features.cpu().numpy().squeeze().astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting video: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)

@torch.no_grad()
def extract_face_features(image_path=None, video_path=None):
    """Extract face features using DINOv2-Base -> 768-dim."""
    try:
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert('RGB')
        elif video_path and Path(video_path).exists():
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return np.zeros(FEATURE_DIM, dtype=np.float32)
        else:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        
        return features.cpu().numpy().squeeze().astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting face: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)

def extract_tabular_features(participant_dir):
    """Extract tabular features from OpenFace AUs -> 768-dim."""
    try:
        features = []
        
        if participant_dir is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        
        au_files = list(Path(participant_dir).glob('*_CLNF_AUs.txt')) + \
                   list(Path(participant_dir).glob('*AUs*.csv')) + \
                   list(Path(participant_dir).glob('*.csv'))
        
        if au_files:
            for au_file in au_files[:1]:
                try:
                    df = pd.read_csv(au_file, sep=',|\t', engine='python')
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        au_features = df[numeric_cols].mean().values
                        features.extend(au_features[:100])
                except:
                    pass
        
        if len(features) == 0:
            features = np.zeros(100, dtype=np.float32)
        elif len(features) < 100:
            features = np.pad(features, (0, 100 - len(features)))
        else:
            features = np.array(features[:100])
        
        features_tensor = torch.from_numpy(np.array(features)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            projected = tabular_projection(features_tensor)
        
        return projected.cpu().numpy().squeeze().astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting tabular: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)

def extract_participant_data(pid):
    """Extract tar.gz or zip for a participant to temp directory."""
    extract_dir = Path(TEMP_PATH) / f'{pid}_P'
    
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    
    tar_path = Path(DATA_PATH) / 'Extended-DAIC-WOZ' / 'data' / f'{pid}_P.tar.gz'
    if tar_path.exists():
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(TEMP_PATH)
        return extract_dir
    
    zip_path = Path(DATA_PATH) / 'DAIC-WOZ' / f'{pid}_P.zip'
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(TEMP_PATH)
        return extract_dir
    
    return None

def find_media_files(participant_dir):
    """Find audio, video, and transcript files."""
    files = {'audio': None, 'video': None, 'transcript': None, 'images': []}
    
    if not participant_dir or not participant_dir.exists():
        return files
    
    for pattern in ['*AUDIO.wav', '*_audio.wav', '*.wav']:
        found = list(participant_dir.rglob(pattern))
        if found:
            files['audio'] = found[0]
            break
    
    for pattern in ['*VIDEO.mp4', '*_video.mp4', '*.mp4', '*.avi']:
        found = list(participant_dir.rglob(pattern))
        if found:
            files['video'] = found[0]
            break
    
    for pattern in ['*TRANSCRIPT.csv', '*transcript*.txt', '*_TRANSCRIPT.csv']:
        found = list(participant_dir.rglob(pattern))
        if found:
            files['transcript'] = found[0]
            break
    
    files['images'] = list(participant_dir.rglob('*.jpg')) + list(participant_dir.rglob('*.png'))
    
    return files

def load_transcript(transcript_path):
    """Load and concatenate transcript text."""
    if transcript_path is None:
        return ""
    
    try:
        if str(transcript_path).endswith('.csv'):
            df = pd.read_csv(transcript_path, sep='\t')
            text_col = [c for c in df.columns if 'text' in c.lower() or 'value' in c.lower()]
            if text_col:
                return ' '.join(df[text_col[0]].astype(str).tolist())
        else:
            with open(transcript_path, 'r') as f:
                return f.read()
    except:
        pass
    return ""

def process_participant(pid):
    """Extract all SOTA features for a single participant."""
    
    
    participant_dir = extract_participant_data(pid)
    if participant_dir is None:
        np.save(f'{OUTPUT_PATH}/audio/{pid}_audio.npy', np.zeros(FEATURE_DIM, dtype=np.float32))
        np.save(f'{OUTPUT_PATH}/text/{pid}_text.npy', np.zeros(FEATURE_DIM, dtype=np.float32))
        np.save(f'{OUTPUT_PATH}/video/{pid}_video.npy', np.zeros(FEATURE_DIM, dtype=np.float32))
        np.save(f'{OUTPUT_PATH}/image/{pid}_image.npy', np.zeros(FEATURE_DIM, dtype=np.float32))
        np.save(f'{OUTPUT_PATH}/tabular/{pid}_tabular.npy', np.zeros(FEATURE_DIM, dtype=np.float32))
        np.savez(f'{OUTPUT_PATH}/combined/{pid}_features.npz',
                 audio=np.zeros(FEATURE_DIM, dtype=np.float32),
                 text=np.zeros(FEATURE_DIM, dtype=np.float32),
                 video=np.zeros(FEATURE_DIM, dtype=np.float32),
                 image=np.zeros(FEATURE_DIM, dtype=np.float32),
                 tabular=np.zeros(FEATURE_DIM, dtype=np.float32))
        return False, "no_data"
    
    files = find_media_files(participant_dir)
    
    audio_feat = extract_audio_features(str(files['audio'])) if files['audio'] else np.zeros(FEATURE_DIM, dtype=np.float32)
    
    transcript = load_transcript(files['transcript'])
    text_feat = extract_text_features(transcript)
    
    video_feat = extract_video_features(str(files['video'])) if files['video'] else np.zeros(FEATURE_DIM, dtype=np.float32)
    
    if files['images']:
        face_feat = extract_face_features(image_path=str(files['images'][len(files['images'])//2]))
    elif files['video']:
        face_feat = extract_face_features(video_path=str(files['video']))
    else:
        face_feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    
    tabular_feat = extract_tabular_features(participant_dir)
    
    np.save(f'{OUTPUT_PATH}/audio/{pid}_audio.npy', audio_feat)
    np.save(f'{OUTPUT_PATH}/text/{pid}_text.npy', text_feat)
    np.save(f'{OUTPUT_PATH}/video/{pid}_video.npy', video_feat)
    np.save(f'{OUTPUT_PATH}/image/{pid}_image.npy', face_feat)
    np.save(f'{OUTPUT_PATH}/tabular/{pid}_tabular.npy', tabular_feat)
    
    np.savez(f'{OUTPUT_PATH}/combined/{pid}_features.npz',
             audio=audio_feat,
             text=text_feat,
             video=video_feat,
             image=face_feat,
             tabular=tabular_feat)
    
    if participant_dir.exists():
        shutil.rmtree(participant_dir)
    
    return True, "success"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 H⁵-OmniFusion: SOTA Feature Extraction Pipeline")
    print("="*60)
    
    PARTICIPANT_IDS = get_participant_ids()
    print(f"\nFound {len(PARTICIPANT_IDS)} participants")
    print(f"ID range: {min(PARTICIPANT_IDS)} - {max(PARTICIPANT_IDS)}")
    
    load_models()
    
    print(f"\n🚀 Processing {len(PARTICIPANT_IDS)} participants with SOTA models...")
    print("="*60)
    
    results = {'success': 0, 'skipped': 0, 'no_data': 0, 'error': 0}
    
    for i, pid in enumerate(tqdm(PARTICIPANT_IDS, desc="Extracting SOTA features")):
        try:
            success, status = process_participant(pid)
            results[status if status in results else 'success'] += 1
        except Exception as e:
            print(f"\n  PID {pid}: Error - {e}")
            results['error'] += 1
        
        if (i + 1) % 10 == 0 and device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"✓ Extraction complete!")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  No Data: {results['no_data']}")
    print(f"  Errors:  {results['error']}")
    
    print("\n📊 SOTA Feature Extraction Summary")
    print("="*60)
    
    for subdir in ['audio', 'text', 'video', 'image', 'tabular', 'combined']:
        path = Path(OUTPUT_PATH) / subdir
        if subdir == 'combined':
            files = list(path.glob('*.npz'))
        else:
            files = list(path.glob('*.npy'))
        
        if files:
            if subdir == 'combined':
                data = np.load(files[0])
                shapes = {k: data[k].shape for k in data.files}
                print(f"{subdir}: {len(files)} files, shapes: {shapes}")
            else:
                sample = np.load(files[0])
                non_zero = sum(1 for f in files if np.count_nonzero(np.load(f)) > 0)
                print(f"{subdir}: {len(files)} files, shape: {sample.shape}, non-zero: {non_zero}")
        else:
            print(f"{subdir}: 0 files")
    
    if Path(TEMP_PATH).exists():
        shutil.rmtree(TEMP_PATH)
        print("\n✓ Cleaned up temporary files")
    
    print(f"\n🎉 SOTA Feature extraction complete!")
    print(f"Output saved to: {OUTPUT_PATH}")
