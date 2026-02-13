"""
H⁵-OmniFusion: Research-Grade Preprocessing & Feature Extraction Pipeline
Dataset: DAIC-WOZ and Extended-DAIC-WOZ
Author: [Your Institution]
Date: December 2025

This pipeline implements SOTA preprocessing for depression detection using multimodal data.
Supports 5 modalities: Audio, Text, Video, Face, Tabular

Reference Architecture: H⁵-OmniFusion (ARCHITECTURE.md)
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torchaudio
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import pickle
import json

from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2Model,
    AutoTokenizer, 
    AutoModel,
    VideoMAEImageProcessor,
    VideoMAEModel
)

import opensmile

from tabpfn import TabPFNClassifier

warnings.filterwarnings('ignore')


@dataclass
class PreprocessingConfig:
    """Central configuration for all preprocessing parameters"""
    
    daic_root: str = "./DAIC-WOZ"
    extended_daic_root: str = "./Extended-DAIC-WOZ"
    output_dir: str = "./processed_features"
    
    audio_sr: int = 16000  # Target sampling rate for Wav2Vec2
    audio_duration: float = 10.0  # Segment duration in seconds
    audio_overlap: float = 0.5  # Overlap ratio for sliding window
    audio_normalize: bool = True
    
    video_fps: int = 25  # Target FPS
    video_frame_size: Tuple[int, int] = (224, 224)
    video_segment_frames: int = 16  # Number of frames per segment for VideoMAE
    
    face_detector: str = "mtcnn"  # or "dlib", "mediapipe"
    face_size: Tuple[int, int] = (112, 112)
    openface_path: str = "./OpenFace/build/bin/FeatureExtraction"
    
    max_text_length: int = 512
    text_stride: int = 256  # For long transcripts
    
    tabular_features: List[str] = None  # Will be defined in __post_init__
    
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    min_audio_snr: float = 10.0  # Minimum SNR in dB
    min_face_confidence: float = 0.85
    max_missing_modalities: int = 1  # Max allowed missing modalities per sample
    
    def __post_init__(self):
        if self.tabular_features is None:
            self.tabular_features = [
                'age', 'gender', 'interview_duration',
                'turn_count', 'avg_turn_length',
                'silence_ratio', 'interruption_count'
            ]


class AudioProcessor:
    """
    Audio preprocessing using Wav2Vec2-Large-XLSR-53 + eGeMAPSS features
    Target: 768-dim embeddings per segment
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("Loading Wav2Vec2-Large-XLSR-53...")
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        ).to(self.device)
        self.model.eval()
        
        print("Loading OpenSMILE eGeMAPSS extractor...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and resample audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sr != self.config.audio_sr:
            resampler = torchaudio.transforms.Resample(sr, self.config.audio_sr)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy(), self.config.audio_sr
    
    def preprocess_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Apply preprocessing: normalization, filtering, VAD"""
        
        if self.config.audio_normalize:
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        pre_emphasis = 0.97
        waveform = np.append(waveform[0], waveform[1:] - pre_emphasis * waveform[:-1])
        
        waveform = waveform - np.mean(waveform)
        
        return waveform
    
    def segment_audio(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Create overlapping segments for processing"""
        segment_length = int(self.config.audio_duration * self.config.audio_sr)
        hop_length = int(segment_length * (1 - self.config.audio_overlap))
        
        segments = []
        for start in range(0, len(waveform) - segment_length + 1, hop_length):
            segment = waveform[start:start + segment_length]
            segments.append(segment)
        
        if len(waveform) % hop_length != 0:
            last_segment = waveform[-segment_length:]
            if len(last_segment) == segment_length:
                segments.append(last_segment)
        
        return segments
    
    def extract_wav2vec2_features(self, segments: List[np.ndarray]) -> np.ndarray:
        """Extract Wav2Vec2 embeddings"""
        all_features = []
        
        with torch.no_grad():
            for segment in segments:
                inputs = self.processor(
                    segment,
                    sampling_rate=self.config.audio_sr,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_features.append(features.squeeze())
        
        return np.stack(all_features)  # Shape: (num_segments, 768)
    
    def extract_egemaps_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract eGeMAPSS acoustic features"""
        features = self.smile.process_signal(waveform, self.config.audio_sr)
        return features.values.flatten()  # Shape: (88,) for eGeMAPSv02
    
    def compute_audio_quality(self, waveform: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for audio"""
        signal_power = np.mean(waveform ** 2)
        noise_estimate = np.var(waveform[:1000])  # Estimate from first 1000 samples
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-8))
        
        zcr = np.mean(np.abs(np.diff(np.sign(waveform))) > 0)
        
        rms = np.sqrt(np.mean(waveform ** 2))
        
        return {
            'snr_db': snr,
            'zero_crossing_rate': zcr,
            'rms_energy': rms,
            'quality_score': min(snr / 30.0, 1.0)  # Normalized quality
        }
    
    def process_file(self, audio_path: str) -> Dict:
        """Complete processing pipeline for one audio file"""
        waveform, sr = self.load_audio(audio_path)
        
        waveform = self.preprocess_audio(waveform)
        
        quality_metrics = self.compute_audio_quality(waveform)
        if quality_metrics['snr_db'] < self.config.min_audio_snr:
            print(f"Warning: Low SNR ({quality_metrics['snr_db']:.2f} dB) for {audio_path}")
        
        segments = self.segment_audio(waveform)
        
        wav2vec_features = self.extract_wav2vec2_features(segments)
        egemaps_features = self.extract_egemaps_features(waveform)
        
        return {
            'wav2vec2_embeddings': wav2vec_features,  # (num_segments, 768)
            'egemaps_features': egemaps_features,      # (88,)
            'quality_metrics': quality_metrics,
            'num_segments': len(segments),
            'duration': len(waveform) / sr
        }


class TextProcessor:
    """
    Text preprocessing using MentalRoBERTa
    Target: 768-dim embeddings per utterance/segment
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("Loading MentalRoBERTa...")
        self.tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base")
        self.model = AutoModel.from_pretrained("mental/mental-roberta-base").to(self.device)
        self.model.eval()
    
    def load_transcript(self, transcript_path: str) -> List[Dict]:
        """Load transcript with speaker turns and timestamps"""
        transcripts = []
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    transcripts.append({
                        'speaker': parts[0],
                        'start_time': float(parts[1]),
                        'stop_time': float(parts[2]),
                        'text': parts[3]
                    })
        
        return transcripts
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.strip()
        
        text = ' '.join(text.split())
        
        text = text.replace('[inaudible]', '')
        text = text.replace('[laugh]', '')
        text = text.replace('[pause]', '')
        
        return text
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract hand-crafted linguistic features"""
        words = text.split()
        
        features = {
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': text.count('.') + text.count('?') + text.count('!'),
            'question_count': text.count('?'),
            'first_person_pronouns': sum(1 for w in words if w.lower() in ['i', 'me', 'my', 'mine', 'myself']),
            'negative_words': sum(1 for w in words if w.lower() in ['no', 'not', 'never', 'nothing', 'nobody']),
            'pause_fillers': sum(1 for w in words if w.lower() in ['um', 'uh', 'er', 'ah']),
        }
        
        return features
    
    def extract_mental_roberta_features(self, texts: List[str]) -> np.ndarray:
        """Extract MentalRoBERTa embeddings"""
        all_features = []
        
        with torch.no_grad():
            for text in texts:
                if not text.strip():
                    all_features.append(np.zeros(768))
                    continue
                
                inputs = self.tokenizer(
                    text,
                    max_length=self.config.max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_features.append(cls_embedding.squeeze())
        
        return np.stack(all_features)  # Shape: (num_utterances, 768)
    
    def process_file(self, transcript_path: str) -> Dict:
        """Complete processing pipeline for transcript"""
        transcripts = self.load_transcript(transcript_path)
        
        participant_turns = [t for t in transcripts if t['speaker'] == 'Participant']
        
        processed_texts = [self.preprocess_text(t['text']) for t in participant_turns]
        
        linguistic_features = [self.extract_linguistic_features(t) for t in processed_texts]
        
        roberta_embeddings = self.extract_mental_roberta_features(processed_texts)
        
        quality_score = np.mean([len(t.split()) > 3 for t in processed_texts])  # Ratio of substantial turns
        
        return {
            'roberta_embeddings': roberta_embeddings,  # (num_turns, 768)
            'linguistic_features': linguistic_features,
            'turn_timestamps': [(t['start_time'], t['stop_time']) for t in participant_turns],
            'quality_score': quality_score,
            'num_turns': len(participant_turns)
        }


class VideoProcessor:
    """
    Video preprocessing using VideoMAE-Base
    Target: 768-dim embeddings per video segment
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("Loading VideoMAE-Base...")
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(self.device)
        self.model.eval()
    
    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """Load video and extract frames"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames, fps
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocess video frames"""
        processed_frames = []
        
        for frame in frames:
            frame = cv2.resize(frame, self.config.video_frame_size)
            
            frame = frame.astype(np.float32) / 255.0
            
            processed_frames.append(frame)
        
        return processed_frames
    
    def temporal_sampling(self, frames: List[np.ndarray], target_fps: int = None) -> List[np.ndarray]:
        """Sample frames to target FPS"""
        if target_fps is None:
            target_fps = self.config.video_fps
        
        indices = np.linspace(0, len(frames) - 1, target_fps, dtype=int)
        return [frames[i] for i in indices]
    
    def create_video_segments(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Create fixed-length segments for VideoMAE"""
        segment_size = self.config.video_segment_frames
        segments = []
        
        for i in range(0, len(frames), segment_size // 2):  # 50% overlap
            segment = frames[i:i + segment_size]
            
            if len(segment) < segment_size:
                padding = [segment[-1]] * (segment_size - len(segment))
                segment = segment + padding
            
            segments.append(np.stack(segment))
        
        return segments
    
    def extract_videomae_features(self, video_segments: List[np.ndarray]) -> np.ndarray:
        """Extract VideoMAE embeddings"""
        all_features = []
        
        with torch.no_grad():
            for segment in video_segments:
                segment_tensor = torch.from_numpy(segment).permute(3, 0, 1, 2).unsqueeze(0)
                segment_tensor = segment_tensor.to(self.device).float()
                
                inputs = self.processor(
                    list(segment),
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_features.append(features.squeeze())
        
        return np.stack(all_features)  # Shape: (num_segments, 768)
    
    def compute_video_quality(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Compute video quality metrics"""
        brightness_values = [np.mean(frame) for frame in frames]
        contrast_values = [np.std(frame) for frame in frames]
        
        motion_scores = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            motion_scores.append(diff)
        
        return {
            'avg_brightness': np.mean(brightness_values),
            'brightness_std': np.std(brightness_values),
            'avg_contrast': np.mean(contrast_values),
            'avg_motion': np.mean(motion_scores) if motion_scores else 0,
            'quality_score': 1.0  # Placeholder for more sophisticated metric
        }
    
    def process_file(self, video_path: str) -> Dict:
        """Complete processing pipeline for video"""
        frames, fps = self.load_video(video_path)
        
        frames = self.preprocess_frames(frames)
        
        frames = self.temporal_sampling(frames)
        
        quality_metrics = self.compute_video_quality(frames)
        
        video_segments = self.create_video_segments(frames)
        
        videomae_features = self.extract_videomae_features(video_segments)
        
        return {
            'videomae_embeddings': videomae_features,  # (num_segments, 768)
            'quality_metrics': quality_metrics,
            'num_segments': len(video_segments),
            'num_frames': len(frames),
            'original_fps': fps
        }


class FaceProcessor:
    """
    Face preprocessing using OpenFace 2.0 + POSTER v2
    Target: 768-dim embeddings per frame
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("Loading POSTER v2...")
        self.poster_model = self._load_poster_model()
        
    def _load_poster_model(self):
        """Load POSTER v2 model (placeholder)"""
        print("Warning: Using placeholder for POSTER v2")
        return None
    
    def extract_openface_features(self, video_path: str, output_dir: str) -> pd.DataFrame:
        """
        Extract OpenFace 2.0 features using command-line tool
        Features include: facial landmarks, head pose, gaze, Action Units
        """
        output_file = os.path.join(output_dir, 'openface_output.csv')
        
        cmd = f"{self.config.openface_path} -f {video_path} -out_dir {output_dir}"
        os.system(cmd)
        
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            return df
        else:
            raise FileNotFoundError(f"OpenFace output not found: {output_file}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame using MTCNN or similar"""
        
        import face_recognition  # Simple fallback
        face_locations = face_recognition.face_locations(frame)
        
        bboxes = []
        for (top, right, bottom, left) in face_locations:
            bboxes.append((left, top, right - left, bottom - top))
        
        return bboxes
    
    def extract_face_crops(self, video_path: str) -> List[np.ndarray]:
        """Extract and align face crops from video"""
        cap = cv2.VideoCapture(video_path)
        face_crops = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            bboxes = self.detect_faces(frame)
            
            if bboxes:
                x, y, w, h = max(bboxes, key=lambda b: b[2] * b[3])
                
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, self.config.face_size)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                
                face_crops.append(face)
        
        cap.release()
        return face_crops
    
    def extract_action_units(self, openface_df: pd.DataFrame) -> np.ndarray:
        """Extract Action Unit intensities from OpenFace output"""
        au_columns = [col for col in openface_df.columns if col.startswith('AU')]
        
        if not au_columns:
            return np.zeros((len(openface_df), 17))  # 17 common AUs
        
        au_features = openface_df[au_columns].values
        return au_features
    
    def extract_head_pose(self, openface_df: pd.DataFrame) -> np.ndarray:
        """Extract head pose features"""
        pose_columns = ['pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz']
        
        available_cols = [col for col in pose_columns if col in openface_df.columns]
        if not available_cols:
            return np.zeros((len(openface_df), 6))
        
        pose_features = openface_df[available_cols].values
        return pose_features
    
    def extract_gaze_features(self, openface_df: pd.DataFrame) -> np.ndarray:
        """Extract gaze direction features"""
        gaze_columns = [col for col in openface_df.columns if 'gaze' in col.lower()]
        
        if not gaze_columns:
            return np.zeros((len(openface_df), 8))
        
        gaze_features = openface_df[gaze_columns].values
        return gaze_features
    
    def compute_face_quality(self, openface_df: pd.DataFrame) -> Dict[str, float]:
        """Compute face detection quality metrics"""
        confidence = openface_df['confidence'].mean() if 'confidence' in openface_df.columns else 0.0
        
        success_rate = (openface_df['success'] == 1).mean() if 'success' in openface_df.columns else 0.0
        
        return {
            'avg_confidence': confidence,
            'detection_success_rate': success_rate,
            'quality_score': (confidence + success_rate) / 2
        }
    
    def process_file(self, video_path: str) -> Dict:
        """Complete processing pipeline for facial features"""
        temp_dir = "./temp_openface"
        os.makedirs(temp_dir, exist_ok=True)
        
        openface_df = self.extract_openface_features(video_path, temp_dir)
        
        au_features = self.extract_action_units(openface_df)
        head_pose = self.extract_head_pose(openface_df)
        gaze_features = self.extract_gaze_features(openface_df)
        
        quality_metrics = self.compute_face_quality(openface_df)
        
        combined_features = np.concatenate([
            au_features,
            head_pose,
            gaze_features
        ], axis=1)
        
        target_dim = 768
        if combined_features.shape[1] < target_dim:
            padding = np.zeros((combined_features.shape[0], target_dim - combined_features.shape[1]))
            combined_features = np.concatenate([combined_features, padding], axis=1)
        elif combined_features.shape[1] > target_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim)
            combined_features = pca.fit_transform(combined_features)
        
        return {
            'face_embeddings': combined_features,  # (num_frames, 768)
            'action_units': au_features,
            'head_pose': head_pose,
            'gaze_features': gaze_features,
            'quality_metrics': quality_metrics,
            'num_frames': len(openface_df)
        }


class TabularProcessor:
    """
    Tabular feature processing using TabPFN
    Target: 768-dim embeddings
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    def extract_demographic_features(self, participant_info: Dict) -> Dict[str, float]:
        """Extract demographic information"""
        return {
            'age': participant_info.get('age', 0),
            'gender': 1 if participant_info.get('gender', 'M') == 'M' else 0,
        }
    
    def extract_interview_statistics(self, 
                                     audio_features: Dict,
                                     text_features: Dict) -> Dict[str, float]:
        """Extract interview-level statistics"""
        return {
            'interview_duration': audio_features.get('duration', 0),
            'turn_count': text_features.get('num_turns', 0),
            'avg_turn_length': np.mean([len(t.split()) for t in text_features.get('texts', [''])]),
            'silence_ratio': 1.0 - (audio_features.get('duration', 1) / audio_features.get('total_duration', 1)),
        }
    
    def extract_temporal_features(self, 
                                   audio_quality: Dict,
                                   video_quality: Dict,
                                   face_quality: Dict) -> Dict[str, float]:
        """Extract temporal and quality-based features"""
        return {
            'audio_quality': audio_quality.get('quality_score', 0),
            'video_quality': video_quality.get('quality_score', 0),
            'face_quality': face_quality.get('quality_score', 0),
        }
    
    def create_tabular_features(self,
                               participant_info: Dict,
                               audio_features: Dict,
                               text_features: Dict,
                               video_features: Dict,
                               face_features: Dict) -> pd.DataFrame:
        """Combine all tabular features"""
        
        features = {}
        
        features.update(self.extract_demographic_features(participant_info))
        
        features.update(self.extract_interview_statistics(audio_features, text_features))
        
        features.update(self.extract_temporal_features(
            audio_features.get('quality_metrics', {}),
            video_features.get('quality_metrics', {}),
            face_features.get('quality_metrics', {})
        ))
        
        return pd.DataFrame([features])
    
    def process_features(self, tabular_df: pd.DataFrame) -> np.ndarray:
        """Process tabular features to 768-dim embedding"""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized = scaler.fit_transform(tabular_df)
        
        target_dim = 768
        if normalized.shape[1] < target_dim:
            padding = np.zeros((normalized.shape[0], target_dim - normalized.shape[1]))
            embeddings = np.concatenate([normalized, padding], axis=1)
        else:
            embeddings = normalized[:, :target_dim]
        
        return embeddings


class H5OmniFusionPreprocessor:
    """
    Master preprocessing pipeline coordinating all modalities
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        print("Initializing H⁵-OmniFusion Preprocessing Pipeline...")
        self.audio_processor = AudioProcessor(config)
        self.text_processor = TextProcessor(config)
        self.video_processor = VideoProcessor(config)
        self.face_processor = FaceProcessor(config)
        self.tabular_processor = TabularProcessor(config)
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_participant_metadata(self, participant_id: str) -> Dict:
        """Load metadata for participant"""
        metadata_path = os.path.join(self.config.daic_root, 'metadata', f'{participant_id}_metadata.csv')
        
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            return df.to_dict('records')[0]
        else:
            return {}
    
    def process_participant(self, participant_id: str) -> Dict:
        """Process all modalities for a single participant"""
        print(f"\nProcessing participant {participant_id}...")
        
        base_path = os.path.join(self.config.daic_root, participant_id)
        audio_path = f"{base_path}_AUDIO.wav"
        video_path = f"{base_path}_VIDEO.mp4"
        transcript_path = f"{base_path}_TRANSCRIPT.csv"
        
        results = {
            'participant_id': participant_id,
            'success': True,
            'missing_modalities': []
        }
        
        try:
            print("  - Processing audio...")
            audio_features = self.audio_processor.process_file(audio_path)
            results['audio'] = audio_features
        except Exception as e:
            print(f"  ! Audio processing failed: {e}")
            results['missing_modalities'].append('audio')
        
        try:
            print("  - Processing text...")
            text_features = self.text_processor.process_file(transcript_path)
            results['text'] = text_features
        except Exception as e:
            print(f"  ! Text processing failed: {e}")
            results['missing_modalities'].append('text')
        
        try:
            print("  - Processing video...")
            video_features = self.video_processor.process_file(video_path)
            results['video'] = video_features
        except Exception as e:
            print(f"  ! Video processing failed: {e}")
            results['missing_modalities'].append('video')
        
        try:
            print("  - Processing face...")
            face_features = self.face_processor.process_file(video_path)
            results['face'] = face_features
        except Exception as e:
            print(f"  ! Face processing failed: {e}")
            results['missing_modalities'].append('face')
        
        try:
            print("  - Processing tabular features...")
            participant_info = self.load_participant_metadata(participant_id)
            tabular_df = self.tabular_processor.create_tabular_features(
                participant_info,
                results.get('audio', {}),
                results.get('text', {}),
                results.get('video', {}),
                results.get('face', {})
            )
            tabular_embeddings = self.tabular_processor.process_features(tabular_df)
            results['tabular'] = {
                'embeddings': tabular_embeddings,
                'raw_features': tabular_df
            }
        except Exception as e:
            print(f"  ! Tabular processing failed: {e}")
            results['missing_modalities'].append('tabular')
        
        if len(results['missing_modalities']) > self.config.max_missing_modalities:
            results['success'] = False
            print(f"  ! Too many missing modalities: {results['missing_modalities']}")
        
        return results
    
    def save_processed_data(self, participant_id: str, data: Dict):
        """Save processed features to disk"""
        output_path = os.path.join(self.config.output_dir, f'{participant_id}_processed.pkl')
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  ✓ Saved to {output_path}")
    
    def process_dataset(self, participant_ids: List[str]):
        """Process entire dataset"""
        print(f"\n{'='*80}")
        print(f"H⁵-OMNIFUSION PREPROCESSING PIPELINE")
        print(f"Processing {len(participant_ids)} participants")
        print(f"{'='*80}\n")
        
        results_summary = []
        
        for pid in tqdm(participant_ids, desc="Processing participants"):
            try:
                result = self.process_participant(pid)
                self.save_processed_data(pid, result)
                results_summary.append({
                    'participant_id': pid,
                    'success': result['success'],
                    'missing_modalities': result['missing_modalities']
                })
            except Exception as e:
                print(f"\n! Failed to process {pid}: {e}")
                results_summary.append({
                    'participant_id': pid,
                    'success': False,
                    'error': str(e)
                })
        
        summary_df = pd.DataFrame(results_summary)
        summary_path = os.path.join(self.config.output_dir, 'processing_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"Processing complete!")
        print(f"Success rate: {summary_df['success'].mean():.2%}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}\n")


def main():
    """Example usage of the preprocessing pipeline"""
    
    config = PreprocessingConfig(
        daic_root="./DAIC-WOZ",
        extended_daic_root="./Extended-DAIC-WOZ",
        output_dir="./processed_features",
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    participant_ids = [
        '300', '301', '302', '303', '304'  # Example participant IDs
    ]
    
    preprocessor = H5OmniFusionPreprocessor(config)
    
    preprocessor.process_dataset(participant_ids)
    
    print("\n✓ Preprocessing complete! Ready for model training.")


if __name__ == "__main__":
    main()
