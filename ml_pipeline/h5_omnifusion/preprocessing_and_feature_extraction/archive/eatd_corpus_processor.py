"""
H5-OmniFusion EATD-Corpus Processor
Handles Chinese language processing with missing video modality
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

class EATDCorpusProcessor:
    """
    Process EATD-Corpus (Mandarin Chinese) dataset.
    
    EATD-Corpus Structure per participant (t_1 to t_111):
    - label.txt: Binary depression label (0/1)
    - new_label.txt: Continuous severity score
    - positive.txt, positive.wav: Positive emotion recording
    - negative.txt, negative.wav: Negative emotion recording
    - neutral.txt, neutral.wav: Neutral recording
    """
    
    EMOTION_TYPES = ['positive', 'negative', 'neutral']
    
    def __init__(self, base_path: str, device: str = 'cuda'):
        self.base_path = base_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self._zh_tokenizer = None
        self._zh_model = None
        self._wav2vec = None
        self._wav2vec_processor = None
    
    @property
    def zh_tokenizer(self):
        if self._zh_tokenizer is None:
            from transformers import AutoTokenizer
            try:
                self._zh_tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
            except:
                self._zh_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        return self._zh_tokenizer
    
    @property
    def zh_model(self):
        if self._zh_model is None:
            from transformers import AutoModel
            try:
                self._zh_model = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(self.device).eval()
            except:
                self._zh_model = AutoModel.from_pretrained('bert-base-chinese').to(self.device).eval()
        return self._zh_model
    
    @property
    def wav2vec_processor(self):
        if self._wav2vec_processor is None:
            from transformers import Wav2Vec2FeatureExtractor
            self._wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                'facebook/wav2vec2-large-xlsr-53'
            )
        return self._wav2vec_processor
    
    @property
    def wav2vec(self):
        if self._wav2vec is None:
            from transformers import Wav2Vec2Model
            self._wav2vec = Wav2Vec2Model.from_pretrained(
                'facebook/wav2vec2-large-xlsr-53'
            ).to(self.device).eval()
        return self._wav2vec
    
    def get_participant_ids(self) -> List[str]:
        """Get list of valid participant folder names."""
        ids = []
        for name in os.listdir(self.base_path):
            if name.startswith('t_') and os.path.isdir(os.path.join(self.base_path, name)):
                ids.append(name)
        return sorted(ids)
    
    def load_labels(self, participant_id: str) -> Dict[str, Any]:
        """Load labels for a participant."""
        p_dir = os.path.join(self.base_path, participant_id)
        
        labels = {'binary': None, 'continuous': None}
        
        label_path = os.path.join(p_dir, 'label.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                labels['binary'] = int(f.read().strip())
        
        new_label_path = os.path.join(p_dir, 'new_label.txt')
        if os.path.exists(new_label_path):
            with open(new_label_path, 'r', encoding='utf-8') as f:
                labels['continuous'] = float(f.read().strip())
        
        return labels
    
    def process_audio(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """Process Chinese audio and extract Wav2Vec2 embedding."""
        import librosa
        
        waveform, sr = librosa.load(audio_path, sr=16000)
        
        from audio_enhancements import (
            PeakNormalizer, LoudnessNormalizer, VoiceActivityDetector,
            PauseAnalyzer, SighDetector, BreathIntervalAnalyzer, AudioQualityChecker
        )
        
        waveform = PeakNormalizer.normalize(waveform)
        waveform = LoudnessNormalizer().normalize(waveform)
        
        features = {}
        features.update(PauseAnalyzer().analyze(waveform))
        features.update(SighDetector().detect(waveform))
        features.update(BreathIntervalAnalyzer().extract(waveform))
        features.update(AudioQualityChecker().check(waveform))
        
        inputs = self.wav2vec_processor(waveform, sampling_rate=16000, 
                                         return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = self.wav2vec(inputs.input_values.to(self.device))
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        
        return embedding, features
    
    def process_text(self, text_path: str) -> Tuple[np.ndarray, Dict]:
        """Process Chinese text and extract Chinese-BERT embedding."""
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        inputs = self.zh_tokenizer(text, return_tensors='pt', truncation=True, 
                                    max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.zh_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        
        from text_enhancements import ChinesePsycholinguisticExtractor, MultilingualSentimentAnalyzer
        
        features = ChinesePsycholinguisticExtractor().extract(text)
        features.update(MultilingualSentimentAnalyzer().analyze_chinese(text))
        features['word_count'] = len(text)
        features['char_count'] = len(text.replace(' ', ''))
        
        return embedding, features
    
    def process_participant(self, participant_id: str) -> Dict[str, Any]:
        """Process all data for a single EATD-Corpus participant."""
        p_dir = os.path.join(self.base_path, participant_id)
        
        result = {
            'participant_id': participant_id,
            'labels': self.load_labels(participant_id),
            'audio_embeddings': {},
            'text_embeddings': {},
            'audio_features': {},
            'text_features': {},
            'modality_available': {'audio': True, 'text': True, 'video': False, 'face': False},
            'imputed_modalities': ['video', 'face']
        }
        
        for emotion in self.EMOTION_TYPES:
            audio_path = os.path.join(p_dir, f'{emotion}.wav')
            text_path = os.path.join(p_dir, f'{emotion}.txt')
            
            if os.path.exists(audio_path):
                try:
                    emb, feats = self.process_audio(audio_path)
                    result['audio_embeddings'][emotion] = emb
                    result['audio_features'][emotion] = feats
                except Exception as e:
                    print(f"Audio error {participant_id}/{emotion}: {e}")
            
            if os.path.exists(text_path):
                try:
                    emb, feats = self.process_text(text_path)
                    result['text_embeddings'][emotion] = emb
                    result['text_features'][emotion] = feats
                except Exception as e:
                    print(f"Text error {participant_id}/{emotion}: {e}")
        
        result['audio_embedding_combined'] = self._aggregate_embeddings(
            list(result['audio_embeddings'].values())
        )
        result['text_embedding_combined'] = self._aggregate_embeddings(
            list(result['text_embeddings'].values())
        )
        
        from fusion_enhancements import ModalityImputer
        imputer = ModalityImputer()
        
        embeddings = {
            'audio': result['audio_embedding_combined'],
            'text': result['text_embedding_combined']
        }
        imputed = imputer.impute(embeddings, ['audio', 'text'])
        result['video_embedding'] = imputed['video']
        result['face_embedding'] = imputed['face']
        
        return result
    
    def _aggregate_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Aggregate multiple embeddings via mean pooling."""
        if not embeddings:
            return np.zeros(768, dtype=np.float32)
        valid = [e for e in embeddings if e is not None and len(e) == 768]
        if not valid:
            return np.zeros(768, dtype=np.float32)
        return np.mean(valid, axis=0).astype(np.float32)
    
    def process_all(self, output_dir: str) -> None:
        """Process all EATD-Corpus participants and save features."""
        import json
        from tqdm import tqdm
        
        os.makedirs(output_dir, exist_ok=True)
        
        for pid in tqdm(self.get_participant_ids(), desc="Processing EATD-Corpus"):
            try:
                result = self.process_participant(pid)
                
                np.save(os.path.join(output_dir, f'{pid}_audio.npy'), 
                       result['audio_embedding_combined'])
                np.save(os.path.join(output_dir, f'{pid}_text.npy'), 
                       result['text_embedding_combined'])
                np.save(os.path.join(output_dir, f'{pid}_video.npy'), 
                       result['video_embedding'])
                np.save(os.path.join(output_dir, f'{pid}_face.npy'), 
                       result['face_embedding'])
                
                metadata = {
                    'participant_id': pid,
                    'labels': result['labels'],
                    'modality_available': result['modality_available'],
                    'imputed_modalities': result['imputed_modalities'],
                    'audio_features': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv 
                                          for kk, vv in v.items()} 
                                      for k, v in result['audio_features'].items()},
                    'text_features': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv 
                                         for kk, vv in v.items()} 
                                     for k, v in result['text_features'].items()}
                }
                
                with open(os.path.join(output_dir, f'{pid}_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {pid}: {e}")


class CrossLingualProcessor:
    """Fallback processor using XLM-RoBERTa for language-agnostic processing."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._tokenizer = None
        self._model = None
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import XLMRobertaTokenizer
            self._tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            from transformers import XLMRobertaModel
            self._model = XLMRobertaModel.from_pretrained('xlm-roberta-base').to(self.device).eval()
        return self._model
    
    def extract_embedding(self, text: str) -> np.ndarray:
        """Extract language-agnostic embedding."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
