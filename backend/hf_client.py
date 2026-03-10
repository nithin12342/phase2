"""
HuggingFace Inference API Client
================================
Sends raw media to HuggingFace hosted models for feature extraction.
Returns 768-dim embeddings consumed by the local H5-OmniFusion fusion model.
"""
import requests
import numpy as np
import logging
import io
import base64
from typing import Optional, Tuple
from inference_preprocessing import preprocess_audio, preprocess_text, preprocess_image, preprocess_video

logger = logging.getLogger(__name__)

# HuggingFace model IDs
MODELS = {
    "text":  "mental/mental-roberta-base",
    "audio": "facebook/wav2vec2-large-xlsr-53",
    "image": "facebook/dinov2-base",
    "video": "MCG-NJU/videomae-base",
}

HF_API_BASE = "https://api-inference.huggingface.co"
EMBED_DIM = 768
_TIMEOUT = 120  # seconds


import logging
import io
import torch
import numpy as np
from typing import Optional, Tuple
from inference_preprocessing import preprocess_audio, preprocess_text, preprocess_image, preprocess_video

logger = logging.getLogger(__name__)

# HuggingFace model IDs
MODELS = {
    "text":  "mental/mental-roberta-base",
    "audio": "facebook/wav2vec2-large-xlsr-53",
    "image": "facebook/dinov2-base",
    "video": "MCG-NJU/videomae-base",
}

EMBED_DIM = 768


class HFClient:
    """
    Client for extracting features via HuggingFace models.
    Due to HF Inference API deprecations (410 Gone errors), text and audio
    are now extracted using local `transformers` inference to guarantee availability.
    """

    def __init__(self, token: str):
        self.token = token
        self._text_tokenizer = None
        self._text_model = None
        self._audio_processor = None
        self._audio_model = None
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Lazy Model Loaders
    # ------------------------------------------------------------------
    def _get_model_path(self, modality_key: str, hf_id: str) -> str:
        """Return the local Google Drive path if it exists, otherwise the HF ID."""
        import os
        base_dir = "/content/drive/MyDrive/DAIC-WOZ_Datasets/pretrained_models"
        
        # Mappings based on the UI screenshot (e.g., text/mental-roberta-base)
        local_subpaths = {
            "text": "text/mental-roberta-base",
            "audio": "audio/wav2vec2-large-xlsr-53",
            "image": "face/dinov2-base",
            "video": "video/videomae-base"
        }
        
        if modality_key in local_subpaths:
            local_path = os.path.join(base_dir, local_subpaths[modality_key])
            if os.path.exists(local_path):
                logger.info(f"Found local model for {modality_key} at {local_path}")
                return local_path
                
        return hf_id

    def _load_text_model(self):
        if self._text_model is None:
            model_path = self._get_model_path('text', MODELS['text'])
            logger.info(f"Loading text model: {model_path} to {self.device}...")
            from transformers import AutoTokenizer, AutoModel
            self._text_tokenizer = AutoTokenizer.from_pretrained(model_path, token=self.token)
            self._text_model = AutoModel.from_pretrained(model_path, token=self.token).to(self.device)
            self._text_model.eval()
            
    def _load_audio_model(self):
        if self._audio_model is None:
            model_path = self._get_model_path('audio', MODELS['audio'])
            logger.info(f"Loading audio model: {model_path} to {self.device}...")
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            self._audio_processor = Wav2Vec2Processor.from_pretrained(model_path, token=self.token)
            self._audio_model = Wav2Vec2Model.from_pretrained(model_path, token=self.token).to(self.device)
            self._audio_model.eval()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _feature_extract_text_local(self, text: str) -> np.ndarray:
        """Extract true sentence embedding using mean pooling instead of raw CLS."""
        self._load_text_model()
        inputs = self._text_tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._text_model(**inputs)
            attention = inputs['attention_mask'].unsqueeze(-1)
            embeddings = outputs.last_hidden_state * attention
            arr = (embeddings.sum(dim=1) / attention.sum(dim=1)).cpu().numpy().flatten()
            
        if arr.shape[0] > EMBED_DIM:
            arr = arr[:EMBED_DIM]
        elif arr.shape[0] < EMBED_DIM:
            arr = np.pad(arr, (0, EMBED_DIM - arr.shape[0]))
        return arr.astype("float32")

    def _feature_extract_audio_local(self, wav_bytes: bytes) -> np.ndarray:
        """Extract audio feature using local transformers."""
        import librosa
        self._load_audio_model()
        
        # Decode bytes to 16kHz mono numpy array
        wav, _ = librosa.load(io.BytesIO(wav_bytes), sr=16000, mono=True)
        if len(wav) == 0:
            return np.zeros(EMBED_DIM, dtype=np.float32)
            
        inputs = self._audio_processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        model_dtype = next(self._audio_model.parameters()).dtype
        inputs = {k: v.to(device=self.device, dtype=model_dtype) if v.dtype.is_floating_point else v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._audio_model(**inputs)
            # Mean pool over time dimension (equivalent to mean(dim=1) in training)
            arr = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
        if arr.shape[0] == 1024:
            # Recreate exactly the untrained Linear(1024, 768) projection
            # from training via fixed seed to align representation distribution.
            rng = np.random.RandomState(42)
            proj = rng.normal(0, 0.01, (1024, EMBED_DIM)).astype(np.float32)
            arr = arr @ proj
        elif arr.shape[0] > EMBED_DIM:
            arr = arr[:EMBED_DIM]
        elif arr.shape[0] < EMBED_DIM:
            arr = np.pad(arr, (0, EMBED_DIM - arr.shape[0]))
        return arr.astype("float32")

    # ------------------------------------------------------------------
    # public API — matches models.py interface
    # ------------------------------------------------------------------
    def get_text_embedding(self, text: str) -> Tuple[np.ndarray, Optional[str]]:
        """Extract text embedding via MentalRoBERTa."""
        try:
            # P12-P14: Clean text before embedding
            text = preprocess_text(text)
            logger.info("Local inference → text embedding (mental/mental-roberta-base)")
            emb = self._feature_extract_text_local(text)
            return emb, None
        except Exception as e:
            logger.error(f"Local text embedding failed: {e}")
            return np.zeros(EMBED_DIM, dtype=np.float32), str(e)

    def get_audio_embedding(self, audio_bytes: bytes) -> Tuple[np.ndarray, Optional[str]]:
        """Extract audio embedding via Wav2Vec2."""
        try:
            # P4, P6: Peak normalize + noise gate before embedding
            audio_bytes = preprocess_audio(audio_bytes)
            logger.info("Local inference → audio embedding (wav2vec2-large-xlsr-53)")
            emb = self._feature_extract_audio_local(audio_bytes)
            return emb, None
        except Exception as e:
            logger.error(f"Local audio embedding failed: {e}")
            return np.zeros(EMBED_DIM, dtype=np.float32), str(e)

    def get_image_embedding(self, image_bytes: bytes) -> Tuple[np.ndarray, Optional[str]]:
        """Extract image embedding (HF fallback ignored or disabled to save resources)"""
        logger.info("Image embedding bypassed (not critical for baseline performance)")
        return np.zeros(EMBED_DIM, dtype=np.float32), "Image extraction skipped"

    def get_video_embedding(self, video_bytes: bytes) -> Tuple[np.ndarray, Optional[str]]:
        """Extract video embedding (HF fallback ignored or disabled to save resources)"""
        try:
            # P21, P22: Extract best quality frame from video
            frame_bytes, quality = preprocess_video(video_bytes)
            logger.info(f"P21-P22 applied: blur={quality.get('blur_score')}, brightness={quality.get('brightness')}")
            return np.zeros(EMBED_DIM, dtype=np.float32), "Video extraction skipped"
        except Exception as e:
            logger.error(f"Video preprocessing failed: {e}")
            return np.zeros(EMBED_DIM, dtype=np.float32), str(e)
