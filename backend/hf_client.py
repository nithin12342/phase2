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


class HFClient:
    """Stateless client for HuggingFace Inference API (feature-extraction)."""

    def __init__(self, token: str):
        self.headers = {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _feature_extract_text(self, model_id: str, text: str) -> Optional[np.ndarray]:
        """Send text to HF feature-extraction endpoint → embedding."""
        url = f"{HF_API_BASE}/pipeline/feature-extraction/{model_id}"
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        resp = requests.post(url, headers=self.headers, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # HF returns [[token_embeddings...]] → mean-pool to single vector
        arr = np.array(data, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]  # remove batch dim → (seq_len, hidden)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)  # mean-pool over tokens → (hidden,)
        return arr[:EMBED_DIM].astype("float32")

    def _feature_extract_binary(self, model_id: str, data: bytes,
                                 content_type: str = "application/octet-stream") -> Optional[np.ndarray]:
        """Send binary (audio/image) to HF feature-extraction → embedding."""
        url = f"{HF_API_BASE}/pipeline/feature-extraction/{model_id}"
        headers = {**self.headers, "Content-Type": content_type}
        resp = requests.post(url, headers=headers, data=data, timeout=_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        arr = np.array(result, dtype=np.float32)
        # Flatten/pool similar to text
        while arr.ndim > 1:
            arr = arr.mean(axis=0)
        if arr.shape[0] < EMBED_DIM:
            arr = np.pad(arr, (0, EMBED_DIM - arr.shape[0]))
        return arr[:EMBED_DIM].astype("float32")

    # ------------------------------------------------------------------
    # public API — matches models.py interface
    # ------------------------------------------------------------------
    def get_text_embedding(self, text: str) -> Tuple[np.ndarray, Optional[str]]:
        """Extract text embedding via MentalRoBERTa."""
        try:
            logger.info("HF API → text embedding (mental/mental-roberta-base)")
            emb = self._feature_extract_text(MODELS["text"], text)
            return emb, None
        except Exception as e:
            logger.error(f"HF text embedding failed: {e}")
            return np.zeros(EMBED_DIM, dtype=np.float32), str(e)

    def get_audio_embedding(self, audio_bytes: bytes) -> Tuple[np.ndarray, Optional[str]]:
        """Extract audio embedding via Wav2Vec2."""
        try:
            logger.info("HF API → audio embedding (wav2vec2-large-xlsr-53)")
            emb = self._feature_extract_binary(MODELS["audio"], audio_bytes, "audio/wav")
            return emb, None
        except Exception as e:
            logger.error(f"HF audio embedding failed: {e}")
            return np.zeros(EMBED_DIM, dtype=np.float32), str(e)

    def get_image_embedding(self, image_bytes: bytes) -> Tuple[np.ndarray, Optional[str]]:
        """Extract image embedding via DinoV2."""
        try:
            logger.info("HF API → image embedding (dinov2-base)")
            emb = self._feature_extract_binary(MODELS["image"], image_bytes, "image/jpeg")
            return emb, None
        except Exception as e:
            logger.error(f"HF image embedding failed: {e}")
            return np.zeros(EMBED_DIM, dtype=np.float32), str(e)

    def get_video_embedding(self, video_bytes: bytes) -> Tuple[np.ndarray, Optional[str]]:
        """
        Extract video embedding.
        VideoMAE API may not accept raw video — fall back to extracting
        a key frame and passing it through DinoV2 as a visual proxy.
        """
        try:
            logger.info("HF API → video embedding (trying VideoMAE)")
            emb = self._feature_extract_binary(MODELS["video"], video_bytes, "video/mp4")
            return emb, None
        except Exception as e:
            logger.warning(f"VideoMAE failed ({e}), falling back to DinoV2 frame extraction")
            try:
                # Fall back: extract a single frame and use DinoV2
                frame_bytes = self._extract_key_frame(video_bytes)
                if frame_bytes:
                    emb = self._feature_extract_binary(MODELS["image"], frame_bytes, "image/jpeg")
                    return emb, "Used frame extraction fallback"
            except Exception as e2:
                logger.error(f"Video fallback also failed: {e2}")
            return np.zeros(EMBED_DIM, dtype=np.float32), str(e)

    def _extract_key_frame(self, video_bytes: bytes) -> Optional[bytes]:
        """Extract a single representative frame from video bytes."""
        try:
            import cv2
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name
            try:
                cap = cv2.VideoCapture(tmp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Grab frame at 25% mark
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, total // 4))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    _, buf = cv2.imencode(".jpg", frame)
                    return buf.tobytes()
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        return None
