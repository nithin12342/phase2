from __future__ import annotations
"""
Multimodal Models for Depression Detection — HuggingFace API + Local Fusion
===========================================================================
Architecture:
  - Feature extraction: HuggingFace Inference API (remote)
  - Fusion + classification: H5-OmniFusion model (local, ~134 MB checkpoint)

This keeps the Azure container lightweight (~200 MB RAM idle).
"""
import os
import sys
import gc
import logging
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────
# models.py is at /app/models.py in Docker → dirname = /app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_PIPELINE_DIR = os.path.join(BASE_DIR, "ml_pipeline")
if ML_PIPELINE_DIR not in sys.path:
    sys.path.append(ML_PIPELINE_DIR)

from config import get_settings
settings = get_settings()

# ── deferred heavy imports ───────────────────────────────────────────────
_FUSION_READY = False
_LOAD_ERROR = "Not initialized"
torch = None

def _import_fusion():
    """Import only torch + fusion model architecture (no HF transformers)."""
    global _FUSION_READY, _LOAD_ERROR, torch
    if _FUSION_READY:
        return True
    try:
        import torch as _t
        torch = _t
        _FUSION_READY = True
        return True
    except ImportError as e:
        _LOAD_ERROR = str(e)
        logger.error(f"torch import failed: {e}")
        return False


# ── survey fallback ──────────────────────────────────────────────────────
_survey_context: dict = {}

def set_survey_context(data: dict):
    global _survey_context
    _survey_context = data

def _survey_fallback(modalities: list) -> str:
    score, mx = 0, 0
    for k in ("growing_stress", "changes_habits", "mental_health_history",
              "family_history", "coping_struggles", "social_weakness"):
        mx += 10
        if str(_survey_context.get(k, "")).lower() in ("yes", "true", "1"):
            score += 10
    pct = (score / mx * 100) if mx else 0
    lvl = "High" if pct > 50 else "Low"
    return (f"Analysis Result (Survey-Based)\n"
            f"Modalities: {', '.join(modalities)}\n"
            f"Depression Risk: {pct:.1f}% ({lvl})\n"
            f"Note: Fusion model unavailable — used questionnaire scoring.")


# ── HuggingFace client singleton ─────────────────────────────────────────
_hf = None

def _get_hf():
    global _hf
    if _hf is None:
        from hf_client import HFClient
        token = settings.huggingface_token or os.environ.get("HUGGINGFACE_TOKEN", "")
        if not token:
            raise RuntimeError("HUGGINGFACE_TOKEN not configured")
        _hf = HFClient(token)
    return _hf


# ── Fusion model (loaded once, kept in RAM — ~134 MB) ────────────────────
_fusion_model = None
_fusion_error = None

def _load_fusion_model():
    global _fusion_model, _fusion_error
    if _fusion_model is not None:
        return _fusion_model
    if not _import_fusion():
        return None

    # Locate checkpoint
    ckpt_path = os.path.join(ML_PIPELINE_DIR, "h5_omnifusion", "checkpoints",
                             "fold4_phase12_latest.pt")
    if settings.custom_model_filename and os.path.exists(settings.custom_model_filename):
        ckpt_path = settings.custom_model_filename

    if not os.path.exists(ckpt_path):
        _fusion_error = f"Checkpoint not found: {ckpt_path}"
        logger.warning(_fusion_error)
        return None

    try:
        logger.info(f"Loading H5-OmniFusion checkpoint: {ckpt_path}")
        # import architecture
        # Import actual Config class
        from h5_omnifusion.src.config import Config
        from h5_omnifusion.src.models.h5_omnifusion import H5OmniFusion

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Manually construct config to match "lite" tier
        config = Config()
        config.d_model = 256
        config.moe.expert_hidden_dim = 128
        config.moe.n_quality_features = 5
        config.fusion.n_latents = 16
        config.fusion.n_perceiver_blocks = 1
        config.fusion.local_n_heads = 4
        config.fusion.modality_n_heads = 4
        config.fusion.modality_n_layers = 1
        config.fusion.perceiver_n_heads = 4
        config.audio.n_mamba_layers = 1
        config.text.n_mamba_layers = 1
        config.video.n_mamba_layers = 1
        config.face.n_lstm_layers = 1
        config.text.use_kan = False
        config.tabular.use_kan = False
        config.video.use_timesformer = False

        model = H5OmniFusion(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        _fusion_model = model
        logger.info("✅ H5-OmniFusion fusion model loaded successfully")
        return model
    except Exception as e:
        _fusion_error = str(e)
        logger.error(f"Fusion model load failed: {e}")
        return None


# =====================================================================
# Public API — called by main.py endpoints
# =====================================================================

def get_text_embedding(text: str):
    return _get_hf().get_text_embedding(text)

def get_audio_embedding(audio_bytes: bytes):
    return _get_hf().get_audio_embedding(audio_bytes)

def get_video_embedding(video_bytes: bytes):
    return _get_hf().get_video_embedding(video_bytes)

def get_image_embedding(image_bytes: bytes):
    return _get_hf().get_image_embedding(image_bytes)

def get_tabular_embedding(tabular_data: dict):
    """Tabular data is encoded as a zero vector placeholder."""
    return np.zeros(768, dtype=np.float32), None


def get_fusion_prediction(embeddings: dict) -> str:
    """Run local fusion model on HF-produced embeddings."""
    modalities = [k for k, v in embeddings.items() if v is not None]
    if not modalities:
        return "No input provided."

    model = _load_fusion_model()
    if model is None:
        return _survey_fallback(modalities)

    try:
        inputs = {}
        for k, v in embeddings.items():
            if v is not None and k in ("text", "audio", "video", "image", "tabular"):
                key = "face_features" if k == "image" else f"{k}_features"
                inputs[key] = torch.from_numpy(v).float().unsqueeze(0)

        with torch.no_grad():
            outputs, _ = model(inputs)
            prob = float(outputs["binary_prob"].item())
            risk = prob * 100

        level = "High" if risk > 50 else "Low"
        return (f"H5-OmniFusion Analysis\n"
                f"Modalities: {', '.join(modalities)}\n"
                f"Depression Risk: {risk:.1f}% ({level})\n"
                f"Model: H5-OmniFusion v2 (Champion Fold-4)")
    except Exception as e:
        logger.error(f"Fusion inference failed: {e}")
        return _survey_fallback(modalities)


def get_model_status() -> dict:
    return {
        "mode": "hf_api_fusion",
        "fusion_loaded": _fusion_model is not None,
        "fusion_error": _fusion_error,
        "hf_configured": bool(settings.huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")),
        "h5_loaded": _fusion_model is not None,
    }


def load_all_models():
    """Pre-load fusion model (can be called from startup or first request)."""
    _load_fusion_model()