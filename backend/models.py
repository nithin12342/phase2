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
# models.py is at backend/models.py → go up one level to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                   # .../backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)                                # .../phase 2
ML_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline")
# Docker compat: if ml_pipeline is co-located (e.g. /app/ml_pipeline), use that
if not os.path.exists(ML_PIPELINE_DIR):
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

    from collections import OrderedDict

    # Locate checkpoint
    H5_ROOT = os.path.join(ML_PIPELINE_DIR, "h5_omnifusion")
    ckpt_path = os.path.join(H5_ROOT, "checkpoints", "fold4_phase12_latest.pt")
    if settings.custom_model_filename and os.path.exists(settings.custom_model_filename):
        ckpt_path = settings.custom_model_filename

    if not os.path.exists(ckpt_path):
        _fusion_error = f"Checkpoint not found: {ckpt_path}"
        logger.warning(_fusion_error)
        return None

    try:
        # Use importlib to load H5Config directly from file path.
        # This bypasses the conflict where Python caches backend/config.py
        # as the "config" module, preventing config.model_config from resolving.
        import importlib.util

        model_config_path = os.path.join(H5_ROOT, "config", "model_config.py")
        spec = importlib.util.spec_from_file_location("h5_model_config", model_config_path)
        h5_cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h5_cfg_mod)
        H5Config = h5_cfg_mod.H5Config
        ComputeTier = h5_cfg_mod.ComputeTier

        # Import the model architecture (src.models needs h5_omnifusion on path)
        if H5_ROOT not in sys.path:
            sys.path.insert(0, H5_ROOT)
        from src.models.h5_omnifusion import H5OmniFusion

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = checkpoint.get("model_state_dict", checkpoint)
        sd = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                         for k, v in sd.items())

        # ── Auto-detect config from checkpoint (mirrors notebook) ────────
        d_model = 256
        if "latent_fusion.latent_init" in sd:
            d_model = sd["latent_fusion.latent_init"].shape[1]

        n_latents = 16
        if "latent_fusion.latent_init" in sd:
            n_latents = sd["latent_fusion.latent_init"].shape[0]

        expert_hidden = 128
        if "moe.audio_expert.network.0.weight" in sd:
            expert_hidden = sd["moe.audio_expert.network.0.weight"].shape[0]

        bidirectional = True
        if "face_encoder.seq_proj.weight" in sd:
            bidirectional = (sd["face_encoder.seq_proj.weight"].shape[1] == d_model)

        au_dim = 35
        if "face_encoder.au_proj.weight" in sd:
            au_dim = sd["face_encoder.au_proj.weight"].shape[1]

        tier = ComputeTier.MEDIUM if d_model > 64 else ComputeTier.MICRO
        config = H5Config.from_tier(tier)
        config.d_model = d_model
        config.fusion.n_latents = n_latents
        config.moe.expert_hidden_dim = expert_hidden
        config.face.bidirectional = bidirectional
        config.face.au_dim = au_dim
        if "moe.gate.gate_network.0.weight" in sd:
            config.moe.gate_hidden_dim = sd["moe.gate.gate_network.0.weight"].shape[0]

        model = H5OmniFusion(config)
        msg = model.load_state_dict(sd, strict=False)
        logger.info(f"Weights loaded — missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")
        model.eval()
        _fusion_model = model
        logger.info("✅ H5-OmniFusion fusion model loaded successfully")
        return model
    except Exception as e:
        import traceback
        _fusion_error = str(e)
        logger.error(f"Fusion model load failed: {e}\n{traceback.format_exc()}")
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
            phq = float(outputs["phq_score"].item())

        status = "Depression" if prob >= 0.50 else "Not Depressed"

        # Build per-modality model list
        hf_models = {
            "text": "mental/mental-roberta-base",
            "audio": "facebook/wav2vec2-large-xlsr-53",
            "image": "facebook/dinov2-base",
            "video": "MCG-NJU/videomae-base",
            "tabular": "Local (zero-vector)"
        }
        models_used = "\n".join(f"  • {m}: {hf_models.get(m, 'N/A')}" for m in modalities)

        # Build preprocessing steps applied
        preproc_steps = {
            "audio": "P4 Peak Norm, P6 Noise Gate",
            "text": "P12-P14 Clean/Normalize",
            "image": "P23-P24 Resize+Normalize",
            "video": "P21-P22 Frame Extract+Quality Filter",
            "tabular": "Survey encoding"
        }
        preproc_applied = "\n".join(f"  • {m}: {preproc_steps.get(m, 'None')}" for m in modalities)

        return (
            f"--- Prediction Results ---\n"
            f"Prediction: {status}\n\n"
            f"Preprocessing Applied:\n{preproc_applied}\n\n"
            f"Feature Extraction (Local Transformers Checkpoints):\n{models_used}\n\n"
            f"Classification: H5-OmniFusion (Local Checkpoint)\n"
            f"Modalities Used: {', '.join(modalities)}"
        )
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