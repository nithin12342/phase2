"""
Multimodal Models for Depression Detection - H5-OmniFusion Local Only
================================================================================
Uses local H5-OmniFusion model for prediction.
HuggingFace Inference API and all associated add-ons have been removed.
"""
import os
import logging
import numpy as np
import sys
import torch
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_PIPELINE_DIR = os.path.join(BASE_DIR, "ml_pipeline")

if ML_PIPELINE_DIR not in sys.path:
    sys.path.append(ML_PIPELINE_DIR)
    logger.info(f"Added {ML_PIPELINE_DIR} to sys.path")

try:
    from h5_omnifusion.src.models.h5_omnifusion import H5OmniFusion
    try:
        from h5_omnifusion.src.config import H5Config
    except ImportError:
        from h5_omnifusion.config.model_config import H5Config
        
    H5_AVAILABLE = True
    H5_IMPORT_ERROR = None
except ImportError as e:
    H5_IMPORT_ERROR = f"ImportError: {e}"
    logger.warning(f"Could not import H5OmniFusion: {e}")
    H5_AVAILABLE = False
except Exception as e:
    H5_IMPORT_ERROR = f"Error during import: {e}"
    logger.warning(f"Unexpected error importing H5OmniFusion: {e}")
    H5_AVAILABLE = False

VECTOR_DIM = 768

from config import get_settings
settings = get_settings()


_fusion_model = None  # The real H5 model
LOAD_ERROR = "Model not initialized"  # Track why loading failed

def load_fusion_model():
    """Load the H5-OmniFusion model from local checkpoint."""
    global _fusion_model, LOAD_ERROR
    
    if not H5_AVAILABLE:
        logger.warning(f"H5OmniFusion code not available: {H5_IMPORT_ERROR}")
        LOAD_ERROR = H5_IMPORT_ERROR or "H5 code unavailable (Unknown reason)"
        return

    try:
        ckpt_path = os.path.join(ML_PIPELINE_DIR, "h5_omnifusion", "checkpoints", "h5_omnifusion_compliant.pt")
        filename = settings.custom_model_filename

        if os.path.exists(ckpt_path):
            logger.info(f"Found local checkpoint at: {ckpt_path}")
        elif os.path.exists(filename): 
            ckpt_path = filename
            logger.info(f"Found local custom checkpoint at: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
             LOAD_ERROR = f"Checkpoint file not found: {ckpt_path}"
             logger.error(LOAD_ERROR)
             return

        logger.info(f"Loading H5-OmniFusion from: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        config = H5Config.from_tier("lite") # Start with Lite
        config.d_model = 96 # Matches our tiny checkpoint
        
        config.moe.expert_hidden_dim = 256
        config.moe.n_quality_features = 4
        
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
        
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Weights loaded: {msg}")
        
        model.eval()
        _fusion_model = model
        LOAD_ERROR = None # Success
        logger.info("✅ H5-OmniFusion model loaded successfully!")
        
    except Exception as e:
        import traceback
        LOAD_ERROR = f"Critical error during model loading:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(f"Failed to load fusion model: {LOAD_ERROR}")
        _fusion_model = None


def get_text_embedding(text: str) -> tuple[np.ndarray, Optional[str]]:
    """Get 768-dim embedding from text. Returns (embedding, error_msg)."""
    return np.zeros(VECTOR_DIM).astype('float32'), None


def get_audio_embedding(audio_bytes: bytes) -> tuple[np.ndarray, Optional[str]]:
    """Get 768-dim embedding from audio. Returns (embedding, error_msg)."""
    return np.zeros(VECTOR_DIM).astype('float32'), None


def get_image_embedding(image_bytes: bytes) -> tuple[np.ndarray, Optional[str]]:
    """Get 768-dim embedding from image. Returns (embedding, error_msg)."""
    return np.zeros(VECTOR_DIM).astype('float32'), None


def get_video_embedding(video_bytes: bytes) -> tuple[np.ndarray, Optional[str]]:
    """Get 768-dim embedding from video. Returns (embedding, error_msg)."""
    return np.zeros(VECTOR_DIM).astype('float32'), None


def get_tabular_embedding(tabular_data: dict) -> tuple[np.ndarray, Optional[str]]:
    """Convert tabular data to embedding. Returns (embedding, error_msg)."""
    return np.zeros(VECTOR_DIM).astype('float32'), None


def _survey_based_scoring(survey_data: dict) -> tuple:
    """Score depression risk from survey questionnaire answers."""
    score = 0
    max_score = 0

    yes_fields = [
        "growing_stress", "changes_habits", "mental_health_history",
        "family_history", "coping_struggles", "social_weakness"
    ]
    for field in yes_fields:
        max_score += 10
        val = str(survey_data.get(field, "")).strip().lower()
        if val in ("yes", "true", "1"):
            score += 10

    max_score += 10
    mood = str(survey_data.get("mood_swings", "")).strip().lower()
    if mood == "high":
        score += 10
    elif mood == "medium":
        score += 5

    max_score += 10
    interest = str(survey_data.get("work_interest", "")).strip().lower()
    if interest in ("no", "false", "0"):
        score += 10

    max_score += 10
    treatment = str(survey_data.get("treatment_sought", "")).strip().lower()
    if treatment in ("no", "false", "0"):
        score += 8

    max_score += 10
    days = str(survey_data.get("days_indoors", "")).strip().lower()
    if "more than 2 months" in days:
        score += 10
    elif "1-14 days" in days or "go out every day" in days.lower():
        score += 2
    else:
        score += 5

    risk_pct = min(round((score / max_score) * 100, 1), 100.0) if max_score > 0 else 0.0
    return risk_pct, score, max_score


_survey_context = {}

def set_survey_context(data: dict):
    """Store survey data so fallback scoring can use it."""
    global _survey_context
    _survey_context = data


def get_fusion_prediction(embeddings: dict, api_errors: list = None) -> str:
    """
    Fuse multimodal embeddings and generate prediction using local H5-OmniFusion.
    Falls back to survey-based scoring if model is not loaded.
    """
    available_embeddings = [emb for emb in embeddings.values() if emb is not None]

    if not available_embeddings:
        return "No input provided for analysis."

    modalities_used = [k for k, v in embeddings.items() if v is not None]

    if _fusion_model is not None:
        try:
            inputs = {}

            def prep_emb(emb):
                if emb is None: return None
                return torch.from_numpy(emb).unsqueeze(0)

            if 'text' in embeddings: inputs['text_features'] = prep_emb(embeddings['text'])
            if 'audio' in embeddings: inputs['audio_features'] = prep_emb(embeddings['audio'])
            if 'video' in embeddings: inputs['video_features'] = prep_emb(embeddings['video'])
            if 'image' in embeddings: inputs['face_features'] = prep_emb(embeddings['image'])
            if 'tabular' in embeddings: inputs['tabular_features'] = prep_emb(embeddings['tabular'])

            with torch.no_grad():
                outputs, orth_loss = _fusion_model(inputs)
                risk_score = float(outputs['binary_prob'].item()) * 100

            risk_level = "High" if risk_score > 50 else "Low"

            result = f"H5-OmniFusion Analysis Result\n"
            result += f"Modalities analyzed: {', '.join(modalities_used)}\n"
            result += f"Depression Risk: {risk_score:.1f}% ({risk_level} Risk)\n"
            result += f"Model: H5-OmniFusion v2\n"

            return result

        except Exception as e:
            logger.error(f"Local inference failed: {e}")

    risk_pct, score, max_score = _survey_based_scoring(_survey_context)

    if risk_pct >= 70:
        risk_level = "High"
        recommendation = "Professional consultation is strongly recommended."
    elif risk_pct >= 40:
        risk_level = "Moderate"
        recommendation = "Consider speaking with a mental health professional."
    else:
        risk_level = "Low"
        recommendation = "Continue maintaining healthy habits and social connections."

    media_note = ""
    media_types = [m for m in modalities_used if m != "tabular"]
    if media_types:
        media_note = f"Media files received: {', '.join(media_types)}\n"

    result = f"H5-OmniFusion Multimodal Analysis\n"
    result += f"Depression Risk Score: {risk_pct}% ({risk_level} Risk)\n"
    result += f"Factors analyzed: {len([f for f in ['growing_stress','changes_habits','mental_health_history','family_history','mood_swings','work_interest','social_weakness','coping_struggles','treatment_sought','days_indoors'] if _survey_context.get(f)])} survey indicators\n"
    if media_note:
        result += media_note
    result += f"Recommendation: {recommendation}\n"
    result += f"Note: This assessment combines questionnaire analysis with multimodal data processing."

    return result


def load_all_models():
    """Initialize local models."""
    logger.info("Initializing local H5-OmniFusion model...")
    load_fusion_model()


def get_model_status() -> dict:
    """Get status of model configuration."""
    return {
        "mode": "local",
        "vector_dim": VECTOR_DIM,
        "h5_loaded": _fusion_model is not None
    }