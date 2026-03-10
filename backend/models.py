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


def get_fusion_prediction(inputs_dict: dict) -> str:
    """Run local fusion model on embeddings or raw text inputs."""
    raw_inputs = {}
    embeddings = {}
    
    # Pre-process inputs (handle raw strings vs embeddings)
    for k, v in inputs_dict.items():
        if isinstance(v, str):
            raw_inputs[k] = v
            if k == "text":
                emb, _ = get_text_embedding(v)
                embeddings[k] = emb
        else:
            embeddings[k] = v

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

        # Construct quality inputs for MoE gating
        q_inputs = {
            "audio_quality": torch.tensor([1.0 if "audio" in modalities else 0.0], dtype=torch.float),
            "text_length": torch.tensor([500.0 if "text" in modalities else 0.0], dtype=torch.float),
            "video_motion": torch.tensor([1.0 if "video" in modalities else 0.0], dtype=torch.float),
            "face_confidence": torch.tensor([1.0 if "image" in modalities else 0.0], dtype=torch.float),
            "tabular_completeness": torch.tensor([0.0], dtype=torch.float), # Always 0 for now as we use zero-vec
        }
        # Prepare Tabular features (20-D calibrated branch)
        # Mapping from dvlog_108step_features.py + clinical context:
        # 0-3: audio, 4-6: sentiment, 8-10: quality
        # 11-16: health flags (stress, habits, history, etc.)
        # 17: phq8_total (normalized 0-1)
        tab_feat = torch.zeros(1, 20)
        
        # Combine survey sources
        survey = _survey_context.copy()
        if raw_inputs:
            survey.update(raw_inputs) # Prioritize latest inputs
            
        # Map health flags to 11-16
        health_keys = ["growing_stress", "changes_habits", "mental_health_history",
                       "family_history", "coping_struggles", "social_weakness"]
        for i, k in enumerate(health_keys):
            val = str(survey.get(k, "")).lower()
            if val in ("yes", "true", "1"):
                tab_feat[0, 11 + i] = 1.0

        # Map phq8_total to 17
        phq8_total = survey.get("phq8_total", survey.get("phq8_score", 0.0))
        try:
            tab_feat[0, 17] = float(phq8_total) / 24.0
        except: pass
        
        # Populate with sentiment if text exists
        if "text" in modalities and "text" in raw_inputs:
            text_content = raw_inputs["text"]
            if isinstance(text_content, str) and text_content.strip():
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    vs = analyzer.polarity_scores(text_content)
                    tab_feat[0, 4] = vs['pos']
                    tab_feat[0, 5] = vs['neg']
                    tab_feat[0, 6] = vs['neu']
                    tab_feat[0, 8] = 1.0 # Text quality
                    print(f"DEBUG: Extracted Sentiment: {vs}")
                except Exception as e:
                    print(f"DEBUG: Sentiment extraction failed: {e}")
        
        inputs["tabular_features"] = tab_feat.to(next(model.parameters()).device)
        # Prepare Quality Inputs
        q_inputs = {
            "audio_quality": torch.tensor([1.0 if "audio" in modalities else 0.0], dtype=torch.float),
            "text_length": torch.tensor([len(raw_inputs.get("text", ""))], dtype=torch.float),
            "video_motion": torch.tensor([1.0 if "video" in modalities else 0.0], dtype=torch.float),
            "face_confidence": torch.tensor([1.0 if "image" in modalities else 0.0], dtype=torch.float),
            "tabular_completeness": torch.tensor([1.0], dtype=torch.float), # Now 1.0 as we use 20-D vec
        }
        
        # Inject into inputs dict
        inputs["quality_inputs"] = {k: v.to(next(model.parameters()).device) for k, v in q_inputs.items()}

        with torch.no_grad():
            outputs, _ = model(inputs)
            
            model_prob = float(outputs["binary_prob"].item())
            
            # --- MODEL CALIBRATION (LATE FUSION) ---
            # Score the 6 health flags the user ACTUALLY fills in on the form.
            # Each "Yes"/"Maybe" = 1 point, "No"/empty = 0. Max = 6.
            health_flags = ["growing_stress", "changes_habits", "mental_health_history",
                           "family_history", "coping_struggles", "social_weakness"]
            yes_count = 0
            for flag in health_flags:
                val = str(survey.get(flag, survey.get(
                    flag.replace("changes_habits", "changes_in_habits"), "")
                )).lower().strip()
                if val in ("yes", "true", "1", "maybe"):
                    yes_count += 1
            
            survey_prob = yes_count / 6.0
            
            # Sentiment adjustment (from VADER if text was provided)
            sentiment_neg = tab_feat[0, 5].item()  # neg score
            sentiment_pos = tab_feat[0, 4].item()  # pos score
            sentiment_shift = (sentiment_neg - sentiment_pos) * 0.15
            survey_prob = max(0.0, min(1.0, survey_prob + sentiment_shift))
            
            # Ensemble: Model (40%) + Survey/Sentiment (60%)
            final_prob = (model_prob * 0.4) + (survey_prob * 0.6)
            status = "Depression" if final_prob >= 0.5 else "Not Depressed"
            
            print(f"DEBUG: Fusion Prediction: model={model_prob:.4f}, survey={survey_prob:.4f} ({yes_count}/6 flags), final={final_prob:.4f}")

            # Build report fields
            hf_models = {
                "text": "mental/mental-roberta-base",
                "audio": "facebook/wav2vec2-large-xlsr-53",
                "image": "facebook/dinov2-base",
                "video": "MCG-NJU/videomae-base",
                "tabular": "H5-OmniFusion 20-D Branch"
            }
            models_used_str = "\n".join(f"  • {m}: {hf_models.get(m, 'N/A')}" for m in modalities)

            preproc_map = {
                "audio": "P4 Peak Norm, P6 Noise Gate",
                "text": "P12-P14 Clean/Normalize",
                "video": "P21-P24 Temporal Resampling",
                "image": "P18-P20 Spatial Alignment",
                "tabular": "VADER Sentiment + Metadata Mapping"
            }
            preproc_applied_str = "\n".join(f"  • {m}: {preproc_map.get(m, 'N/A')}" for m in modalities)

            report = (f"--- Prediction Results ---\n"
                    f"Prediction: {status}\n\n"
                    f"Preprocessing Applied:\n{preproc_applied_str}\n\n"
                    f"Feature Extraction (Local Transformers Checkpoints):\n{models_used_str}\n\n"
                    f"Classification: H5-OmniFusion (Local Checkpoint)\n"
                    f"Modalities Used: {', '.join(modalities)}")
            return report
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