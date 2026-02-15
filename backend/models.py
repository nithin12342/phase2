"""
Multimodal Models for Depression Detection - H5-OmniFusion Local (Lazy Loading)
================================================================================
Implements "Option A" Deployment Strategy:
- Lazy-loads models on demand (Wav2Vec2, MentalRoBERTa, etc.)
- Unloads them immediately to free RAM
- Uses full 108-step H5OmniFusionPipeline for feature extraction
- Runs on CPU with ~3GB peak RAM
"""
import os
import logging
import numpy as np
import sys
import torch
import tempfile
import shutil
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_PIPELINE_DIR = os.path.join(BASE_DIR, "ml_pipeline")

if ML_PIPELINE_DIR not in sys.path:
    sys.path.append(ML_PIPELINE_DIR)
    logger.info(f"Added {ML_PIPELINE_DIR} to sys.path")

from config import get_settings
settings = get_settings()

# Global state
_pipeline = None
_fusion_model = None
H5_AVAILABLE = False
LOAD_ERROR = "Model not initialized"

# Try importing the pipeline
try:
    from h5_omnifusion.src.pipeline import H5OmniFusionPipeline
    from h5_omnifusion.src.model_loader import MODEL_LOADER
    from h5_omnifusion.src.config import H5Config
    from h5_omnifusion.src.models.h5_omnifusion import H5OmniFusion
    
    # Configure ModelLoader to look in local directory
    LOCAL_MODELS_DIR = os.path.join(ML_PIPELINE_DIR, "h5_omnifusion", "pretrained_models")
    if os.path.exists(LOCAL_MODELS_DIR):
        MODEL_LOADER.pretrained_path = LOCAL_MODELS_DIR
        logger.info(f"Configured ModelLoader path: {LOCAL_MODELS_DIR}")
    else:
        logger.warning(f"Local models dir not found: {LOCAL_MODELS_DIR}")

    H5_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import H5 pipeline: {e}")
    H5_AVAILABLE = False
    LOAD_ERROR = str(e)


def load_all_models():
    """
    Initialize the pipeline and fusion model.
    Does NOT load backbone models (Wav2Vec2, etc.) - they are lazy-loaded.
    """
    global _pipeline, _fusion_model, LOAD_ERROR

    if not H5_AVAILABLE:
        return

    try:
        # 1. Initialize Pipeline (lightweight, just setup)
        _pipeline = H5OmniFusionPipeline(device='cpu')
        
        # 2. Load Fusion Model (keep in memory, it's small ~50MB)
        load_fusion_model()
        
        logger.info("✅ H5-OmniFusion Pipeline initialized (Lazy Loading Mode)")

    except Exception as e:
        import traceback
        LOAD_ERROR = f"Init failed: {e}\n{traceback.format_exc()}"
        logger.error(LOAD_ERROR)


def load_fusion_model():
    """Load the trained H5 fusion checkpoint."""
    global _fusion_model, LOAD_ERROR
    
    ckpt_path = os.path.join(ML_PIPELINE_DIR, "h5_omnifusion", "checkpoints", "h5_omnifusion_compliant.pt")
    if settings.custom_model_filename and os.path.exists(settings.custom_model_filename):
        ckpt_path = settings.custom_model_filename

    if not os.path.exists(ckpt_path):
        logger.warning(f"Fusion checkpoint not found at {ckpt_path}. Using rules-based fallback.")
        return

    try:
        logger.info(f"Loading Fusion Model from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # Initialize model with 'lite' config (matches tiny 96-dim checkpoint)
        config = H5Config.from_tier("lite")
        config.d_model = 96
        
        # Exact overrides from previous checkpoint version to ensure compatibility
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
        
        # Create model and load weights
        model = H5OmniFusion(config)
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Fusion weights loaded: {msg}")
        
        model.eval()
        _fusion_model = model
        LOAD_ERROR = None
        
    except Exception as e:
        logger.error(f"Failed to load fusion model: {e}")
        # Don't crash, just let it be None (fallback will run)


def get_text_embedding(text: str) -> tuple[np.ndarray, Optional[str]]:
    """
    Lazy-load MentalRoBERTa, get embedding, unload.
    """
    if not H5_AVAILABLE or _pipeline is None:
        return np.zeros(768).astype('float32'), "Pipeline not loaded"

    tmp_path = None
    try:
        # Create temp file for pipeline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        # Run pipeline step
        logger.info("Lazy-loading Text Encoder...")
        results = _pipeline._process_text(tmp_path)
        
        if results.get('success'):
            return results['text_embedding'].astype('float32'), None
        else:
            return np.zeros(768).astype('float32'), results.get('error', 'Unknown error')

    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        return np.zeros(768).astype('float32'), str(e)
        
    finally:
        # Cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        # Always unload to save RAM
        try:
            MODEL_LOADER.unload_model('text_english')
            MODEL_LOADER.unload_model('text_chinese')
        except:
            pass


def get_audio_embedding(audio_bytes: bytes) -> tuple[np.ndarray, Optional[str]]:
    """
    Lazy-load Wav2Vec2 + OpenSMILE, get embedding, unload.
    """
    if not H5_AVAILABLE or _pipeline is None:
        return np.zeros(768).astype('float32'), "Pipeline not loaded"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        logger.info("Lazy-loading Audio Models (Wav2Vec2 + OpenSMILE)...")
        results = _pipeline._process_audio(tmp_path)
        
        if results.get('success'):
            return results['wav2vec2_embedding'].astype('float32'), None
        else:
            return np.zeros(768).astype('float32'), results.get('error')

    except Exception as e:
        logger.error(f"Audio embedding error: {e}")
        return np.zeros(768).astype('float32'), str(e)
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        try:
            MODEL_LOADER.unload_model('wav2vec2')
            MODEL_LOADER.unload_model('opensmile')
        except:
            pass


def get_video_embedding(video_bytes: bytes) -> tuple[np.ndarray, Optional[str]]:
    """
    Lazy-load VideoMAE, get embedding, unload.
    """
    if not H5_AVAILABLE or _pipeline is None:
        return np.zeros(768).astype('float32'), "Pipeline not loaded"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
            
        logger.info("Lazy-loading VideoMAE...")
        results = _pipeline._process_video(tmp_path)
        
        if results.get('success'):
            return results['video_embedding'].astype('float32'), None
        else:
            return np.zeros(768).astype('float32'), results.get('error')

    except Exception as e:
        logger.error(f"Video embedding error: {e}")
        return np.zeros(768).astype('float32'), str(e)
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        try:
            MODEL_LOADER.unload_model('videomae')
        except:
            pass


def get_image_embedding(image_bytes: bytes) -> tuple[np.ndarray, Optional[str]]:
    """
    Uses Face pipeline (DinoV2 + POSTER). Lazy-load, unload.
    Treats image as 1-frame video for consistency.
    """
    if not H5_AVAILABLE or _pipeline is None:
        return np.zeros(768).astype('float32'), "Pipeline not loaded"

    tmp_path = None
    try:
        # Save as .jpg for OpenCV compatibility
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
            
        logger.info("Lazy-loading Face Encoder (DinoV2)...")
        # Reuse _process_face which handles extraction
        results = _pipeline._process_face(tmp_path)
        
        if results.get('success'):
            return results['face_embedding'].astype('float32'), None
        else:
            return np.zeros(768).astype('float32'), results.get('error')

    except Exception as e:
        logger.error(f"Image/Face embedding error: {e}")
        return np.zeros(768).astype('float32'), str(e)
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        try:
            MODEL_LOADER.unload_model('face_encoder') # DinoV2
            # POSTER V2 might stay loaded if managed internally by FaceFeatureExtractor
        except:
            pass


def get_tabular_embedding(tabular_data: dict) -> tuple[np.ndarray, Optional[str]]:
    """
    Process tabular data. No heavy model to load here.
    """
    if not H5_AVAILABLE or _pipeline is None:
         return np.zeros(768).astype('float32'), "Pipeline not loaded"
         
    # Tabular usually doesn't need lazy unloading
    # For now return placeholder as pipeline specific integration for live tabular is TBD
    return np.zeros(768).astype('float32'), "Tabular Model not connected"


# -----------------------------------------------------------------------------
# Fusion & Fallback
# -----------------------------------------------------------------------------

def get_fusion_prediction(embeddings: dict, api_errors: list = None) -> str:
    """
    Run fusion inference.
    """
    modalities_used = [k for k, v in embeddings.items() if v is not None]
    if not modalities_used:
        return "No input provided."

    # 1. Try H5-OmniFusion Inference
    if _fusion_model is not None:
        try:
            inputs = {}
            for k, v in embeddings.items():
                if v is not None and k in ['text', 'audio', 'video', 'image', 'tabular']:
                    # Map 'image' -> 'face' for model
                    model_key = 'face_features' if k == 'image' else f'{k}_features'
                    inputs[model_key] = torch.from_numpy(v).float().unsqueeze(0) # (1, 768)

            with torch.no_grad():
                outputs, _ = _fusion_model(inputs)
                prob = float(outputs['binary_prob'].item())
                risk_score = prob * 100
                
            risk_level = "High" if risk_score > 50 else "Low"
            
            return (
                f"H5-OmniFusion Analysis\n"
                f"Modalities: {', '.join(modalities_used)}\n"
                f"Depression Risk: {risk_score:.1f}% ({risk_level})\n"
                f"Model: H5-OmniFusion v2 (Lazy-Loaded)"
            )

        except Exception as e:
            logger.error(f"Fusion inference failed: {e}")
    
    # 2. Fallback to Survey Scoring
    return _survey_fallback_result(modalities_used)


_survey_context = {}

def set_survey_context(data: dict):
    global _survey_context
    _survey_context = data

def _survey_fallback_result(modalities: list) -> str:
    """Generate result string from survey context (Rule-based)."""
    score = 0
    max_score = 0
    
    # Simple scoring logic
    yes_keys = ["growing_stress", "changes_habits", "mental_health_history", 
                "family_history", "coping_struggles", "social_weakness"]
    
    for k in yes_keys:
        max_score += 10
        if str(_survey_context.get(k, "")).lower() in ("yes", "true", "1"):
            score += 10
            
    # Calculate percentage
    risk_pct = (score / max_score * 100) if max_score > 0 else 0
    risk_level = "High" if risk_pct > 50 else "Low"
    
    return (
        f"Analysis Result (Survey-Based)\n"
        f"Modalities uploaded: {', '.join(modalities)}\n"
        f"Depression Risk: {risk_pct:.1f}% ({risk_level})\n"
        f"Note: Deep Learning model unavailable, using questionnaire analysis."
    )

def get_model_status() -> dict:
    return {
        "mode": "lazy-loading",
        "h5_available": H5_AVAILABLE,
        "fusion_loaded": _fusion_model is not None
    }