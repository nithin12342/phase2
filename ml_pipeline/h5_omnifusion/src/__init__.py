"""
H5-OmniFusion Package
Multimodal Depression Detection Preprocessing Pipeline

Implements:
- 40 Production Steps
- 59 Research Steps  
- 9 Advanced Innovations

All embeddings output at 768-dimensions for fusion compatibility.
Supports DAIC-WOZ (English) and EATD-Corpus (Mandarin Chinese).
"""

from .config import Config, CFG
from .utils import DEVICE, ensure_768_dim, clear_memory
from .model_loader import ModelLoader, MODEL_LOADER
from .pipeline import H5OmniFusionPipeline, run_pipeline

__version__ = '1.0.0'
__all__ = [
    'Config', 'CFG',
    'DEVICE', 'ensure_768_dim', 'clear_memory',
    'ModelLoader', 'MODEL_LOADER',
    'H5OmniFusionPipeline', 'run_pipeline'
]
