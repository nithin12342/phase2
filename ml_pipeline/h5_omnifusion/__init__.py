"""H⁵-OmniFusion: Multimodal Depression Detection System."""

__version__ = "1.0.0"
__author__ = "H5-OmniFusion Team"

from .src.models.h5_omnifusion import H5OmniFusion
from .config.model_config import H5Config, ComputeTier
from .config.training_config import TrainingConfig

__all__ = ["H5OmniFusion", "H5Config", "ComputeTier", "TrainingConfig"]
