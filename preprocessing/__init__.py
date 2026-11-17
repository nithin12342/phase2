"""H5-OmniFusion preprocessing pipeline (binary depression prediction).

Processes D-Vlog YouTube videos through download, decode, ASR, GPU encoders,
and the Rust feature kernel, then assembles the 200-feature vector with a
binary depression label. Training lives in a separate area.
"""
__version__ = "0.1.0"