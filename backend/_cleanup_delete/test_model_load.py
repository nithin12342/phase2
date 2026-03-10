"""Test model loading in the exact backend environment."""
import sys, os

# Simulate exact backend environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ML_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline")
if not os.path.exists(ML_PIPELINE_DIR):
    ML_PIPELINE_DIR = os.path.join(BASE_DIR, "ml_pipeline")

# Import backend config first (just like models.py does at module level)
sys.path.insert(0, BASE_DIR)
from config import get_settings
settings = get_settings()
print(f"Backend config loaded OK")
print(f"ML_PIPELINE_DIR: {ML_PIPELINE_DIR} (exists: {os.path.exists(ML_PIPELINE_DIR)})")

import torch
from collections import OrderedDict
import importlib.util
import numpy as np

H5_ROOT = os.path.join(ML_PIPELINE_DIR, "h5_omnifusion")
ckpt_path = os.path.join(H5_ROOT, "checkpoints", "fold4_phase12_latest.pt")
print(f"Checkpoint: {ckpt_path} (exists: {os.path.exists(ckpt_path)})")

# importlib-based loading (bypasses config module conflict)
model_config_path = os.path.join(H5_ROOT, "config", "model_config.py")
print(f"model_config.py: {model_config_path} (exists: {os.path.exists(model_config_path)})")

spec = importlib.util.spec_from_file_location("h5_model_config", model_config_path)
h5_cfg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h5_cfg_mod)
H5Config = h5_cfg_mod.H5Config
ComputeTier = h5_cfg_mod.ComputeTier
print("H5Config loaded via importlib OK")

# Import model architecture
if H5_ROOT not in sys.path:
    sys.path.insert(0, H5_ROOT)
from src.models.h5_omnifusion import H5OmniFusion
print("H5OmniFusion imported OK")

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = checkpoint.get("model_state_dict", checkpoint)
sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())

# Auto-detect config
d_model = sd["latent_fusion.latent_init"].shape[1] if "latent_fusion.latent_init" in sd else 256
n_latents = sd["latent_fusion.latent_init"].shape[0] if "latent_fusion.latent_init" in sd else 16
expert_hidden = sd["moe.audio_expert.network.0.weight"].shape[0] if "moe.audio_expert.network.0.weight" in sd else 128
au_dim = sd["face_encoder.au_proj.weight"].shape[1] if "face_encoder.au_proj.weight" in sd else 35

tier = ComputeTier.MEDIUM if d_model > 64 else ComputeTier.MICRO
config = H5Config.from_tier(tier)
config.d_model = d_model
config.fusion.n_latents = n_latents
config.moe.expert_hidden_dim = expert_hidden
config.face.bidirectional = True
config.face.au_dim = au_dim
if "moe.gate.gate_network.0.weight" in sd:
    config.moe.gate_hidden_dim = sd["moe.gate.gate_network.0.weight"].shape[0]

model = H5OmniFusion(config)
msg = model.load_state_dict(sd, strict=False)
model.eval()
print(f"Model loaded! Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

# Test predictions
inputs_zero = {k: torch.zeros(1, 768) for k in ["text_features", "audio_features", "video_features", "face_features", "tabular_features"]}
inputs_rand = {k: torch.randn(1, 768) * 0.5 for k in ["text_features", "audio_features", "video_features", "face_features", "tabular_features"]}

with torch.no_grad():
    out_z, _ = model(inputs_zero)
    out_r, _ = model(inputs_rand)
    print(f"Zeros  -> prob={out_z['binary_prob'].item():.4f}, phq={out_z['phq_score'].item():.2f}")
    print(f"Random -> prob={out_r['binary_prob'].item():.4f}, phq={out_r['phq_score'].item():.2f}")

print("\nAll checks passed!")
