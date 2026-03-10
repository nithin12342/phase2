import sys, os, torch
from collections import OrderedDict
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ML = os.path.join(PROJECT_ROOT, "ml_pipeline")
H5_ROOT = os.path.join(ML, "h5_omnifusion")

# Insert h5_omnifusion root BEFORE backend so config resolves correctly
sys.path.insert(0, H5_ROOT)

from config.model_config import H5Config, ComputeTier
from src.models.h5_omnifusion import H5OmniFusion

ckpt_path = os.path.join(H5_ROOT, "checkpoints", "fold4_phase12_latest.pt")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = checkpoint.get("model_state_dict", checkpoint)
sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())

d_model = sd["latent_fusion.latent_init"].shape[1] if "latent_fusion.latent_init" in sd else 256
config = H5Config.from_tier(ComputeTier.MEDIUM)
config.d_model = d_model
config.fusion.n_latents = sd["latent_fusion.latent_init"].shape[0] if "latent_fusion.latent_init" in sd else 16
config.moe.expert_hidden_dim = sd["moe.audio_expert.network.0.weight"].shape[0] if "moe.audio_expert.network.0.weight" in sd else 128
config.face.bidirectional = True
config.face.au_dim = sd["face_encoder.au_proj.weight"].shape[1] if "face_encoder.au_proj.weight" in sd else 35
if "moe.gate.gate_network.0.weight" in sd:
    config.moe.gate_hidden_dim = sd["moe.gate.gate_network.0.weight"].shape[0]

model = H5OmniFusion(config)
msg = model.load_state_dict(sd, strict=False)
model.eval()
print(f"Missing keys: {len(msg.missing_keys)}")
print(f"Unexpected keys: {len(msg.unexpected_keys)}")

# Zeros
inputs = {k: torch.zeros(1, 768) for k in ["text_features", "audio_features", "video_features", "face_features", "tabular_features"]}
with torch.no_grad():
    out, _ = model(inputs)
    bp = out["binary_prob"].item()
    ps = out["phq_score"].item()
    print(f"Zeros: prob={bp:.4f}, phq={ps:.2f}")

# Small random
inputs2 = {k: torch.randn(1, 768) * 0.3 for k in ["text_features", "audio_features", "video_features", "face_features", "tabular_features"]}
with torch.no_grad():
    out2, _ = model(inputs2)
    bp2 = out2["binary_prob"].item()
    ps2 = out2["phq_score"].item()
    print(f"Random: prob={bp2:.4f}, phq={ps2:.2f}")
