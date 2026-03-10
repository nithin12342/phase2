import os
import sys
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

backend_dir = r"c:\Users\thela\OneDrive\Desktop\phase 2\backend"
sys.path.append(backend_dir)
os.chdir(backend_dir)

import models
print("Loading model...")
m = models._load_fusion_model()
if m is None:
    print("Model failed to load.")
    exit(1)

# Inspect a few random layers to see if they are initialized or collapsed
print("\n--- Linear layer 'latent_fusion.latent_init' ---")
if hasattr(m, 'latent_fusion') and hasattr(m.latent_fusion, 'latent_init'):
    print(m.latent_fusion.latent_init.sum().item())
else:
    print("Not found")
    
print("\n--- Gate Network 'moe.gate.gate_network.0.weight' ---")
if hasattr(m, 'moe') and hasattr(m.moe, 'gate') and hasattr(m.moe.gate, 'gate_network'):
    print(m.moe.gate.gate_network[0].weight.sum().item())
else:
    print("Not found")

print("\n--- Binary classification head ---")
print(m.classifier[0].weight.sum().item())

# Create a mock input to trace what it's doing
inputs = {
    'audio_features': torch.randn(1, 768),
    'text_features': torch.randn(1, 768),
    'tabular_features': torch.randn(1, 768),
    'video_features': torch.randn(1, 768),
    'face_features': torch.randn(1, 768),
}

with torch.no_grad():
    outputs, _ = m(inputs)
    print("\n--- Outputs ---")
    print("Logits:", outputs.get("logits", "N/A"))
    print("Binary Prob:", outputs.get("binary_prob", "N/A"))
    print("PHQ:", outputs.get("phq_score", "N/A"))
