import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend')
import models

def test_spoof(phq):
    print(f"\n--- SPOOF TEST: PHQ={phq} ---")
    inputs = {
        "audio_features": torch.zeros(1, 768),
        "text_features": torch.zeros(1, 768),
        "video_features": torch.zeros(1, 768),
        "face_features": torch.zeros(1, 768),
        "tabular_features": torch.zeros(1, 20)
    }
    # SPOOF AUDIO INDICES (0-3) WITH PHQ!
    val = float(phq) / 24.0
    inputs["tabular_features"][0, 0:4] = val
    inputs["tabular_features"][0, 11:18] = val # Health flags too
    
    q_inputs = {
        "audio_quality": torch.zeros(1),
        "text_length": torch.ones(1) * 500.0,
        "video_motion": torch.zeros(1),
        "face_confidence": torch.zeros(1),
        "tabular_completeness": torch.ones(1)
    }
    
    model = models._load_fusion_model()
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    q_inputs = {k: v.to(next(model.parameters()).device) for k, v in q_inputs.items()}
    inputs["quality_inputs"] = q_inputs
    
    with torch.no_grad():
        outputs, _ = model(inputs)
    
    print(f"PROB: {outputs['binary_prob'].item():.4f}")
    gate = outputs["gate_weights"][0].tolist()
    print(f"GATE: {gate}")

test_spoof(0.0)
test_spoof(24.0)
