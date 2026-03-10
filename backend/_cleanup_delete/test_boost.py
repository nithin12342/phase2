import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend')
import models

def test_boost(text, pos, neg):
    print(f"\n--- BOOST TEST: {text} ---")
    inputs = {
        "audio_features": torch.zeros(1, 768),
        "text_features": torch.zeros(1, 768),
        "video_features": torch.zeros(1, 768),
        "face_features": torch.zeros(1, 768),
        "tabular_features": torch.zeros(1, 20)
    }
    # BOOSTED SIGNAL!
    inputs["tabular_features"][0, 4] = pos * 20.0 
    inputs["tabular_features"][0, 5] = neg * 20.0
    inputs["tabular_features"][0, 8] = 1.0
    
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

test_boost("Happy (20x)", 1.0, 0.0)
test_boost("Sad (20x)", 0.0, 1.0)
