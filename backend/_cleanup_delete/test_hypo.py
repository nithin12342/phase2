import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend')
import models

# Force tabular-only by mocking inputs
def test_tabular_only(text, sentiment_pos, sentiment_neg):
    print(f"\n--- TESTING: {text} ---")
    # Custom call to get_fusion_prediction logic
    # but zeroing out text_features
    inputs = {
        "audio_features": torch.zeros(1, 768),
        "text_features": torch.zeros(1, 768), # ZEROED!
        "video_features": torch.zeros(1, 768),
        "face_features": torch.zeros(1, 768),
        "tabular_features": torch.zeros(1, 20)
    }
    inputs["tabular_features"][0, 4] = sentiment_pos
    inputs["tabular_features"][0, 5] = sentiment_neg
    inputs["tabular_features"][0, 8] = 1.0 # text quality
    
    q_inputs = {
        "audio_quality": torch.zeros(1),
        "text_length": torch.tensor([float(len(text))]),
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
    
    prob = outputs["binary_prob"].item()
    gate = outputs["gate_weights"][0].tolist()
    print(f"PROB: {prob:.4f}")
    print(f"GATE: {gate}")

test_tabular_only("Happy text", 1.0, 0.0)
test_tabular_only("Sad text", 0.0, 1.0)
