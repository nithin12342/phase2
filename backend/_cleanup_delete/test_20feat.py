import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend')
from hf_client import HFClient
import models

m = models._load_fusion_model()

# Construct explicit quality_inputs forcing the model to ignore missing modalities
# and heavily weight the text and tabular features.
q = {
    'audio_quality': torch.zeros(1),
    'face_confidence': torch.zeros(1),
    'video_motion': torch.zeros(1),
    'tabular_completeness': torch.ones(1), # Real tabular data coming!
    'text_length': torch.ones(1) * 500.0,
}

# 20-D Tabular features mapping (guessed from dvlog_108step_features.py)
# Index 4: sentiment_positive
# Index 5: sentiment_negative
# Index 6: sentiment_neutral
def get_20feat(pos=0.0, neg=0.0, neu=0.0):
    f = torch.zeros(1, 20)
    f[0, 4] = pos
    f[0, 5] = neg
    f[0, 6] = neu
    # Set quality scores in tabular too
    f[0, 8] = 1.0 # text_quality
    f[0, 9] = 0.0 # video_quality
    return f

# Case 1: Happy Tabular + Text
inputs_happy = {
    'audio_features': torch.zeros(1, 768),
    'tabular_features': get_20feat(pos=1.0, neg=0.0, neu=0.0),
    'video_features': torch.zeros(1, 768),
    'face_features': torch.zeros(1, 768),
    'text_features': torch.zeros(1, 768), # Let's see if tabular alone does it
    'quality_inputs': q
}

# Case 2: Sad Tabular + Text
inputs_sad = {
    'audio_features': torch.zeros(1, 768),
    'tabular_features': get_20feat(pos=0.0, neg=1.0, neu=0.0),
    'video_features': torch.zeros(1, 768),
    'face_features': torch.zeros(1, 768),
    'text_features': torch.zeros(1, 768),
    'quality_inputs': q
}

with torch.no_grad():
    res_happy, _ = m(inputs_happy)
    res_sad, _ = m(inputs_sad)
    
print('HAPPY (Tabular 20) PROB:', res_happy['binary_prob'].item())
print('HAPPY GATE WEIGHTS:', res_happy['gate_weights'][0].tolist())
print('HAPPY EXPERT LOGITS:', res_happy['expert_logits'][0].tolist())

print('\nSAD (Tabular 20) PROB:', res_sad['binary_prob'].item())
print('SAD GATE WEIGHTS:', res_sad['gate_weights'][0].tolist())
print('SAD EXPERT LOGITS:', res_sad['expert_logits'][0].tolist())
