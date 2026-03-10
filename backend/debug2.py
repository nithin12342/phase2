import os
import sys
import torch

backend_dir = r"c:\Users\thela\OneDrive\Desktop\phase 2\backend"
sys.path.append(backend_dir)
os.chdir(backend_dir)

import models
m = models._load_fusion_model()

inputs1 = {k: torch.zeros(1, 768) for k in ['audio_features', 'text_features', 'tabular_features', 'video_features', 'face_features']}

inputs2 = {k: torch.randn(1, 768) for k in inputs1.keys()}

with torch.no_grad():
    res1, _ = m(inputs1)
    print("ZEROS PROB:", res1['binary_prob'].item())

    res2, _ = m(inputs2)
    print("RANDOM PROB:", res2['binary_prob'].item())

    # Try 10 random inputs
    print("10 RANDOM TRIALS:")
    for i in range(10):
        inp = {k: torch.randn(1, 768) for k in inputs1.keys()}
        print(f" Trial {i}: {m(inp)[0]['binary_prob'].item():.5f}")
