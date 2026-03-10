import numpy as np
import models
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath('models.py'))
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), "ml_pipeline"))
models._import_fusion()

def predict_with_stats(name, emb_dict):
    res = models.get_fusion_prediction(emb_dict)
    is_dep = "Depression" in res and "Not Depressed" not in res
    print(f"{name} -> Depressed: {is_dep}")

# Test different scales
for scale in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    embs = {k: (np.random.randn(768) * scale).astype('float32') for k in ["text", "audio", "video", "image", "tabular"]}
    predict_with_stats(f"Gaussian (std={scale})", embs)

for val in [-1.0, -0.1, 0.0, 0.1, 1.0]:
    embs = {k: (np.ones(768) * val).astype('float32') for k in ["text", "audio", "video", "image", "tabular"]}
    predict_with_stats(f"Constant ({val})", embs)
