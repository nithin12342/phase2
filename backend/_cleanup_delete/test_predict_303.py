import json
import numpy as np
import models
import os
import sys

# Ensure ML_PIPELINE is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_PIPELINE_DIR = os.path.join(os.path.dirname(BASE_DIR), "ml_pipeline")
if ML_PIPELINE_DIR not in sys.path:
    sys.path.append(ML_PIPELINE_DIR)

models._import_fusion()
print("Fusion Model Available:", models._FUSION_READY)

embeddings = {}
# Generate random embeddings to see baseline behavior
for k in ["text", "audio", "video", "image", "tabular"]:
    embeddings[k] = np.random.randn(768).astype('float32')

print("\n=== Random Embeddings ===")
res = models.get_fusion_prediction(embeddings)
print("Result Contains Depression?:", "Depression" in res and "Not Depressed" not in res)

print("\n=== Zero Embeddings ===")
zero_embeddings = {k: np.zeros(768, dtype=np.float32) for k in ["text", "audio", "video", "image", "tabular"]}
res_zero = models.get_fusion_prediction(zero_embeddings)
print("Result Contains Depression?:", "Depression" in res_zero and "Not Depressed" not in res_zero)
