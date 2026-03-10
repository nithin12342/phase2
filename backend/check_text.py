import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)

backend_dir = r"c:\Users\thela\OneDrive\Desktop\phase 2\backend"
sys.path.append(backend_dir)
os.chdir(backend_dir)

from hf_client import HFClient
import models

m = models._load_fusion_model()
if m is None:
    print("Failed to load model")
    exit(1)

token = os.environ.get('HUGGINGFACE_TOKEN', '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')
c = HFClient(token)

print("Extracting text embedding for a healthy statement...")
emb, err = c.get_text_embedding('I feel very happy today because I got a new job. Life is great!')
if err:
    print("Error extracting text:", err)

inputs = {
    'audio_features': torch.zeros(1, 768),
    'tabular_features': torch.zeros(1, 768),
    'video_features': torch.zeros(1, 768),
    'face_features': torch.zeros(1, 768),
    'text_features': torch.from_numpy(emb).unsqueeze(0)
}

with torch.no_grad():
    res, _ = m(inputs)
    prob = res['binary_prob'].item()
    print("PROBABILITY WITH TEXT:", prob)
