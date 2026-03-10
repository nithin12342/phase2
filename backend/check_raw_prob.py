import os
import sys
import torch
import numpy as np

# Add project roots
backend_dir = r"c:\Users\thela\OneDrive\Desktop\phase 2\backend"
sys.path.append(backend_dir)
os.chdir(backend_dir)

import models
from hf_client import HFClient

sample_dir = r"c:\Users\thela\OneDrive\Desktop\phase 2\demo_samples\sample_4_Not_Depressed_PID303"
print(f"Testing sample {sample_dir}")

client = HFClient('dummy_token')

# Load files
with open(os.path.join(sample_dir, 'audio.wav'), 'rb') as f:
    audio_bytes = f.read()

dengan_tabular = {
    'gender': 'F', 'country': 'US', 'occupation': 'none', 'days_indoors': '0', 
    'is_self_employed': 'No', 'self_employed_date': '', 'growing_stress': 'No', 
    'changes_habits': 'No', 'mental_health_history': 'No', 'family_history': 'No', 
    'treatment_sought': 'No', 'mood_swings': 'Low', 'work_interest': 'Yes', 
    'social_weakness': 'No', 'coping_struggles': 'No', 'interview_attended': 'No', 
    'care_options_awareness': 'No'
}

print("Extracting Audio...")
audio_emb, _ = client.get_audio_embedding(audio_bytes)
print("Extracting Tabular...")
tab_emb, _ = models.get_tabular_embedding(dengan_tabular)

embeddings = {
    'audio': audio_emb,
    'tabular': tab_emb,
    'text': np.zeros(768, dtype=np.float32),
    'video': np.zeros(768, dtype=np.float32),
    'image': np.zeros(768, dtype=np.float32),
}

# Load model locally
_fusion_model = models._load_fusion_model()
if _fusion_model is None:
    print("FAILED TO LOAD MODEL")
    exit(1)

inputs = {}
for k, v in embeddings.items():
    key = "face_features" if k == "image" else f"{k}_features"
    inputs[key] = torch.from_numpy(v).float().unsqueeze(0)

print("\n--- Running Inference ---")
with torch.no_grad():
    outputs, _ = _fusion_model(inputs)
    prob = float(torch.sigmoid(outputs["logits"])[0][0].item() if "logits" in outputs else outputs["binary_prob"].item())
    phq = float(outputs["phq_score"].item())

print(f"RAW PROBABILITY: {prob:.8f}")
print(f"THRESHOLD (0.85) EXCEEDED?: {prob >= 0.85}")
print(f"PREDICTED PHQ Score: {phq:.2f}")

# Check expert weights to see if audio is overwhelming everything
if "expert_weights" in outputs:
    print("\nExpert Weights distribution:")
    weights = outputs["expert_weights"][0].cpu().numpy()
    for i, w in enumerate(weights):
        print(f"Expert {i}: {w:.4f}")
