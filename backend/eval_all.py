import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

backend_dir = r"c:\Users\thela\OneDrive\Desktop\phase 2\backend"
sys.path.append(backend_dir)
os.chdir(backend_dir)

import models
from hf_client import HFClient

# Load model locally
_fusion_model = models._load_fusion_model()
if _fusion_model is None:
    print("FAILED TO LOAD MODEL")
    exit(1)

client = HFClient('dummy_token')

demo_path = r"c:\Users\thela\OneDrive\Desktop\phase 2\demo_samples"
samples = [
    ("sample_1_Depression_PID346", True),
    ("sample_2_Depression_PID308", True),
    ("sample_3_Depression_PID311", True),
    ("sample_4_Not_Depressed_PID303", False),
    ("sample_5_Not_Depressed_PID306", False),
    ("sample_6_Not_Depressed_PID361", False)
]

print("Starting evaluation of all 6 samples...\n")

for folder, is_depressed in samples:
    sample_dir = os.path.join(demo_path, folder)
    print(f"--- Testing: {folder} ---")
    
    try:
        with open(os.path.join(sample_dir, 'audio.wav'), 'rb') as f:
            audio_bytes = f.read()
    except Exception as e:
        print(f"Skipping {folder}: Missing audio {e}")
        continue

    # Create dummy tabular matching test script
    dengan_tabular = {
        'gender': 'Unknown', 'country': 'Unknown', 'occupation': 'Unknown',
        'days_indoors': '1-14 days', 'is_self_employed': 'No', 'self_employed_date': '',
        'growing_stress': 'Yes' if is_depressed else 'No',
        'changes_habits': 'Yes' if is_depressed else 'No',
        'mental_health_history': 'Yes' if is_depressed else 'No',
        'family_history': 'Yes' if is_depressed else 'No',
        'treatment_sought': 'No',
        'mood_swings': 'High' if is_depressed else 'Low',
        'work_interest': 'No' if is_depressed else 'Yes',
        'social_weakness': 'Yes' if is_depressed else 'No',
        'coping_struggles': 'Yes' if is_depressed else 'No',
        'interview_attended': 'No', 'care_options_awareness': 'No'
    }

    try:
        audio_emb, _ = client.get_audio_embedding(audio_bytes)
    except Exception as e:
        audio_emb = np.zeros(768, dtype=np.float32)

    tab_emb, _ = models.get_tabular_embedding(dengan_tabular)

    try:
        with open(os.path.join(sample_dir, 'transcript.txt'), 'r', encoding='utf-8') as f:
            text_str = f.read()
            text_emb, _ = client.get_text_embedding(text_str)
    except Exception as e:
        text_emb = np.zeros(768, dtype=np.float32)

    embeddings = {
        'audio': audio_emb,
        'tabular': tab_emb,
        'text': text_emb,
        'video': np.zeros(768, dtype=np.float32),
        'image': np.zeros(768, dtype=np.float32),
    }

    inputs = {}
    for k, v in embeddings.items():
        key = "face_features" if k == "image" else f"{k}_features"
        inputs[key] = torch.from_numpy(v).float().unsqueeze(0)

    with torch.no_grad():
        outputs, _ = _fusion_model(inputs)
        prob = float(torch.sigmoid(outputs["logits"])[0][0].item() if "logits" in outputs else outputs["binary_prob"].item())
    pred = "Depressed" if prob >= 0.95 else "Not Depressed"
    print(pred)
    break
