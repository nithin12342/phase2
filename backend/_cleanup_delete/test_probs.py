import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

backend_dir = r"c:\Users\thela\OneDrive\Desktop\phase 2\backend"
sys.path.append(backend_dir)
os.chdir(backend_dir)

from hf_client import HFClient
import models

m = models._load_fusion_model()
c = HFClient('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')

d = r"c:\Users\thela\OneDrive\Desktop\phase 2\demo_samples"
for folder in os.listdir(d):
    sample_dir = os.path.join(d, folder)
    if not os.path.isdir(sample_dir): continue
    
    try:
        txt = open(os.path.join(sample_dir, 'transcript.txt'), encoding='utf-8').read()
        emb, _ = c.get_text_embedding(txt)
        
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
        
        print(f"{folder:35s}: {prob:.6f}")
    except Exception as e:
        print(f"Error on {folder}: {e}")
