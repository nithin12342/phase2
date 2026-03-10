import sys
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend')
from hf_client import HFClient

# Load client
c = HFClient('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')

text_happy = "I am incredibly happy and my life is perfect! I feel so much joy everyday."
text_sad = "I feel miserable and depressed everyday, I have no hope, and I want to cry."

inputs = c._text_tokenizer(
    [text_happy, text_sad],
    truncation=True,
    padding=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = c._text_model(**inputs)
    attention = inputs['attention_mask'].unsqueeze(-1)
    embeddings = outputs.last_hidden_state * attention
    arr_h = (embeddings[0].sum(dim=0) / attention[0].sum(dim=0)).cpu().numpy().flatten()
    arr_s = (embeddings[1].sum(dim=0) / attention[1].sum(dim=0)).cpu().numpy().flatten()
    emb_h = arr_h
    emb_s = arr_s

print('Happy shape:', emb_h.shape, 'Sad shape:', emb_s.shape)
print('Distance squared:', ((emb_h - emb_s) ** 2).sum())
print('Diff mean:', (emb_h - emb_s).mean())

# Look at min max
print('Happy minmax:', emb_h.min(), emb_h.max())
print('Sad minmax:', emb_s.min(), emb_s.max())
