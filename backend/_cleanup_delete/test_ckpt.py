import torch
ckpt = torch.load(r'c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\checkpoints\fold4_phase12_latest.pt', map_location='cpu', weights_only=False)
sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
for k in list(sd.keys()):
    print(k, sd[k].shape)
