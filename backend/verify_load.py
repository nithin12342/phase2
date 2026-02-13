import sys
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_PIPELINE_DIR = os.path.join(BASE_DIR, "ml_pipeline")
if ML_PIPELINE_DIR not in sys.path:
    sys.path.append(ML_PIPELINE_DIR)

log_file = open("verify_log.txt", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

print(f"DEBUG: ML_PIPELINE_DIR = {ML_PIPELINE_DIR}")
print(f"DEBUG: Contents of dir: {os.listdir(ML_PIPELINE_DIR)}")

try:
    from h5_omnifusion.models.h5_omnifusion import H5OmniFusion
    from h5_omnifusion.config.model_config import H5Config
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

ckpt_path = os.path.join(ML_PIPELINE_DIR, "h5_omnifusion", "checkpoints", "h5_omnifusion_compliant.pt")
print(f"DEBUG: Checkpoint path = {ckpt_path}")

if not os.path.exists(ckpt_path):
    print("❌ Checkpoint file NOT FOUND")
    sys.exit(1)

try:
    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("✅ Checkpoint loaded from disk")
    
    print("Instantiating model...")
    config = H5Config.from_tier("lite") 
    config.d_model = 96 
    config.moe.expert_hidden_dim = 256
    config.moe.n_quality_features = 4 # Match checkpoint
    config.fusion.n_latents = 16
    config.fusion.n_perceiver_blocks = 1
    
    config.fusion.local_n_heads = 4
    config.fusion.modality_n_heads = 4
    config.fusion.modality_n_layers = 1
    config.fusion.perceiver_n_heads = 4
    
    config.audio.n_mamba_layers = 1
    config.text.n_mamba_layers = 1
    config.video.n_mamba_layers = 1
    config.face.n_lstm_layers = 1
    
    config.text.use_kan = False
    config.tabular.use_kan = False
    config.video.use_timesformer = False
    
    model = H5OmniFusion(config)
    print("Model instantiated")
    
    try:
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Weights loaded result: {msg}")
    except RuntimeError as e:
        print(f"RUNTIME ERROR during load_state_dict: {e}")
        
except Exception as e:
    print(f"Usage/Load Error: {e}")
    import traceback
    traceback.print_exc()
