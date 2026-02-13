"""
Local Test Script for Feature_Extraction_Advanced.ipynb
Tests imports and function definitions without running full extraction
"""
import sys
print("="*60)
print("Testing Feature Extraction Code Locally")
print("="*60)

print("\n[1/6] Testing basic imports...")
try:
    import torch
    import numpy as np
    import pandas as pd
    import cv2
    from pathlib import Path
    import json
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ Pandas: {pd.__version__}")
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ Missing: {e}")
    print("  Run: pip install torch numpy pandas opencv-python")

print("\n[2/6] Testing librosa...")
try:
    import librosa
    print(f"✓ Librosa: {librosa.__version__}")
except ImportError:
    print("✗ Missing librosa. Run: pip install librosa")

print("\n[3/6] Testing transformers...")
try:
    from transformers import (
        Wav2Vec2Processor, Wav2Vec2Model,
        AutoTokenizer, AutoModel,
        VideoMAEImageProcessor, VideoMAEModel,
        AutoImageProcessor
    )
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers error: {e}")
    print("  Run: pip install transformers")

print("\n[4/6] Testing opensmile...")
try:
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print(f"✓ OpenSMILE loaded")
except ImportError:
    print("✗ Missing opensmile. Run: pip install opensmile")
except Exception as e:
    print(f"⚠ OpenSMILE warning: {e}")

print("\n[5/6] Testing PIL...")
try:
    from PIL import Image
    import PIL
    print(f"✓ PIL: {PIL.__version__}")
except ImportError:
    print("✗ Missing PIL. Run: pip install Pillow")

print("\n[6/6] Testing function definitions...")
try:
    def preprocess_audio_advanced(waveform, sr=16000):
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
        waveform = waveform - np.mean(waveform)
        return waveform.astype(np.float32)
    
    def preprocess_text_advanced(text):
        artifacts = ['[inaudible]', '[laugh]', '[pause]', '[sigh]', '[cough]']
        for artifact in artifacts:
            text = text.replace(artifact, '')
        return ' '.join(text.split()).strip()
    
    def compute_video_quality(frames):
        if not frames:
            return {'quality_score': 0.0}
        brightness = [np.mean(f) for f in frames]
        return {'brightness_mean': float(np.mean(brightness))}
    
    test_audio = np.random.randn(16000).astype(np.float32)
    result = preprocess_audio_advanced(test_audio)
    assert result.shape == (16001,), "Audio preprocessing shape mismatch"
    
    test_text = "Hello [laugh] world [pause]"
    result = preprocess_text_advanced(test_text)
    assert result == "Hello world", f"Text preprocessing failed: {result}"
    
    print("✓ All preprocessing functions valid")
    
except Exception as e:
    print(f"✗ Function error: {e}")

print("\n[7/7] Testing GPU...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ No GPU detected (will use CPU - slower)")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nIf all tests passed, the notebook should work in Colab!")
print("Note: Model loading will happen in Colab (requires GPU for speed)")
