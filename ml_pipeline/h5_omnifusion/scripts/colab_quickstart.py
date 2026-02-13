

"""
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/phase 2/ml_pipeline/h5_omnifusion')
"""

"""
from src import H5OmniFusionPipeline, CFG, run_pipeline

# Check device
from src.utils import DEVICE
print(f"Running on: {DEVICE}")
"""

"""
# Update paths to match your Google Drive structure
CFG.DAIC_WOZ_PATH = '/content/drive/MyDrive/DAIC-WOZ_Datasets'
CFG.EATD_CORPUS_PATH = '/content/drive/MyDrive/EATD-Corpus'
CFG.PRETRAINED_PATH = '/content/drive/MyDrive/pretrained_models'
CFG.OUTPUT_PATH = '/content/drive/MyDrive/h5_outputs'
"""

"""
# Example: Process a single DAIC-WOZ participant
result = run_pipeline(
    audio_path='/content/drive/MyDrive/DAIC-WOZ_Datasets/300/300_AUDIO.wav',
    video_path='/content/drive/MyDrive/DAIC-WOZ_Datasets/300/300_CLNF_AUs.txt',  # Or video file
    transcript_path='/content/drive/MyDrive/DAIC-WOZ_Datasets/300/300_TRANSCRIPT.csv',
    participant_id='300'
)

# Check results
print(f"Success: {result['success']}")
print(f"Audio embedding shape: {result['audio']['wav2vec2_embedding'].shape}")
print(f"Text embedding shape: {result['text']['text_embedding'].shape}")
print(f"Fused embedding shape: {result['fusion']['fused_embedding'].shape}")
"""

"""
# Initialize pipeline
pipeline = H5OmniFusionPipeline()

# Process DAIC-WOZ
results = pipeline.process_dataset(dataset='daic_woz', split='train')
print(f"Processed {len(results)} participants")

# Or process EATD-Corpus (no video)
# results = pipeline.process_dataset(dataset='eatd_corpus')
"""

"""
import h5py
import numpy as np

# Load a saved H5 file
with h5py.File('/content/drive/MyDrive/h5_outputs/300.h5', 'r') as f:
    print("Participant:", f.attrs['participant_id'])
    print("Language:", f.attrs['language'])
    print("Quality:", f.attrs['overall_quality'])
    
    print("\\nEmbeddings:")
    for key in f['embeddings'].keys():
        data = f['embeddings'][key][:]
        print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
"""

"""
# Display all extracted features
def print_features(result):
    print("\\n=== AUDIO FEATURES ===")
    print(f"  Wav2Vec2: {result['audio']['wav2vec2_embedding'].shape}")
    print(f"  eGeMAPS: {result['audio']['egemaps_embedding'].shape}")
    print(f"  Pitch: {result['audio']['pitch']}")
    print(f"  Pauses: {result['audio']['pauses']}")
    
    print("\\n=== TEXT FEATURES ===")
    print(f"  Embedding: {result['text']['text_embedding'].shape}")
    print(f"  Sentiment: {result['text']['sentiment']}")
    print(f"  Linguistic: {result['text']['linguistic']}")
    
    print("\\n=== VIDEO FEATURES ===")
    print(f"  Embedding: {result['video']['video_embedding'].shape}")
    print(f"  Motion: {result['video']['optical_flow']}")
    
    print("\\n=== FACE FEATURES ===")
    print(f"  Embedding: {result['face']['face_embedding'].shape}")
    print(f"  Gaze: {result['face']['gaze']}")
    print(f"  Blink: {result['face']['blink']}")
    
    print("\\n=== FUSION ===")
    print(f"  Fused: {result['fusion']['fused_embedding'].shape}")
    print(f"  Congruence: {result['fusion']['congruence']}")

# print_features(result)
"""

print("Colab script ready! Copy cells to your notebook.")
