
import os
import sys
import numpy as np
import torch
import inspect

sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion')
sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\ml_pipeline\h5_omnifusion\preprocessing_and_feature_extraction')

try:
    from pipeline_fusion_main import H5OmniFusionPipeline
    from pipeline_audio import AudioPreprocessor
    from pipeline_video_face import VideoPreprocessor
    from pipeline_text import TextPreprocessor
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def verify_tabular():
    print("\nVerifying Tabular Integration...")
    try:
        class MockCfg:
            EMBED_DIM = 768
            DEVICE = 'cpu'
            DAIC_DIR = '.'
            EATD_DIR = '.'
            OUTPUT_DIR = '.'
            
        pipeline = H5OmniFusionPipeline(cfg=MockCfg())
        
        if hasattr(pipeline, 'tabular_projector') and pipeline.tabular_projector is not None:
            print("✅ TabularProjector instantiated.")
        else:
            print("❌ TabularProjector NOT instantiated.")
            
        if hasattr(pipeline, 'num_norm') and pipeline.num_norm is not None:
            print("✅ NumericalNormalizer instantiated.")
        else:
            print("❌ NumericalNormalizer NOT instantiated.")
            
        
        import inspect
        src = inspect.getsource(pipeline.process_daic_participant)
        if "result['tabular_embedding']" in src and "self.tabular_projector" in src:
             print("✅ Tabular logic present in process_daic_participant.")
        else:
             print("❌ Tabular logic MISSING in process_daic_participant.")

    except Exception as e:
        print(f"Tabular verification failed: {e}")

def verify_augmentation():
    print("\nVerifying Augmentation Wiring...")
    
    sig_audio = inspect.signature(AudioPreprocessor.process)
    if 'augment' in sig_audio.parameters:
        print("✅ AudioPreprocessor.process accepts 'augment' flag.")
    else:
        print("❌ AudioPreprocessor.process MISSING 'augment' flag.")
        
    sig_video = inspect.signature(VideoPreprocessor.process)
    if 'augment' in sig_video.parameters:
        print("✅ VideoPreprocessor.process accepts 'augment' flag.")
    else:
        print("❌ VideoPreprocessor.process MISSING 'augment' flag.")

    sig_text = inspect.signature(TextPreprocessor.process)
    if 'augment' in sig_text.parameters:
        print("✅ TextPreprocessor.process accepts 'augment' flag.")
    else:
        print("❌ TextPreprocessor.process MISSING 'augment' flag.")

if __name__ == "__main__":
    verify_tabular()
    verify_augmentation()
