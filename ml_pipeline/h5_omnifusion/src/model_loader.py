"""
H5-OmniFusion Model Loader Module
Loads models from LOCAL Google Drive paths - NO INTERNET REQUIRED.
"""
import torch
import os
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from .utils import DEVICE, TRANSFORMERS_AVAILABLE, TIMM_AVAILABLE, clear_memory
except ImportError:
    try:
        from utils import DEVICE, TRANSFORMERS_AVAILABLE, TIMM_AVAILABLE, clear_memory
    except ImportError:
        import gc
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            import transformers
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
        try:
            import timm
            TIMM_AVAILABLE = True
        except ImportError:
            TIMM_AVAILABLE = False
        def clear_memory():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if TRANSFORMERS_AVAILABLE:
    from transformers import (
        Wav2Vec2Model, Wav2Vec2FeatureExtractor,
        VideoMAEModel, VideoMAEImageProcessor,
        AutoModel, AutoTokenizer
    )


class ModelLoader:
    """
    Loads models from LOCAL Google Drive paths.
    
    Your Drive structure:
    /content/drive/MyDrive/DAIC-WOZ_Datasets/pretrained_models/
    ├── audio/wav2vec2-large-xlsr-53/
    ├── text/mental-roberta-base/
    ├── text/bert-base-chinese/
    ├── text/chinese-roberta-wwm-ext/
    ├── video/videomae-base/
    ├── face/dinov2-base/
    └── face/POSTER_V2/
    """
    
    DEFAULT_PRETRAINED_PATH = '/content/drive/MyDrive/DAIC-WOZ_Datasets/pretrained_models'
    
    def __init__(self, device=None, use_fp16: bool = True, pretrained_path: str = None):
        if device is None:
            self.device = DEVICE
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.pretrained_path = pretrained_path or self.DEFAULT_PRETRAINED_PATH
        
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}
        self._status: Dict[str, bool] = {}
        
        print(f"ModelLoader initialized. Pretrained path: {self.pretrained_path}")
    
    @property
    def models(self) -> Dict[str, Any]:
        """Expose _models for backward compatibility."""
        return self._models
    
    @property
    def processors(self) -> Dict[str, Any]:
        """Expose _processors for backward compatibility."""
        return self._processors

    
    def load_wav2vec2(self) -> bool:
        """Load Wav2Vec2 from LOCAL path - STRICT."""
        if 'wav2vec2' in self._models:
            return self._status.get('wav2vec2', False)
        
        local_path = os.path.join(self.pretrained_path, 'audio', 'wav2vec2-large-xlsr-53')
        
        try:
            if os.path.exists(local_path):
                print(f"Loading Wav2Vec2 from LOCAL: {local_path}")
                self._processors['wav2vec2'] = Wav2Vec2FeatureExtractor.from_pretrained(local_path, local_files_only=True)
                model = Wav2Vec2Model.from_pretrained(local_path, local_files_only=True)
            else:
                print(f"❌ LOCAL path not found: {local_path}")
                print("⚠️ STRICT MODE: Skipping Wav2Vec2. Please mount Drive and ensure path exists.")
                self._status['wav2vec2'] = False
                return False
            
            if self.use_fp16:
                model = model.half()
            
            self._models['wav2vec2'] = model.to(self.device).eval()
            self._status['wav2vec2'] = True
            print(f"✓ Loaded Wav2Vec2")
            return True
            
        except Exception as e:
            print(f"✗ Wav2Vec2 failed: {e}")
            self._status['wav2vec2'] = False
            return False
    
    def get_wav2vec2(self):
        """Get Wav2Vec2 model and processor."""
        if 'wav2vec2' not in self._models:
            self.load_wav2vec2()
        return self._models.get('wav2vec2'), self._processors.get('wav2vec2')
    
    
    def load_text_encoder(self, language: str = 'english') -> bool:
        """Load text encoder from LOCAL path - STRICT."""
        key = f'text_{language}'
        
        if key in self._models:
            return self._status.get(key, False)
        
        if language == 'chinese':
            local_folders = ['chinese-roberta-wwm-ext', 'bert-base-chinese']
        else:
            local_folders = ['mental-roberta-base', 'roberta-base']
        
        for local_name in local_folders:
            local_path = os.path.join(self.pretrained_path, 'text', local_name)
            
            try:
                if os.path.exists(local_path):
                    print(f"Loading text encoder from LOCAL: {local_path}")
                    self._processors[key] = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                    model = AutoModel.from_pretrained(local_path, local_files_only=True)
                    
                    if self.use_fp16:
                        model = model.half()
                    
                    self._models[key] = model.to(self.device).eval()
                    self._status[key] = True
                    print(f"✓ Loaded text encoder: {local_name}")
                    return True
                else:
                    print(f"Checking {local_path}... not found.")
                
            except Exception as e:
                print(f"✗ Failed {local_name}: {e}")
                continue
        
        print(f"❌ Could not find any local text encoder for {language} in {self.pretrained_path}")
        self._status[key] = False
        return False
    
    def get_text_encoder(self, language: str = 'english'):
        """Get text encoder model and tokenizer."""
        key = f'text_{language}'
        if key not in self._models:
            self.load_text_encoder(language)
        return self._models.get(key), self._processors.get(key)
    
    
    def load_videomae(self) -> bool:
        """Load VideoMAE from LOCAL path - STRICT."""
        if 'videomae' in self._models:
            return self._status.get('videomae', False)
        
        local_path = os.path.join(self.pretrained_path, 'video', 'videomae-base')
        
        try:
            if os.path.exists(local_path):
                print(f"Loading VideoMAE from LOCAL: {local_path}")
                self._processors['videomae'] = VideoMAEImageProcessor.from_pretrained(local_path, local_files_only=True)
                model = VideoMAEModel.from_pretrained(local_path, local_files_only=True)
                
                if self.use_fp16:
                    model = model.half()
                
                self._models['videomae'] = model.to(self.device).eval()
                self._status['videomae'] = True
                print("✓ Loaded VideoMAE")
                return True
            else:
                 print(f"❌ LOCAL path not found: {local_path}")
        except Exception as e:
            print(f"VideoMAE failed: {e}")
        
        print("⚠️ STRICT MODE: Skipping VideoMAE (No internet fallback).")
        self._status['videomae'] = False
        return False
    
    def get_videomae(self):
        """Get VideoMAE or fallback model."""
        if 'videomae' not in self._models:
            self.load_videomae()
        return self._models.get('videomae'), self._processors.get('videomae')
    
    def load_video_encoder(self) -> bool:
        return self.load_videomae()
    
    
    def load_face_encoder(self) -> bool:
        """Load face encoder from LOCAL path - STRICT."""
        if 'face_encoder' in self._models:
            return self._status.get('face_encoder', False)
        
        local_path = os.path.join(self.pretrained_path, 'face', 'dinov2-base')
        
        try:
            if os.path.exists(local_path):
                print(f"Loading face encoder from LOCAL: {local_path}")
                model = AutoModel.from_pretrained(local_path, local_files_only=True)
                
                if self.use_fp16:
                    model = model.half()
                
                self._models['face_encoder'] = model.to(self.device).eval()
                self._status['face_encoder'] = True
                print("✓ Loaded face encoder (DinoV2)")
                return True
            else:
                print(f"❌ LOCAL path not found: {local_path}")
        except Exception as e:
            print(f"DinoV2 load failed: {e}")
        
        print("⚠️ STRICT MODE: Skipping Face Encoder (No internet fallback).")
        self._status['face_encoder'] = False
        return False
    
    def get_face_encoder(self):
        """Get face encoder model."""
        if 'face_encoder' not in self._models:
            self.load_face_encoder()
        return self._models.get('face_encoder'), None
    
    
    def load_opensmile(self) -> bool:
        """Load OpenSMILE for eGeMAPS features."""
        if 'opensmile' in self._models:
            return self._status.get('opensmile', False)
        
        try:
            import opensmile
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            self._models['opensmile'] = smile
            self._status['opensmile'] = True
            print("✓ Loaded OpenSMILE eGeMAPSv02")
            return True
            
        except ImportError:
            print("OpenSMILE not installed")
            self._status['opensmile'] = False
            return False
        except Exception as e:
            print(f"OpenSMILE failed: {e}")
            self._status['opensmile'] = False
            return False
    
    def get_opensmile(self):
        """Get OpenSMILE extractor."""
        if 'opensmile' not in self._models:
            self.load_opensmile()
        return self._models.get('opensmile'), None
    
    
    def unload_model(self, key: str):
        """Unload a specific model to free memory."""
        if key in self._models:
            del self._models[key]
            self._status[key] = False
        if key in self._processors:
            del self._processors[key]
        clear_memory()
    
    def unload_all(self):
        """Unload all models."""
        self._models.clear()
        self._processors.clear()
        self._status.clear()
        clear_memory()
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of all models."""
        return self._status.copy()


MODEL_LOADER = ModelLoader()
