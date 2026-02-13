"""
Video Preprocessing Module
Implements Steps 21-24 and R32-R36 from H5-OmniFusion specification.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import CV2_AVAILABLE, robust_video_load

if CV2_AVAILABLE:
    import cv2


class FrameExtractor:
    """
    Extract frames from video at target FPS.
    Steps 21, R32.
    """
    
    def __init__(self, target_fps: int = 5, num_frames: int = 16):
        self.target_fps = target_fps
        self.num_frames = num_frames
    
    def extract(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract frames uniformly from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            (frames, metadata) - frames shape (N, H, W, 3)
        """
        if not CV2_AVAILABLE:
            return self._fallback_result()
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._fallback_result()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames == 0:
                cap.release()
                return self._fallback_result()
            
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((height, width, 3), dtype=np.uint8))
            
            cap.release()
            
            metadata = {
                'success': True,
                'total_frames': total_frames,
                'original_fps': fps,
                'width': width,
                'height': height,
                'duration_sec': duration,
                'extracted_frames': len(frames)
            }
            
            return np.array(frames), metadata
            
        except Exception as e:
            print(f"Frame extraction error: {e}")
            return self._fallback_result()
    
    def _fallback_result(self) -> Tuple[np.ndarray, Dict]:
        """Return fallback empty frames."""
        frames = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        metadata = {
            'success': False,
            'total_frames': 0,
            'original_fps': 0,
            'width': 224,
            'height': 224,
            'duration_sec': 0,
            'extracted_frames': 0
        }
        return frames, metadata


class QualityFilter:
    """
    Filter frames by blur and brightness quality.
    Steps 22, R33-R34.
    
    Thresholds:
    - Blur: Laplacian variance > 50
    - Brightness: Mean 80-180 (0-255 scale)
    """
    
    def __init__(self, 
                 blur_threshold: float = 50.0,
                 brightness_min: int = 80,
                 brightness_max: int = 180):
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
    
    def filter(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Filter frames by quality criteria.
        
        Args:
            frames: Array of frames (N, H, W, 3)
            
        Returns:
            (filtered_frames, quality_info)
        """
        if not CV2_AVAILABLE or len(frames) == 0:
            return frames, self._default_quality_info(len(frames))
        
        quality_scores = []
        passed_indices = []
        
        for i, frame in enumerate(frames):
            blur_score = self._calculate_blur_score(frame)
            brightness = self._calculate_brightness(frame)
            
            blur_ok = blur_score > self.blur_threshold
            brightness_ok = self.brightness_min <= brightness <= self.brightness_max
            
            quality_scores.append({
                'index': i,
                'blur_score': blur_score,
                'brightness': brightness,
                'blur_ok': blur_ok,
                'brightness_ok': brightness_ok,
                'passed': blur_ok and brightness_ok
            })
            
            if blur_ok and brightness_ok:
                passed_indices.append(i)
        
        if len(passed_indices) < 4:
            passed_indices = list(range(len(frames)))
        
        filtered_frames = frames[passed_indices]
        
        quality_info = {
            'original_count': len(frames),
            'passed_count': len(passed_indices),
            'pass_rate': len(passed_indices) / len(frames) if len(frames) > 0 else 0,
            'quality_scores': quality_scores,
            'passed_indices': passed_indices
        }
        
        return filtered_frames, quality_info
    
    def _calculate_blur_score(self, frame: np.ndarray) -> float:
        """Calculate Laplacian variance (higher = less blurry)."""
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()
        except:
            return 0.0
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate mean brightness."""
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            return gray.mean()
        except:
            return 127.0
    
    def _default_quality_info(self, num_frames: int) -> Dict:
        return {
            'original_count': num_frames,
            'passed_count': num_frames,
            'pass_rate': 1.0,
            'quality_scores': [],
            'passed_indices': list(range(num_frames))
        }


class FrameNormalizer:
    """
    Normalize frames using ImageNet statistics.
    Steps 23, R35.
    
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    
    def __init__(self, 
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def normalize(self, frames: np.ndarray) -> np.ndarray:
        """
        Normalize frames.
        
        Args:
            frames: Array (N, H, W, 3) with values 0-255
            
        Returns:
            Normalized frames (N, H, W, 3) with float values
        """
        frames_float = frames.astype(np.float32) / 255.0
        
        normalized = (frames_float - self.mean) / self.std
        
        return normalized.astype(np.float32)
    
    def denormalize(self, frames: np.ndarray) -> np.ndarray:
        """Reverse normalization for visualization."""
        denorm = (frames * self.std) + self.mean
        denorm = np.clip(denorm * 255, 0, 255).astype(np.uint8)
        return denorm


class FrameResizer:
    """
    Resize frames to target resolution.
    Steps 24, R36.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def resize(self, frames: np.ndarray) -> np.ndarray:
        """
        Resize frames to target size.
        
        Args:
            frames: Array (N, H, W, 3)
            
        Returns:
            Resized frames (N, target_H, target_W, 3)
        """
        if not CV2_AVAILABLE:
            return frames
        
        resized = []
        for frame in frames:
            frame_resized = cv2.resize(
                frame, 
                (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            resized.append(frame_resized)
        
        return np.array(resized)
    
    def resize_with_padding(self, frames: np.ndarray) -> np.ndarray:
        """Resize with padding to maintain aspect ratio."""
        if not CV2_AVAILABLE:
            return frames
        
        resized = []
        for frame in frames:
            h, w = frame.shape[:2]
            target_h, target_w = self.target_size
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
            
            resized.append(canvas)
        
        return np.array(resized)


class VideoPreprocessor:
    """
    Unified video preprocessing pipeline (Steps 21-24, R32-R36).
    """
    
    def __init__(self, config=None):
        cfg = config or CFG
        
        self.frame_extractor = FrameExtractor(cfg.TARGET_FPS, cfg.NUM_FRAMES)
        self.quality_filter = QualityFilter(
            cfg.BLUR_THRESHOLD,
            cfg.BRIGHTNESS_MIN,
            cfg.BRIGHTNESS_MAX
        )
        self.normalizer = FrameNormalizer(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
        self.resizer = FrameResizer(cfg.FRAME_SIZE)
    
    def process(self, video_path: str, filter_quality: bool = True,
                normalize: bool = True) -> Dict:
        """
        Run complete video preprocessing pipeline.
        
        Returns:
            Dict with frames, normalized_frames, and metadata
        """
        frames, extract_info = self.frame_extractor.extract(video_path)
        
        if not extract_info['success']:
            return self._failure_result(extract_info)
        
        original_frames = frames.copy()
        
        if filter_quality:
            frames, quality_info = self.quality_filter.filter(frames)
        else:
            quality_info = self.quality_filter._default_quality_info(len(frames))
        
        frames = self.resizer.resize(frames)
        
        frames = self._ensure_frame_count(frames, CFG.NUM_FRAMES)
        
        if normalize:
            normalized_frames = self.normalizer.normalize(frames)
        else:
            normalized_frames = frames.astype(np.float32) / 255.0
        
        return {
            'success': True,
            'frames': frames,  # (N, 224, 224, 3) uint8
            'normalized_frames': normalized_frames,  # (N, 224, 224, 3) float32
            'extract_info': extract_info,
            'quality_info': quality_info
        }
    
    def _ensure_frame_count(self, frames: np.ndarray, target_count: int) -> np.ndarray:
        """Ensure exactly target_count frames."""
        current = len(frames)
        
        if current == target_count:
            return frames
        elif current > target_count:
            indices = np.linspace(0, current - 1, target_count, dtype=int)
            return frames[indices]
        else:
            padding = target_count - current
            last_frame = frames[-1:] if len(frames) > 0 else np.zeros((1, 224, 224, 3), dtype=np.uint8)
            return np.concatenate([frames] + [last_frame] * padding, axis=0)
    
    def _failure_result(self, extract_info: Dict) -> Dict:
        """Return standardized failure result."""
        empty_frames = np.zeros((CFG.NUM_FRAMES, 224, 224, 3), dtype=np.uint8)
        return {
            'success': False,
            'frames': empty_frames,
            'normalized_frames': np.zeros((CFG.NUM_FRAMES, 224, 224, 3), dtype=np.float32),
            'extract_info': extract_info,
            'quality_info': {'original_count': 0, 'passed_count': 0, 'pass_rate': 0}
        }
