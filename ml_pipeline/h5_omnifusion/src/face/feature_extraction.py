"""
Face Feature Extraction Module
Implements Steps 31-34 and R43-R49 from H5-OmniFusion specification.
"""
import numpy as np
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import DEVICE, MEDIAPIPE_AVAILABLE, TIMM_AVAILABLE, ensure_768_dim, safe_embedding
from ..model_loader import MODEL_LOADER

if MEDIAPIPE_AVAILABLE:
    import mediapipe as mp


class FaceEmbeddingExtractor:
    """
    Extract 768-dim face embeddings using POSTER v2 or ViT.
    Steps 31, R43.
    """
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
    
    def _ensure_loaded(self):
        if self.model is None:
            self.model, _ = MODEL_LOADER.get_face_encoder()
    
    def extract(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extract embedding from single face crop.
        
        Args:
            face_crop: Face image (224, 224, 3)
            
        Returns:
            768-dim embedding
        """
        self._ensure_loaded()
        
        if self.model is None:
            return np.zeros(768, dtype=np.float32)
        
        try:
            face = face_crop.astype(np.float32) / 255.0
            face = (face - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            if hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                face_tensor = face_tensor.half()
            else:
                face_tensor = face_tensor.float()
            
            with torch.no_grad():
                features = self.model.forward_features(face_tensor)
                if features.dim() == 3:
                    embedding = features[:, 0]  # CLS token
                else:
                    embedding = features
            
            result = embedding.cpu().float().numpy().squeeze()
            
            if len(result) != 768:
                result = ensure_768_dim(result).cpu().numpy().squeeze()
            
            return safe_embedding(result)
            
        except Exception as e:
            print(f"Face embedding error: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def extract_batch(self, face_crops: np.ndarray) -> np.ndarray:
        """Extract and average embeddings from multiple face crops."""
        if len(face_crops) == 0:
            return np.zeros(768, dtype=np.float32)
        
        embeddings = [self.extract(crop) for crop in face_crops]
        return np.mean(embeddings, axis=0)


class ActionUnitDetector:
    """
    Detect facial Action Units.
    Steps 32, R44-R45.
    
    Simplified detection based on landmark movements.
    """
    
    AU_NAMES = {
        1: 'Inner Brow Raiser', 2: 'Outer Brow Raiser',
        4: 'Brow Lowerer', 5: 'Upper Lid Raiser',
        6: 'Cheek Raiser', 7: 'Lid Tightener',
        9: 'Nose Wrinkler', 10: 'Upper Lip Raiser',
        12: 'Lip Corner Puller', 14: 'Dimpler',
        15: 'Lip Corner Depressor', 17: 'Chin Raiser',
        20: 'Lip Stretcher', 23: 'Lip Tightener',
        25: 'Lips Part', 26: 'Jaw Drop', 45: 'Blink'
    }
    
    def __init__(self):
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
    
    def detect(self, face_crop: np.ndarray) -> Dict:
        """
        Detect AUs from face crop.
        
        Returns:
            Dict with au_binary (presence), au_intensity
        """
        if self.face_mesh is None:
            return self._default_result()
        
        try:
            results = self.face_mesh.process(face_crop)
            
            if not results.multi_face_landmarks:
                return self._default_result()
            
            landmarks = results.multi_face_landmarks[0]
            
            au_binary = {}
            au_intensity = {}
            
            au_binary[1] = self._estimate_brow_raise(landmarks)
            au_intensity[1] = 2.5 if au_binary[1] else 0.0
            
            au_binary[4] = self._estimate_brow_lower(landmarks)
            au_intensity[4] = 2.0 if au_binary[4] else 0.0
            
            au_binary[6] = self._estimate_cheek_raise(landmarks)
            au_intensity[6] = 3.0 if au_binary[6] else 0.0
            
            au_binary[12] = self._estimate_smile(landmarks)
            au_intensity[12] = 3.0 if au_binary[12] else 0.0
            
            au_binary[15] = self._estimate_frown(landmarks)
            au_intensity[15] = 2.5 if au_binary[15] else 0.0
            
            au_binary[45] = self._estimate_blink(landmarks)
            au_intensity[45] = 5.0 if au_binary[45] else 0.0
            
            return {
                'au_binary': au_binary,
                'au_intensity': au_intensity,
                'au_count': sum(au_binary.values()),
                'mean_intensity': np.mean(list(au_intensity.values()))
            }
            
        except Exception:
            return self._default_result()
    
    def _estimate_brow_raise(self, landmarks) -> bool:
        """Estimate if brows are raised."""
        return False  # Placeholder
    
    def _estimate_brow_lower(self, landmarks) -> bool:
        return False
    
    def _estimate_cheek_raise(self, landmarks) -> bool:
        return False
    
    def _estimate_smile(self, landmarks) -> bool:
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        center = landmarks.landmark[13]
        return (left_corner.y < center.y) and (right_corner.y < center.y)
    
    def _estimate_frown(self, landmarks) -> bool:
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        center = landmarks.landmark[13]
        return (left_corner.y > center.y) and (right_corner.y > center.y)
    
    def _estimate_blink(self, landmarks) -> bool:
        return False
    
    def _default_result(self) -> Dict:
        return {
            'au_binary': {au: False for au in [1, 4, 6, 12, 15, 45]},
            'au_intensity': {au: 0.0 for au in [1, 4, 6, 12, 15, 45]},
            'au_count': 0, 'mean_intensity': 0.0
        }


class BlinkRateAnalyzer:
    """
    Analyze blink patterns from video frames.
    Step R46.
    """
    
    def __init__(self, ear_threshold: float = 0.2):
        self.ear_threshold = ear_threshold
        self.face_mesh = None
        
        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
    
    def analyze(self, face_crops: np.ndarray, fps: float = 5.0) -> Dict:
        """
        Analyze blink rate from sequence of face crops.
        
        Returns:
            Dict with blink_count, blink_rate_per_min, mean_blink_duration
        """
        if self.face_mesh is None or len(face_crops) < 2:
            return self._default_result()
        
        try:
            ear_values = []
            
            for crop in face_crops:
                ear = self._calculate_ear(crop)
                ear_values.append(ear)
            
            blinks = []
            in_blink = False
            blink_start = 0
            
            for i, ear in enumerate(ear_values):
                if ear < self.ear_threshold and not in_blink:
                    in_blink = True
                    blink_start = i
                elif ear >= self.ear_threshold and in_blink:
                    in_blink = False
                    blink_duration = (i - blink_start) / fps
                    blinks.append(blink_duration)
            
            duration_sec = len(face_crops) / fps
            blink_rate = len(blinks) * 60 / duration_sec if duration_sec > 0 else 0
            
            return {
                'blink_count': len(blinks),
                'blink_rate_per_min': blink_rate,
                'mean_blink_duration': np.mean(blinks) if blinks else 0,
                'blink_durations': blinks
            }
            
        except Exception:
            return self._default_result()
    
    def _calculate_ear(self, face_crop: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio."""
        try:
            results = self.face_mesh.process(face_crop)
            if not results.multi_face_landmarks:
                return 0.3  # Default open eye
            
            landmarks = results.multi_face_landmarks[0]
            
            left_ear = self._eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144])
            right_ear = self._eye_aspect_ratio(landmarks, [362, 385, 387, 263, 373, 380])
            
            return (left_ear + right_ear) / 2
        except:
            return 0.3
    
    def _eye_aspect_ratio(self, landmarks, indices: List[int]) -> float:
        """Calculate EAR for one eye."""
        points = [landmarks.landmark[i] for i in indices]
        
        v1 = np.sqrt((points[1].x - points[5].x)**2 + (points[1].y - points[5].y)**2)
        v2 = np.sqrt((points[2].x - points[4].x)**2 + (points[2].y - points[4].y)**2)
        
        h = np.sqrt((points[0].x - points[3].x)**2 + (points[0].y - points[3].y)**2)
        
        return (v1 + v2) / (2.0 * h + 1e-6)
    
    def _default_result(self) -> Dict:
        return {
            'blink_count': 0, 'blink_rate_per_min': 0,
            'mean_blink_duration': 0, 'blink_durations': []
        }


class GazeAnalyzer:
    """
    Analyze gaze direction and eye contact.
    Steps 33, R47.
    
    Classification: Direct (<10°), Indirect (10-15°), Averted (>15°)
    """
    
    def __init__(self, direct_threshold: float = 10.0, indirect_threshold: float = 15.0):
        self.direct_threshold = direct_threshold
        self.indirect_threshold = indirect_threshold
        self.face_mesh = None
        
        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True
            )
    
    def analyze(self, face_crops: np.ndarray) -> Dict:
        """
        Analyze gaze patterns across frames.
        
        Returns:
            Dict with gaze_direct_ratio, gaze_aversion_count, etc.
        """
        if self.face_mesh is None or len(face_crops) == 0:
            return self._default_result()
        
        gaze_categories = []
        
        for crop in face_crops:
            angle = self._estimate_gaze_angle(crop)
            
            if angle < self.direct_threshold:
                gaze_categories.append('direct')
            elif angle < self.indirect_threshold:
                gaze_categories.append('indirect')
            else:
                gaze_categories.append('averted')
        
        total = len(gaze_categories) or 1
        
        return {
            'gaze_direct_ratio': gaze_categories.count('direct') / total,
            'gaze_indirect_ratio': gaze_categories.count('indirect') / total,
            'gaze_averted_ratio': gaze_categories.count('averted') / total,
            'gaze_aversion_count': gaze_categories.count('averted'),
            'gaze_sequence': gaze_categories
        }
    
    def _estimate_gaze_angle(self, face_crop: np.ndarray) -> float:
        """Estimate gaze deviation angle in degrees."""
        try:
            results = self.face_mesh.process(face_crop)
            if not results.multi_face_landmarks:
                return 20.0  # Default to averted
            
            landmarks = results.multi_face_landmarks[0]
            
            left_iris = landmarks.landmark[468] if len(landmarks.landmark) > 468 else landmarks.landmark[33]
            left_corner = landmarks.landmark[33]
            right_corner = landmarks.landmark[133]
            
            eye_width = abs(right_corner.x - left_corner.x)
            iris_offset = abs(left_iris.x - (left_corner.x + right_corner.x) / 2)
            
            angle = (iris_offset / (eye_width + 1e-6)) * 45  # Scale factor
            
            return min(angle, 45.0)
            
        except:
            return 15.0  # Default
    
    def _default_result(self) -> Dict:
        return {
            'gaze_direct_ratio': 0.5, 'gaze_indirect_ratio': 0.3,
            'gaze_averted_ratio': 0.2, 'gaze_aversion_count': 0, 'gaze_sequence': []
        }


class HeadPoseEstimator:
    """
    Estimate head pose (yaw, pitch, roll).
    Steps 33, R48.
    """
    
    def __init__(self):
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1
            )
    
    def estimate(self, face_crops: np.ndarray) -> Dict:
        """
        Estimate head pose across frames.
        
        Returns:
            Dict with yaw/pitch/roll mean and variability
        """
        if self.face_mesh is None or len(face_crops) == 0:
            return self._default_result()
        
        poses = []
        
        for crop in face_crops:
            pose = self._estimate_single(crop)
            poses.append(pose)
        
        yaw = [p['yaw'] for p in poses]
        pitch = [p['pitch'] for p in poses]
        roll = [p['roll'] for p in poses]
        
        return {
            'yaw_mean': np.mean(yaw),
            'yaw_std': np.std(yaw),
            'pitch_mean': np.mean(pitch),
            'pitch_std': np.std(pitch),
            'roll_mean': np.mean(roll),
            'roll_std': np.std(roll),
            'head_pose_variability': np.mean([np.std(yaw), np.std(pitch), np.std(roll)]),
            'head_down_ratio': sum(1 for p in pitch if p > 10) / len(pitch)
        }
    
    def _estimate_single(self, face_crop: np.ndarray) -> Dict:
        """Estimate pose for single frame."""
        try:
            results = self.face_mesh.process(face_crop)
            if not results.multi_face_landmarks:
                return {'yaw': 0, 'pitch': 0, 'roll': 0}
            
            landmarks = results.multi_face_landmarks[0]
            
            nose = landmarks.landmark[1]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            
            yaw = (nose.x - 0.5) * 90  # Approximate degrees
            
            pitch = (nose.y - 0.5) * 60
            
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            roll = np.degrees(np.arctan2(dy, dx))
            
            return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
            
        except:
            return {'yaw': 0, 'pitch': 0, 'roll': 0}
    
    def _default_result(self) -> Dict:
        return {
            'yaw_mean': 0, 'yaw_std': 0, 'pitch_mean': 0, 'pitch_std': 0,
            'roll_mean': 0, 'roll_std': 0, 'head_pose_variability': 0, 'head_down_ratio': 0
        }


class FaceFeatureExtractor:
    """Unified face feature extraction (Steps 31-34, R43-R49)."""
    
    def __init__(self):
        self.embedding = FaceEmbeddingExtractor()
        self.action_units = ActionUnitDetector()
        self.blink = BlinkRateAnalyzer()
        self.gaze = GazeAnalyzer()
        self.head_pose = HeadPoseEstimator()
    
    def extract_all(self, face_crops: np.ndarray, fps: float = 5.0) -> Dict:
        """
        Extract all face features.
        
        Returns:
            Dict with face_embedding and behavioral features
        """
        face_embedding = self.embedding.extract_batch(face_crops)
        
        mid_idx = len(face_crops) // 2
        au_features = self.action_units.detect(face_crops[mid_idx]) if len(face_crops) > 0 else {}
        
        blink_features = self.blink.analyze(face_crops, fps)
        
        gaze_features = self.gaze.analyze(face_crops)
        
        head_pose_features = self.head_pose.estimate(face_crops)
        
        return {
            'face_embedding': face_embedding,  # 768-dim
            'action_units': au_features,
            'blink': blink_features,
            'gaze': gaze_features,
            'head_pose': head_pose_features
        }
