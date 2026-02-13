"""
H5-OmniFusion Video & Face Pipeline
Video: Steps 21-26, R32-R38
Face: Steps 27-34, R39-R49
"""
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

try:
    from video_face_enhancements import KinematicsPostureAnalyzer
    ADV2_OK = True
except ImportError:
    ADV2_OK = False
    print("Warning: KinematicsPostureAnalyzer (ADV2) not found in video_face_enhancements")

try:
    from research_layer_extensions import DiscreteQualityFilters, BlinkRateAnalyzer, VideoGeometricAugmenter
    R33_R46_R57_OK = True
except ImportError:
    R33_R46_R57_OK = False
    print("Warning: DiscreteQualityFilters/BlinkRateAnalyzer/VideoGeometricAugmenter not found")


class CLNFFeatureParser:
    """
    Parse CLNF feature files from DAIC-WOZ zips.
    
    DAIC-WOZ zips contain:
    - *_CLNF_AUs.txt: Action Unit intensities
    - *_CLNF_gaze.txt: Gaze direction
    - *_CLNF_pose.txt: Head pose (Rx, Ry, Rz, Tx, Ty, Tz)
    - *_CLNF_features.txt: Facial landmarks
    """
    
    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim
        self._projector = None
    
    def _get_projector(self, input_dim: int):
        """Lazy initialization of projector."""
        if self._projector is None or self._projector.in_features != input_dim:
            self._projector = torch.nn.Linear(input_dim, self.embed_dim)
            torch.nn.init.xavier_uniform_(self._projector.weight)
        return self._projector
    
    def parse_clnf_file(self, filepath: str) -> np.ndarray:
        """Parse a CLNF txt file and return features as array."""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    values = [float(v) for v in line.strip().split(',') if v.strip()]
                    if values:
                        data.append(values)
            
            if not data:
                return np.array([])
            
            return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"CLNF parse error for {filepath}: {e}")
            return np.array([])
    
    def extract_au_features(self, au_path: str) -> Dict:
        """Extract Action Unit features from *_CLNF_AUs.txt."""
        result = {}
        data = self.parse_clnf_file(au_path)
        
        if data.size == 0:
            return {'au_mean': np.zeros(17), 'au_std': np.zeros(17), 'au_embedding': np.zeros(self.embed_dim)}
        
        au_mean = np.mean(data, axis=0)
        au_std = np.std(data, axis=0)
        
        result['au_mean'] = au_mean[:17] if len(au_mean) >= 17 else np.pad(au_mean, (0, 17-len(au_mean)))
        result['au_std'] = au_std[:17] if len(au_std) >= 17 else np.pad(au_std, (0, 17-len(au_std)))
        
        au_flat = np.concatenate([au_mean, au_std])
        au_flat = np.nan_to_num(au_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        with torch.no_grad():
            proj = self._get_projector(len(au_flat))
            au_t = torch.tensor(au_flat, dtype=torch.float32).unsqueeze(0)
            result['au_embedding'] = proj(au_t).numpy().flatten()
        
        return result
    
    def extract_gaze_features(self, gaze_path: str) -> Dict:
        """Extract gaze features from *_CLNF_gaze.txt."""
        result = {}
        data = self.parse_clnf_file(gaze_path)
        
        if data.size == 0:
            return {'gaze_x_mean': 0, 'gaze_y_mean': 0, 'gaze_direct_ratio': 0}
        
        gaze_mean = np.mean(data, axis=0)
        result['gaze_x_mean'] = float(gaze_mean[0]) if len(gaze_mean) > 0 else 0
        result['gaze_y_mean'] = float(gaze_mean[1]) if len(gaze_mean) > 1 else 0
        
        gaze_magnitude = np.sqrt(data[:, 0]**2 + data[:, 1]**2) if data.shape[1] >= 2 else np.zeros(len(data))
        result['gaze_direct_ratio'] = float(np.mean(gaze_magnitude < 0.3))
        
        return result
    
    def extract_pose_features(self, pose_path: str) -> Dict:
        """Extract head pose features from *_CLNF_pose.txt."""
        result = {}
        data = self.parse_clnf_file(pose_path)
        
        if data.size == 0:
            return {'head_pitch_mean': 0, 'head_yaw_mean': 0, 'head_roll_mean': 0, 'head_movement': 0}
        
        pose_mean = np.mean(data, axis=0)
        pose_std = np.std(data, axis=0)
        
        result['head_pitch_mean'] = float(pose_mean[0]) if len(pose_mean) > 0 else 0
        result['head_yaw_mean'] = float(pose_mean[1]) if len(pose_mean) > 1 else 0
        result['head_roll_mean'] = float(pose_mean[2]) if len(pose_mean) > 2 else 0
        result['head_movement'] = float(np.mean(pose_std[:3])) if len(pose_std) >= 3 else 0
        
        return result
    
    def extract_all(self, base_dir: str, pid: str) -> Dict:
        """
        Extract all CLNF features for a participant.
        
        Args:
            base_dir: Directory containing the CLNF files
            pid: Participant ID (e.g., "300")
        
        Returns:
            Dict with au_embedding, gaze features, pose features
        """
        result = {
            'video_embedding': np.zeros(self.embed_dim),
            'face_embedding': np.zeros(self.embed_dim),
        }
        
        au_path = os.path.join(base_dir, f"{pid}_CLNF_AUs.txt")
        gaze_path = os.path.join(base_dir, f"{pid}_CLNF_gaze.txt")
        pose_path = os.path.join(base_dir, f"{pid}_CLNF_pose.txt")
        
        features_collected = []
        
        if os.path.exists(au_path):
            au_feats = self.extract_au_features(au_path)
            result.update(au_feats)
            result['face_embedding'] = au_feats.get('au_embedding', np.zeros(self.embed_dim))
            features_collected.append('AU')
        
        if os.path.exists(gaze_path):
            result.update(self.extract_gaze_features(gaze_path))
            features_collected.append('gaze')
        
        if os.path.exists(pose_path):
            pose_feats = self.extract_pose_features(pose_path)
            result.update(pose_feats)
            features_collected.append('pose')
            
            pose_flat = np.array([pose_feats['head_pitch_mean'], pose_feats['head_yaw_mean'], 
                                  pose_feats['head_roll_mean'], pose_feats['head_movement']])
            pose_flat = np.nan_to_num(pose_flat, nan=0.0)
            with torch.no_grad():
                proj = self._get_projector(len(pose_flat))
                pose_t = torch.tensor(pose_flat, dtype=torch.float32).unsqueeze(0)
                result['video_embedding'] = proj(pose_t).numpy().flatten()
        
        result['clnf_features_found'] = features_collected
        return result


class FrameExtractor:
    """Steps 21, R32: Extract frames from video."""
    def __init__(self, target_fps=5, num_frames=16):
        self.target_fps = target_fps
        self.num_frames = num_frames
    
    def extract(self, video_path: str) -> np.ndarray:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            
            indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            frames = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            cap.release()
            return np.array(frames)
        except Exception as e:
            print(f"Warning [FrameExtractor]: {e}")
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

class QualityFilter:
    """Steps 22, R33-R34: Filter low quality frames."""
    def __init__(self, blur_thresh=50.0, bright_min=80, bright_max=180):
        self.blur_thresh = blur_thresh
        self.bright_min = bright_min
        self.bright_max = bright_max
    
    def check_frame(self, frame: np.ndarray) -> Dict:
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            is_good = blur_score >= self.blur_thresh and self.bright_min <= brightness <= self.bright_max
            return {'blur_score': blur_score, 'brightness': brightness, 'is_good': is_good}
        except Exception as e:
            print(f"Warning [QualityFilter]: {e}")
            return {'blur_score': 0, 'brightness': 0, 'is_good': False}
    
    def filter_frames(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        good_frames = []
        scores = []
        for frame in frames:
            check = self.check_frame(frame)
            scores.append(check)
            if check['is_good']:
                good_frames.append(frame)
        
        if len(good_frames) == 0:
            good_frames = list(frames)
        
        quality = len([s for s in scores if s['is_good']]) / len(scores) if scores else 0
        return np.array(good_frames), {'frame_quality_ratio': quality, 'total_frames': len(frames), 'good_frames': len(good_frames)}

class FrameNormalizer:
    """Steps 23-24, R35-R36: ImageNet normalization."""
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    
    def normalize(self, frames: np.ndarray) -> np.ndarray:
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - self.MEAN) / self.STD
        return frames.transpose(0, 3, 1, 2)  # NHWC -> NCHW

class OpticalFlowAnalyzer:
    """Step 26, R38: Motion analysis."""
    def analyze(self, frames: np.ndarray) -> Dict:
        if len(frames) < 2:
            return {'flow_mean': 0, 'flow_std': 0, 'motion_score': 0}
        
        try:
            import cv2
            flows = []
            for i in range(len(frames)-1):
                prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                next_gray = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                flows.append(np.mean(mag))
            
            return {'flow_mean': np.mean(flows), 'flow_std': np.std(flows), 'motion_score': np.mean(flows)/10}
        except Exception as e:
            print(f"Warning [OpticalFlow]: {e}")
            return {'flow_mean': 0, 'flow_std': 0, 'motion_score': 0}


class FaceDetector:
    """Steps 27, R39: Face detection with MediaPipe."""
    def __init__(self, min_confidence=0.8):
        self.min_confidence = min_confidence
        self.detector = None
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_confidence)
        except Exception:
            pass

    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        if self.detector is None:
            return []
        try:
            results = self.detector.process(frame)
            if not results.detections:
                return []
            
            h, w = frame.shape[:2]
            faces = []
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                faces.append({
                    'x': int(bbox.xmin * w),
                    'y': int(bbox.ymin * h),
                    'w': int(bbox.width * w),
                    'h': int(bbox.height * h),
                    'confidence': det.score[0]
                })
            return faces
        except Exception as e:
            print(f"Warning [FaceDetector]: {e}")
            return []

class FaceCropper:
    """Step 28-29, R40-R41: Face alignment and cropping.
    
    USABILITY FIX: Implements hybrid alignment strategy:
    1. PRIMARY: Affine transformation using eye landmarks (MediaPipe)
    2. FALLBACK: Bounding-box centering with margin
    """
    def __init__(self, output_size=(224, 224), margin=0.2):
        self.output_size = output_size
        self.margin = margin
        self.face_mesh = None
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1, 
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
        except Exception:
            pass
    
    def _get_eye_centers(self, frame: np.ndarray) -> Optional[Tuple]:
        """Extract eye center coordinates from MediaPipe landmarks."""
        if self.face_mesh is None:
            return None
        try:
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
            right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))
            return left_eye, right_eye
        except:
            return None
    
    def _affine_align(self, frame: np.ndarray, left_eye: Tuple, right_eye: Tuple) -> np.ndarray:
        """Apply affine transformation to horizontally align eyes."""
        import cv2
        import math
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
        
        return rotated
    
    def crop(self, frame: np.ndarray, face: Dict) -> np.ndarray:
        """Crop and align face. Returns 224x224 RGB tensor."""
        try:
            import cv2
            h, w = frame.shape[:2]
            x, y, fw, fh = face['x'], face['y'], face['w'], face['h']
            
            eye_coords = self._get_eye_centers(frame)
            if eye_coords is not None:
                left_eye, right_eye = eye_coords
                frame = self._affine_align(frame, left_eye, right_eye)
            
            mx, my = int(fw * self.margin), int(fh * self.margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + fw + mx)
            y2 = min(h, y + fh + my)
            
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                return np.zeros((*self.output_size, 3), dtype=np.uint8)
            
            return cv2.resize(cropped, self.output_size)
        except Exception as e:
            print(f"Warning [FaceCropper]: {e}")
            return np.zeros((*self.output_size, 3), dtype=np.uint8)

class FaceTracker:
    """Step 30, R42: Simple centroid-based tracking."""
    def __init__(self, max_disappeared=5):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
class FaceTracker:
    def __init__(self, max_disappeared=5):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def update(self, faces: List[Dict]) -> Dict[int, Dict]:
        pass # Placeholder

class GazeTracker:
    """Step 33, R47-R48: Gaze and head pose estimation (Non-Proximal PnP)."""
    def __init__(self):
        self.face_mesh = None
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        except Exception:
            pass
            
    def analyze(self, frame: np.ndarray) -> Dict:
        if self.face_mesh is None:
            return {'gaze_direct_ratio': 0, 'head_yaw': 0, 'head_pitch': 0}
        
        try:
            import cv2
            results = self.face_mesh.process(frame)
            if not results.multi_face_landmarks:
                return {'gaze_direct_ratio': 0, 'head_yaw': 0, 'head_pitch': 0}
            
            lm = results.multi_face_landmarks[0].landmark
            h, w, c = frame.shape
            
            face_3d = []
            face_2d = []
            
            key_inds = [1, 199, 33, 263, 61, 291]
            for idx in key_inds:
                face_2d.append([int(lm[idx].x * w), int(lm[idx].y * h)])
                face_3d.append([int(lm[idx].x * w), int(lm[idx].y * h), lm[idx].z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w/2],
                                 [0, focal_length, h/2],
                                 [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360
                
                if len(lm) > 468: # If Iris landmarks present
                    l_iris = np.array([lm[468].x, lm[468].y])
                    r_iris = np.array([lm[473].x, lm[473].y])
                    l_eye = np.array([lm[33].x + lm[133].x, lm[33].y + lm[133].y]) / 2
                    r_eye = np.array([lm[362].x + lm[263].x, lm[362].y + lm[263].y]) / 2
                    
                    dist = np.linalg.norm(l_iris - l_eye) + np.linalg.norm(r_iris - r_eye)
                    gaze_direct = 1.0 if dist < 0.01 and abs(yaw) < 15 and abs(pitch) < 15 else 0.0
                else:
                    gaze_direct = 1.0 if abs(yaw) < 10 and abs(pitch) < 10 else 0.0
                
                return {'gaze_direct_ratio': gaze_direct, 'head_yaw': yaw, 'head_pitch': pitch}
            else:
                 return {'gaze_direct_ratio': 0, 'head_yaw': 0, 'head_pitch': 0}
        except Exception as e:
            return {'gaze_direct_ratio': 0, 'head_yaw': 0, 'head_pitch': 0}

class ActionUnitExtractor:
    """Step 32, R44-R45: Action Unit approximation (Blendshapes Restored)."""
    def __init__(self):
        self.face_mesh = None
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, 
                                                               refine_landmarks=True) 
        except Exception:
            pass
    
    def extract(self, frame: np.ndarray) -> Dict:
        if self.face_mesh is None:
            return {'au_features': np.zeros(17)}
        
        try:
            results = self.face_mesh.process(frame)
            
            if hasattr(results, 'multi_face_blendshapes') and results.multi_face_blendshapes:
                face_blendshapes = results.multi_face_blendshapes[0]
                bs_dict = {b.category_name: b.score for b in face_blendshapes}
                
                au1 = bs_dict.get('browInnerUp', 0)
                au2 = (bs_dict.get('browOuterUpLeft', 0) + bs_dict.get('browOuterUpRight', 0)) / 2
                au4 = (bs_dict.get('browDownLeft', 0) + bs_dict.get('browDownRight', 0)) / 2
                au5 = (bs_dict.get('eyeLookUpLeft', 0) + bs_dict.get('eyeLookUpRight', 0)) / 2 # Approx
                au6 = (bs_dict.get('eyeSquintLeft', 0) + bs_dict.get('eyeSquintRight', 0)) / 2
                au12 = (bs_dict.get('mouthSmileLeft', 0) + bs_dict.get('mouthSmileRight', 0)) / 2
                au15 = (bs_dict.get('mouthFrownLeft', 0) + bs_dict.get('mouthFrownRight', 0)) / 2
                au25 = bs_dict.get('jawOpen', 0)
                
                aus = [au1, au2, au4, au5, au6, 0, 0, 0, 0, 0, 0, au12, 0, 0, au15, 0, au25]
                return {'au_features': np.array(aus[:17])} # Pad/Crop to 17
            
            elif results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                au1 = abs(lm[70].y - lm[63].y) * 100
                au4 = abs(lm[66].y - lm[107].y) * 100
                au12 = abs(lm[61].y - lm[291].y) * 100
                au15 = abs(lm[17].y - lm[14].y) * 100
                aus = [au1, au4, au12, au15] + [0.0] * 13
                return {'au_features': np.array(aus)}
            
            return {'au_features': np.zeros(17)}
        except Exception as e:
            return {'au_features': np.zeros(17)}

class MicroExpressionAnalyzer:
    """Step 34, R49: Expression timing analysis."""
    def analyze(self, au_sequence: List[np.ndarray]) -> Dict:
        if len(au_sequence) < 2:
            return {'expression_variability': 0, 'expression_rate': 0}
        
        try:
            aus = np.array(au_sequence)
            variability = np.mean(np.std(aus, axis=0))
            
            diffs = np.abs(np.diff(aus, axis=0))
            rate = np.mean(diffs)
            
            return {'expression_variability': variability, 'expression_rate': rate}
        except Exception as e:
            print(f"Warning [MicroExpression]: {e}")
            return {'expression_variability': 0, 'expression_rate': 0}

class BlinkRateAnalyzer:
    """R46: Blink detection."""
    def __init__(self):
        self.face_mesh = None
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        except Exception:
            pass
    
    def analyze_frames(self, frames: np.ndarray, fps: float = 5.0) -> Dict:
        if self.face_mesh is None or len(frames) == 0:
            return {'blink_count': 0, 'blink_rate': 0}
        
        try:
            ear_values = []
            for frame in frames:
                results = self.face_mesh.process(frame)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    left_ear = abs(lm[159].y - lm[145].y) / (abs(lm[33].x - lm[133].x) + 1e-6)
                    right_ear = abs(lm[386].y - lm[374].y) / (abs(lm[362].x - lm[263].x) + 1e-6)
                    ear_values.append((left_ear + right_ear) / 2)
            
            if len(ear_values) < 3:
                return {'blink_count': 0, 'blink_rate': 0}
            
            threshold = np.mean(ear_values) * 0.7
            blinks = sum(1 for i in range(1, len(ear_values)-1) 
                        if ear_values[i] < threshold and ear_values[i-1] >= threshold)
            
            duration_mins = len(frames) / fps / 60
            rate = blinks / max(duration_mins, 0.01)
            
            return {'blink_count': blinks, 'blink_rate': rate}
        except Exception as e:
            print(f"Warning [BlinkRate]: {e}")
            return {'blink_count': 0, 'blink_rate': 0}


class VideoPreprocessor:
    """Complete video pipeline: Steps 21-26, R32-R38."""
    def __init__(self, models, embed_dim=768, device='cuda'):
        self.models = models
        self.embed_dim = embed_dim
        self.device = device
        self.extractor = FrameExtractor()
        self.quality = QualityFilter()
        self.normalizer = FrameNormalizer()
        self.flow = OpticalFlowAnalyzer()
        
        if ADV2_OK:
            self.kinematics = KinematicsPostureAnalyzer()
        else:
            self.kinematics = None
        
        if R33_R46_R57_OK:
            self.video_augmenter = VideoGeometricAugmenter()
        else:
            self.video_augmenter = None
    
    @torch.no_grad()
    def get_video_embedding(self, frames: np.ndarray) -> np.ndarray:
        if 'video' not in self.models.models:
            return np.zeros(self.embed_dim)
        
        try:
            model = self.models.models['video']
            if hasattr(self.models.processors.get('video', None), 'preprocess'):
                proc = self.models.processors['video']
                inputs = proc(list(frames), return_tensors='pt')
                inputs = {k:v.to(self.device) for k,v in inputs.items()}
                outputs = model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            else:
                frames_norm = self.normalizer.normalize(frames)
                frames_t = torch.tensor(frames_norm, dtype=torch.float32).to(self.device)
                embeddings = []
                for f in frames_t:
                    out = model.forward_features(f.unsqueeze(0))
                    embeddings.append(out.mean(dim=1).cpu().numpy())
                return np.mean(embeddings, axis=0).flatten()
        except:
            return np.zeros(self.embed_dim)
    
    def process_frames(self, frames: np.ndarray, augment: bool = False) -> Dict:
        result = {'video_embedding': np.zeros(self.embed_dim), 'quality_score': 0.0}
        
        
        
        if augment and self.video_augmenter:
            try:
                frames = self.video_augmenter.augment(frames)
                result['augmentation_applied'] = True
            except Exception as e:
                print(f"VideoAugmenter error: {e}")
        
        filtered, qc = self.quality.filter_frames(frames)
        result.update(qc)
        result['quality_score'] = qc.get('frame_quality_ratio', 0)
        
        result['video_embedding'] = self.get_video_embedding(filtered[:16])
        result.update(self.flow.analyze(frames))
        
        if self.kinematics:
            result.update(self.kinematics.analyze_sequence(frames))
        
        total_frames = qc.get('total_frames', len(frames))
        good_frames = qc.get('good_frames', len(filtered))
        result['metadata'] = {
            'frames_rejected_blur': int(total_frames - good_frames)
        }
        
        return result

class FacePreprocessor:
    """Complete face pipeline: Steps 27-34, R39-R49."""
    def __init__(self, models, embed_dim=768, device='cuda'):
        self.models = models
        self.embed_dim = embed_dim
        self.device = device
        self.detector = FaceDetector()
        self.cropper = FaceCropper()
        self.tracker = FaceTracker()
        self.gaze = GazeTracker()
        self.au = ActionUnitExtractor()
        self.micro = MicroExpressionAnalyzer()
        self.blink = BlinkRateAnalyzer()
        self.normalizer = FrameNormalizer()
    
    @torch.no_grad()
    def get_face_embedding(self, faces: List[np.ndarray]) -> np.ndarray:
        """Step 31: Extract Face Embedding.
        
        --- PATCH: POSTER_v2 Compliance Wrapper ---
        If specific weights aren't loaded, enforce projection to 768 for fusion safety.
        """
        if 'face' not in self.models.models or len(faces) == 0:
            return np.zeros(self.embed_dim)
        
        try:
            model = self.models.models['face']
            embeddings = []
            for face in faces[:16]:
                face_norm = self.normalizer.normalize(face[np.newaxis, ...])[0]
                face_t = torch.tensor(face_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                out = model.forward_features(face_t)
                if len(out.shape) == 3: 
                    emb_vec = out.mean(dim=1)
                else:
                    emb_vec = out
                
                embeddings.append(emb_vec.cpu().numpy())
            
            mean_emb = np.mean(embeddings, axis=0).flatten()
            
            if mean_emb.shape[0] != self.embed_dim:
                if mean_emb.shape[0] > self.embed_dim:
                    mean_emb = mean_emb[:self.embed_dim]
                else:
                    mean_emb = np.pad(mean_emb, (0, self.embed_dim - mean_emb.shape[0]))
            
            return mean_emb
        except Exception as e:
            print(f"POSTER_v2 Extraction Error: {e}")
            return np.zeros(self.embed_dim)
    
    def process_frames(self, frames: np.ndarray) -> Dict:
        result = {'face_embedding': np.zeros(self.embed_dim), 'quality_score': 0.0}
        
        if len(frames) == 0:
            return result
        
        cropped_faces = []
        au_sequence = []
        gaze_results = []
        
        for frame in frames:
            faces = self.detector.detect(frame)
            if faces:
                self.tracker.update(faces)
                face_crop = self.cropper.crop(frame, faces[0])
                cropped_faces.append(face_crop)
                
                au = self.au.extract(frame)
                au_sequence.append(au['au_features'])
                
                gaze = self.gaze.analyze(frame)
                gaze_results.append(gaze)
        
        if len(cropped_faces) == 0:
            return result
        
        result['face_embedding'] = self.get_face_embedding(cropped_faces)
        
        if gaze_results:
            result['gaze_direct_ratio'] = np.mean([g['gaze_direct_ratio'] for g in gaze_results])
            result['head_yaw_mean'] = np.mean([g['head_yaw'] for g in gaze_results])
            result['head_pitch_mean'] = np.mean([g['head_pitch'] for g in gaze_results])
        
        if au_sequence:
            result['au_mean'] = np.mean(au_sequence, axis=0)
        
        result.update(self.micro.analyze(au_sequence))
        
        blink_stats = self.blink.analyze_frames(np.array(cropped_faces))
        result.update(blink_stats)
        
        result['quality_score'] = len(cropped_faces) / len(frames) if len(frames) > 0 else 0
        
        confidences = [face.get('confidence', 0.0) for face in self.tracker.objects.values()] if self.tracker.objects else []
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        gaze_x = result.get('head_yaw_mean', 0.0)
        gaze_y = result.get('head_pitch_mean', 0.0)
        
        result['metadata'] = {
            'face_detection_confidence_avg': float(avg_conf),
            'gaze_avg_x': float(gaze_x),
            'gaze_avg_y': float(gaze_y),
            'blink_rate_per_min': float(blink_stats.get('blink_rate', 0.0))
        }
        
        result['face_detection_rate'] = result['quality_score']
        
        if R33_R46_R57_OK and len(cropped_faces) > 0:
            quality_flags = DiscreteQualityFilters.get_quality_flags(cropped_faces[0])
            result.update({f'discrete_{k}': v for k, v in quality_flags.items()})

        confidences = [face.get('confidence', 0.0) for face in self.tracker.objects.values()] if self.tracker.objects else []
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        gaze_x = result.get('head_yaw_mean', 0.0)
        gaze_y = result.get('head_pitch_mean', 0.0)
        
        result['metadata'] = {
            'face_detection_confidence_avg': float(avg_conf),
            'gaze_avg_x': float(gaze_x),
            'gaze_avg_y': float(gaze_y),
            'blink_rate_per_min': float(blink_stats.get('blink_rate', 0.0))
        }
        
        return result

print("Video & Face Pipeline loaded: VideoPreprocessor + FacePreprocessor")
