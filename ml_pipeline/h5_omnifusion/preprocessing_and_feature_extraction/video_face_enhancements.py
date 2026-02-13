"""
H5-OmniFusion Video/Face Pipeline Enhancements
Steps 21-34 from 40-Step Production Pipeline
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class VideoQualityFilter:
    """Filter frames based on blur and brightness thresholds."""
    
    BLUR_THRESHOLD = 50  # Laplacian variance minimum
    BRIGHTNESS_MIN = 80
    BRIGHTNESS_MAX = 180
    
    def is_quality_frame(self, frame: np.ndarray) -> bool:
        """Check if frame meets quality thresholds."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        return blur >= self.BLUR_THRESHOLD and self.BRIGHTNESS_MIN <= brightness <= self.BRIGHTNESS_MAX
    
    def filter_frames(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """Filter frames and return quality metrics."""
        quality_frames = [f for f in frames if self.is_quality_frame(f)]
        
        metrics = {
            'total_frames': len(frames),
            'quality_frames': len(quality_frames),
            'quality_ratio': len(quality_frames) / (len(frames) + 1e-8)
        }
        
        if len(quality_frames) < 4:
            return frames, metrics
        return quality_frames, metrics


class OpticalFlowAnalyzer:
    """Compute motion magnitude for psychomotor retardation detection."""
    
    def extract(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Extract optical flow features between consecutive frames."""
        if len(frames) < 2:
            return {'flow_mean': 0, 'flow_std': 0, 'flow_max': 0, 'motion_energy': 0}
        
        magnitudes = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            magnitudes.append(np.mean(mag))
            prev_gray = curr_gray
        
        return {
            'flow_mean': float(np.mean(magnitudes)),
            'flow_std': float(np.std(magnitudes)),
            'flow_max': float(np.max(magnitudes)),
            'motion_energy': float(np.sum(magnitudes))
        }


class SimpleFaceTracker:
    """Simple face tracking using IoU matching."""
    
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0
    
    def _iou(self, box1, box2) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter + 1e-8)
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, Tuple]]:
        """Update tracks with new detections. Returns list of (track_id, bbox)."""
        results = []
        
        for det in detections:
            best_iou = 0
            best_track = None
            
            for track in self.tracks:
                iou = self._iou(det, track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            
            if best_iou >= self.iou_threshold and best_track:
                best_track['bbox'] = det
                best_track['age'] = 0
                results.append((best_track['id'], det))
            else:
                new_track = {'id': self.next_id, 'bbox': det, 'age': 0}
                self.tracks.append(new_track)
                results.append((self.next_id, det))
                self.next_id += 1
        
        self.tracks = [t for t in self.tracks if t['age'] < 5]
        for t in self.tracks:
            t['age'] += 1
        
        return results


class GazeCategorizer:
    """Categorize gaze direction into Direct/Indirect/Averted."""
    
    DIRECT_MAX = 10  # degrees
    INDIRECT_MAX = 15  # degrees
    
    def categorize(self, gaze_angles: List[float]) -> Dict[str, float]:
        """Categorize gaze angles into three categories."""
        if not gaze_angles:
            return {'direct_ratio': 0, 'indirect_ratio': 0, 'averted_ratio': 0}
        
        direct = sum(1 for a in gaze_angles if abs(a) <= self.DIRECT_MAX)
        indirect = sum(1 for a in gaze_angles if self.DIRECT_MAX < abs(a) <= self.INDIRECT_MAX)
        averted = sum(1 for a in gaze_angles if abs(a) > self.INDIRECT_MAX)
        total = len(gaze_angles)
        
        return {
            'direct_ratio': direct / total,
            'indirect_ratio': indirect / total,
            'averted_ratio': averted / total,
            'gaze_mean_angle': float(np.mean(np.abs(gaze_angles))),
            'gaze_std_angle': float(np.std(gaze_angles))
        }


class MicroExpressionAnalyzer:
    """Analyze onset and duration of facial movements."""
    
    def analyze(self, au_time_series: Dict[str, List[float]]) -> Dict[str, float]:
        """Analyze temporal dynamics of Action Units.
        
        Args:
            au_time_series: Dict mapping AU name to list of intensities over time
        """
        features = {}
        
        for au_name, values in au_time_series.items():
            if len(values) < 3:
                continue
            
            values = np.array(values)
            
            diff = np.diff(values)
            onsets = np.where(diff > np.std(diff) * 2)[0]
            
            above_mean = values > np.mean(values)
            
            features[f'{au_name}_onset_count'] = len(onsets)
            features[f'{au_name}_duration_ratio'] = float(np.mean(above_mean))
            features[f'{au_name}_variability'] = float(np.std(values))
        
        return features


class VideoQualityChecker:
    """Check video quality against documented thresholds."""
    
    BRIGHTNESS_MIN = 80
    BRIGHTNESS_MAX = 180
    BLUR_MIN = 50
    FACE_CONFIDENCE_MIN = 0.8
    
    def check_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Check single frame quality."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        brightness = np.mean(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        brightness_pass = self.BRIGHTNESS_MIN <= brightness <= self.BRIGHTNESS_MAX
        blur_pass = blur >= self.BLUR_MIN
        
        return {
            'brightness': float(brightness),
            'brightness_pass': brightness_pass,
            'blur_score': float(blur),
            'blur_pass': blur_pass,
            'quality_pass': brightness_pass and blur_pass
        }


class DepressionAUMapper:
    """Map Action Units to depression-relevant categories."""
    
    DEPRESSION_AUS = {
        'AU01': {'name': 'Inner Brow Raiser', 'association': 'Sadness', 'direction': 'positive'},
        'AU04': {'name': 'Brow Lowerer', 'association': 'Anger/Frustration', 'direction': 'positive'},
        'AU12': {'name': 'Lip Corner Puller', 'association': 'Happiness', 'direction': 'negative'},  # Reduced in depression
        'AU15': {'name': 'Lip Corner Depressor', 'association': 'Sadness', 'direction': 'positive'},
    }
    
    def compute_depression_score(self, au_values: Dict[str, float]) -> Dict[str, float]:
        """Compute depression-relevant AU composite scores."""
        sadness_score = 0
        happiness_score = 0
        
        for au, info in self.DEPRESSION_AUS.items():
            intensity = au_values.get(au, 0)
            if info['association'] == 'Sadness':
                sadness_score += intensity
            elif info['association'] == 'Happiness':
                happiness_score += intensity
        
        return {
            'au_sadness_composite': float(sadness_score),
            'au_happiness_composite': float(happiness_score),
            'au_depression_indicator': float(sadness_score - happiness_score)
        }


class HeadPoseAnalyzer:
    """Analyze head pose for depression indicators."""
    
    def analyze(self, yaw_series: List[float], pitch_series: List[float], roll_series: List[float]) -> Dict[str, float]:
        """Analyze head pose dynamics."""
        features = {}
        
        for name, series in [('yaw', yaw_series), ('pitch', pitch_series), ('roll', roll_series)]:
            if not series:
                features.update({f'head_{name}_mean': 0, f'head_{name}_std': 0, f'head_{name}_range': 0})
                continue
            
            arr = np.array(series)
            features[f'head_{name}_mean'] = float(np.mean(arr))
            features[f'head_{name}_std'] = float(np.std(arr))
            features[f'head_{name}_range'] = float(np.ptp(arr))
        
        if yaw_series and pitch_series:
            velocity = np.sqrt(np.diff(yaw_series)**2 + np.diff(pitch_series)**2)
            features['head_movement_velocity'] = float(np.mean(velocity)) if len(velocity) > 0 else 0
        
        return features


class KinematicsPostureAnalyzer:
    """Track body posture, slumping trends, and head movement velocity.
    
    Uses MediaPipe Pose for body keypoint detection.
    Depression biomarkers: physical withdrawal, fatigue, reduced movement.
    Reference: implementation_plan.md ADV2 specification.
    """
    
    def __init__(self):
        self.pose = None
        self.mp_available = False
        
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_available = True
            else:
                print("Warning: MediaPipe solutions API not available, kinematics disabled")
        except (ImportError, AttributeError, Exception) as e:
            print(f"Warning: MediaPipe not available ({e}), kinematics disabled")

        
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract posture metrics from single frame."""
        if not self.mp_available or self.pose is None:
            return None
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(frame)
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])
        
        try:
            nose = get_point(self.NOSE)
            l_shoulder = get_point(self.LEFT_SHOULDER)
            r_shoulder = get_point(self.RIGHT_SHOULDER)
            l_hip = get_point(self.LEFT_HIP)
            r_hip = get_point(self.RIGHT_HIP)
            
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            hip_mid = (l_hip + r_hip) / 2
            
            spine = shoulder_mid - hip_mid
            spine_angle = np.arctan2(spine[2], spine[1])  # Forward lean angle
            
            shoulder_drop = shoulder_mid[1] - hip_mid[1]  # Positive = shoulders above hips
            
            head_offset = nose - shoulder_mid
            head_forward = head_offset[2]  # Forward head posture
            
            shoulder_width = np.linalg.norm(r_shoulder[:2] - l_shoulder[:2])
            
            return {
                'spine_angle': float(np.degrees(spine_angle)),
                'shoulder_drop': float(shoulder_drop),
                'head_forward_offset': float(head_forward),
                'shoulder_width': float(shoulder_width),
                'nose_y': float(nose[1]),
                'detected': True
            }
        except Exception:
            return None
    
    def analyze_sequence(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analyze posture trends over frame sequence.
        
        Args:
            frames: List of video frames (RGB or BGR)
            
        Returns:
            Dict with posture analysis features
        """
        features = {
            'posture_slump_trend': 0.0,
            'head_movement_velocity': 0.0,
            'shoulder_contraction_trend': 0.0,
            'posture_variability': 0.0,
            'body_detected_ratio': 0.0
        }
        
        if len(frames) < 3:
            return features
        
        frame_data = []
        for frame in frames:
            data = self.analyze_frame(frame)
            if data and data.get('detected'):
                frame_data.append(data)
        
        if len(frame_data) < 3:
            features['body_detected_ratio'] = len(frame_data) / len(frames)
            return features
        
        spine_angles = [d['spine_angle'] for d in frame_data]
        shoulder_widths = [d['shoulder_width'] for d in frame_data]
        nose_positions = [d['nose_y'] for d in frame_data]
        
        n = len(spine_angles)
        if n >= 3:
            early = np.mean(spine_angles[:n//3])
            late = np.mean(spine_angles[2*n//3:])
            features['posture_slump_trend'] = float(late - early)
        
        if n >= 3:
            early = np.mean(shoulder_widths[:n//3])
            late = np.mean(shoulder_widths[2*n//3:])
            features['shoulder_contraction_trend'] = float(early - late)
        
        if len(nose_positions) >= 2:
            velocities = np.abs(np.diff(nose_positions))
            features['head_movement_velocity'] = float(np.mean(velocities))
        
        features['posture_variability'] = float(np.std(spine_angles))
        features['body_detected_ratio'] = len(frame_data) / len(frames)
        
        return features


class OpenFaceAUExtractor:
    """Parse AU intensities from pre-extracted OpenFace CLNF files.
    
    Implements P32/R44: Action Unit Detection (17 dims)
    """
    
    AU_COLUMNS = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
    ]
    
    def extract(self, clnf_au_path: str) -> Dict[str, np.ndarray]:
        """Extract AU intensities from *_CLNF_AUs.txt file.
        
        Args:
            clnf_au_path: Path to OpenFace AU output file
            
        Returns:
            Dict with 'au_intensity' (17,) array
        """
        try:
            import pandas as pd
            df = pd.read_csv(clnf_au_path, skipinitialspace=True)
            
            au_cols = [c for c in df.columns if c.strip() in self.AU_COLUMNS]
            if not au_cols:
                au_cols = [c for c in df.columns if '_r' in c and 'AU' in c]
            
            if au_cols:
                au_data = df[au_cols].values.astype(np.float32)
                au_intensity = np.mean(au_data, axis=0)  # Mean across frames
                return {
                    'au_intensity': au_intensity[:17] if len(au_intensity) >= 17 
                                   else np.pad(au_intensity, (0, 17 - len(au_intensity)))
                }
        except Exception as e:
            print(f"[WARNING] OpenFaceAUExtractor failed: {e}")
        
        return {'au_intensity': np.zeros(17, dtype=np.float32)}


class OpenFacePoseExtractor:
    """Parse head pose from pre-extracted OpenFace CLNF files.
    
    Implements P34/R48: Head Pose Estimation (6 dims: Tx, Ty, Tz, Rx, Ry, Rz)
    """
    
    POSE_COLUMNS = ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']
    
    def extract(self, clnf_pose_path: str) -> Dict[str, np.ndarray]:
        """Extract pose features from *_CLNF_pose.txt or combined CSV.
        
        Args:
            clnf_pose_path: Path to OpenFace pose output file
            
        Returns:
            Dict with 'pose_features' (6,) array [Tx, Ty, Tz, yaw, pitch, roll]
        """
        try:
            import pandas as pd
            df = pd.read_csv(clnf_pose_path, skipinitialspace=True)
            
            pose_cols = [c for c in df.columns if c.strip() in self.POSE_COLUMNS]
            if not pose_cols:
                pose_cols = [c for c in df.columns if 'pose_' in c.lower()]
            
            if pose_cols:
                pose_data = df[pose_cols].values.astype(np.float32)
                pose_mean = np.mean(pose_data, axis=0)
                return {
                    'pose_features': pose_mean[:6] if len(pose_mean) >= 6
                                    else np.pad(pose_mean, (0, 6 - len(pose_mean)))
                }
        except Exception as e:
            print(f"[WARNING] OpenFacePoseExtractor failed: {e}")
        
        return {'pose_features': np.zeros(6, dtype=np.float32)}
