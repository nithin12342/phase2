"""
Face Detection Module
Implements Steps 27-30 and R39-R42 from H5-OmniFusion specification.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import CV2_AVAILABLE, MEDIAPIPE_AVAILABLE

if CV2_AVAILABLE:
    import cv2

if MEDIAPIPE_AVAILABLE:
    import mediapipe as mp


class FaceDetector:
    """
    Detect faces using MediaPipe or OpenCV cascade.
    Steps 27, R39.
    """
    
    def __init__(self, min_confidence: float = 0.8):
        self.min_confidence = min_confidence
        self.detector = None
        
        if MEDIAPIPE_AVAILABLE:
            self.detector = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=min_confidence
            )
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame.
        
        Returns:
            List of dicts with 'bbox', 'confidence', 'landmarks'
        """
        if self.detector is None:
            return self._opencv_detect(frame)
        
        try:
            if frame.shape[-1] == 3 and len(frame.shape) == 3:
                rgb_frame = frame if frame.dtype == np.uint8 else (frame * 255).astype(np.uint8)
            else:
                return []
            
            results = self.detector.process(rgb_frame)
            
            if not results.detections:
                return []
            
            h, w = frame.shape[:2]
            faces = []
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                landmarks = {}
                for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                    landmarks[idx] = (int(landmark.x * w), int(landmark.y * h))
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': detection.score[0],
                    'landmarks': landmarks
                })
            
            return faces
            
        except Exception as e:
            return self._opencv_detect(frame)
    
    def _opencv_detect(self, frame: np.ndarray) -> List[Dict]:
        """Fallback OpenCV cascade detection."""
        if not CV2_AVAILABLE:
            return []
        
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            return [{'bbox': tuple(f), 'confidence': 0.9, 'landmarks': {}} for f in faces]
        except:
            return []
    
    def detect_batch(self, frames: np.ndarray) -> List[List[Dict]]:
        """Detect faces in multiple frames."""
        return [self.detect(frame) for frame in frames]


class LandmarkAligner:
    """
    Align faces to canonical pose using landmarks.
    Steps 28, R40.
    """
    
    def __init__(self):
        self.face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
    
    def align(self, frame: np.ndarray, face_bbox: Tuple = None) -> Tuple[np.ndarray, Dict]:
        """
        Align face using 5-point landmarks.
        
        Returns:
            (aligned_face, alignment_info)
        """
        if self.face_mesh is None or not CV2_AVAILABLE:
            return frame, {'aligned': False}
        
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                face_region = frame[max(0,y):y+h, max(0,x):x+w]
            else:
                face_region = frame
            
            results = self.face_mesh.process(face_region)
            
            if not results.multi_face_landmarks:
                return frame, {'aligned': False}
            
            landmarks = results.multi_face_landmarks[0]
            h, w = face_region.shape[:2]
            
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            nose = landmarks.landmark[1]
            
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            angle = np.degrees(np.arctan2(dy, dx))
            
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(face_region, matrix, (w, h))
            
            return aligned, {
                'aligned': True,
                'rotation_angle': angle,
                'eye_distance': np.sqrt(dx**2 + dy**2) * w
            }
            
        except Exception as e:
            return frame, {'aligned': False, 'error': str(e)}


class FaceCropper:
    """
    Crop and resize face region with margin.
    Steps 29, R41.
    """
    
    def __init__(self, margin: float = 0.2, output_size: Tuple[int, int] = (224, 224)):
        self.margin = margin
        self.output_size = output_size
    
    def crop(self, frame: np.ndarray, bbox: Tuple) -> np.ndarray:
        """
        Crop face with margin and resize.
        
        Args:
            frame: Full frame
            bbox: (x, y, width, height)
            
        Returns:
            Cropped face (224, 224, 3)
        """
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        
        margin_x = int(w * self.margin)
        margin_y = int(h * self.margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(W, x + w + margin_x)
        y2 = min(H, y + h + margin_y)
        
        face = frame[y1:y2, x1:x2]
        
        if CV2_AVAILABLE and face.size > 0:
            face = cv2.resize(face, self.output_size)
        else:
            face = np.zeros((*self.output_size, 3), dtype=np.uint8)
        
        return face
    
    def crop_batch(self, frames: np.ndarray, detections: List[List[Dict]]) -> List[np.ndarray]:
        """Crop primary face from each frame."""
        crops = []
        for frame, dets in zip(frames, detections):
            if dets:
                crops.append(self.crop(frame, dets[0]['bbox']))
            else:
                crops.append(np.zeros((*self.output_size, 3), dtype=np.uint8))
        return crops


class FaceTracker:
    """
    Track face identity across frames using centroid tracking.
    Steps 30, R42.
    """
    
    def __init__(self, max_disappeared: int = 5):
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.objects = {}  # id -> centroid
        self.disappeared = {}  # id -> count
    
    def update(self, detections: List[Dict]) -> Dict[int, Tuple]:
        """
        Update tracker with new detections.
        
        Returns:
            Dict mapping object_id -> centroid
        """
        input_centroids = []
        for det in detections:
            x, y, w, h = det['bbox']
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))
        
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self._register(centroid)
        
        elif len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
        
        else:
            self._match_centroids(input_centroids)
        
        return self.objects.copy()
    
    def _register(self, centroid: Tuple):
        """Register new object."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def _deregister(self, obj_id: int):
        """Remove object."""
        del self.objects[obj_id]
        del self.disappeared[obj_id]
    
    def _match_centroids(self, input_centroids: List[Tuple]):
        """Match input centroids to existing objects using distance."""
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())
        
        used_inputs = set()
        
        for obj_id, obj_centroid in zip(obj_ids, obj_centroids):
            min_dist = float('inf')
            min_idx = -1
            
            for idx, input_centroid in enumerate(input_centroids):
                if idx in used_inputs:
                    continue
                
                dist = np.sqrt((obj_centroid[0] - input_centroid[0])**2 + 
                              (obj_centroid[1] - input_centroid[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            
            if min_idx >= 0 and min_dist < 100:  # Threshold
                self.objects[obj_id] = input_centroids[min_idx]
                self.disappeared[obj_id] = 0
                used_inputs.add(min_idx)
            else:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
        
        for idx, centroid in enumerate(input_centroids):
            if idx not in used_inputs:
                self._register(centroid)
    
    def reset(self):
        """Reset tracker state."""
        self.objects.clear()
        self.disappeared.clear()
        self.next_id = 0


class FacePreprocessor:
    """Unified face detection and preprocessing (Steps 27-30, R39-R42)."""
    
    def __init__(self, config=None):
        cfg = config or CFG
        self.detector = FaceDetector(cfg.FACE_CONFIDENCE_THRESHOLD)
        self.aligner = LandmarkAligner()
        self.cropper = FaceCropper(cfg.FACE_MARGIN, cfg.FRAME_SIZE)
        self.tracker = FaceTracker()
    
    def process(self, frames: np.ndarray) -> Dict:
        """
        Process frames for face analysis.
        
        Returns:
            Dict with face_crops, detections, tracking_info
        """
        all_detections = []
        face_crops = []
        tracking_history = []
        
        self.tracker.reset()
        
        for frame in frames:
            dets = self.detector.detect(frame)
            all_detections.append(dets)
            
            tracking = self.tracker.update(dets)
            tracking_history.append(tracking)
            
            if dets:
                crop = self.cropper.crop(frame, dets[0]['bbox'])
            else:
                crop = np.zeros((224, 224, 3), dtype=np.uint8)
            face_crops.append(crop)
        
        face_crops = np.array(face_crops)
        
        frames_with_face = sum(1 for d in all_detections if d)
        detection_rate = frames_with_face / len(frames) if len(frames) > 0 else 0
        
        return {
            'face_crops': face_crops,
            'detections': all_detections,
            'tracking': tracking_history,
            'detection_rate': detection_rate,
            'frames_with_face': frames_with_face
        }
