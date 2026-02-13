"""
Advanced Fusion Features Module
Implements ADV2, ADV4, ADV6, ADV7 from H5-OmniFusion specification.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import DEVICE


class CrossModalCongruence:
    """
    Measure congruence between text sentiment and audio valence.
    ADV6: Detects "Smiling Depression" where verbal contradicts paralinguistic.
    
    congruence = 1 - |text_sentiment - audio_valence|
    """
    
    def calculate(self, text_sentiment: float, audio_valence: float) -> Dict:
        """
        Calculate cross-modal congruence.
        
        Args:
            text_sentiment: Text sentiment score [-1, 1]
            audio_valence: Audio emotional valence [-1, 1]
            
        Returns:
            Dict with congruence_score and interpretation
        """
        text_norm = (text_sentiment + 1) / 2
        audio_norm = (audio_valence + 1) / 2
        
        congruence = 1 - abs(text_norm - audio_norm)
        
        if congruence > 0.8:
            interpretation = 'congruent'
        elif congruence > 0.5:
            interpretation = 'mild_incongruence'
        else:
            interpretation = 'masking_detected'
        
        return {
            'congruence_score': congruence,
            'text_sentiment': text_sentiment,
            'audio_valence': audio_valence,
            'interpretation': interpretation,
            'masking_risk': 1 - congruence
        }
    
    def batch_calculate(self, text_sentiments: List[float], 
                       audio_valences: List[float]) -> Dict:
        """Calculate congruence over multiple segments."""
        if not text_sentiments or not audio_valences:
            return {'mean_congruence': 0.5, 'congruence_scores': []}
        
        min_len = min(len(text_sentiments), len(audio_valences))
        
        scores = []
        for i in range(min_len):
            result = self.calculate(text_sentiments[i], audio_valences[i])
            scores.append(result['congruence_score'])
        
        return {
            'mean_congruence': np.mean(scores),
            'congruence_std': np.std(scores),
            'congruence_scores': scores,
            'masking_segments': sum(1 for s in scores if s < 0.5)
        }


class TemporalTrajectoryEncoder:
    """
    Model feature trajectories over session thirds.
    ADV7: Quantifies fatigue progression and mood instability.
    """
    
    def encode(self, feature_sequence: np.ndarray) -> Dict:
        """
        Calculate slope and curvature of feature trajectory.
        
        Args:
            feature_sequence: Time-ordered feature values
            
        Returns:
            Dict with slope, curvature, trajectory stats
        """
        if len(feature_sequence) < 3:
            return self._default_result()
        
        thirds = np.array_split(feature_sequence, 3)
        third_means = [np.mean(t) for t in thirds]
        
        x = np.arange(len(feature_sequence))
        slope, intercept = np.polyfit(x, feature_sequence, 1)
        
        if len(feature_sequence) >= 3:
            coeffs = np.polyfit(x, feature_sequence, 2)
            curvature = coeffs[0]  # Second-degree coefficient
        else:
            curvature = 0
        
        variability = np.std(feature_sequence)
        
        return {
            'slope': float(slope),
            'curvature': float(curvature),
            'third_means': third_means,
            'start_value': float(feature_sequence[0]),
            'end_value': float(feature_sequence[-1]),
            'change': float(feature_sequence[-1] - feature_sequence[0]),
            'variability': float(variability),
            'trend': 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable')
        }
    
    def encode_multimodal(self, features: Dict[str, np.ndarray]) -> Dict:
        """
        Encode trajectories for multiple features.
        
        Returns:
            Dict mapping feature name to trajectory encoding
        """
        trajectories = {}
        
        for name, values in features.items():
            if len(values) > 0:
                trajectories[name] = self.encode(values)
        
        return trajectories
    
    def _default_result(self) -> Dict:
        return {
            'slope': 0, 'curvature': 0, 'third_means': [0, 0, 0],
            'start_value': 0, 'end_value': 0, 'change': 0,
            'variability': 0, 'trend': 'unknown'
        }


class SymptomClusterer:
    """
    Map features to PHQ-8 symptom clusters.
    ADV4.
    
    PHQ-8 Subscales:
    - Anhedonia (items 1, 2)
    - Sleep (item 3)
    - Fatigue (item 4)
    - Appetite (item 5)
    - Guilt (item 6)
    - Concentration (item 7)
    - Psychomotor (item 8)
    """
    
    SYMPTOM_FEATURES = {
        'anhedonia': ['positive_emotion_ratio', 'smile_rate'],
        'sleep': ['fatigue_vocal_markers', 'pause_ratio'],
        'fatigue': ['speaking_rate', 'energy_mean', 'motion_magnitude'],
        'appetite': [],  # Hard to detect from A/V
        'guilt': ['first_person_ratio', 'negative_emotion_ratio'],
        'concentration': ['gaze_aversion_ratio', 'response_latency_mean'],
        'psychomotor': ['motion_magnitude', 'head_movement_variability']
    }
    
    def cluster(self, features: Dict[str, float]) -> Dict:
        """
        Generate symptom-specific predictions.
        
        Args:
            features: Dict of computed features
            
        Returns:
            Dict with per-symptom risk scores
        """
        symptom_scores = {}
        
        for symptom, relevant_features in self.SYMPTOM_FEATURES.items():
            if not relevant_features:
                symptom_scores[symptom] = 0.5  # Unknown
                continue
            
            values = []
            for feat_name in relevant_features:
                if feat_name in features:
                    values.append(features[feat_name])
            
            if values:
                symptom_scores[symptom] = np.clip(np.mean(values), 0, 1)
            else:
                symptom_scores[symptom] = 0.5
        
        return {
            'symptom_scores': symptom_scores,
            'highest_risk': max(symptom_scores, key=symptom_scores.get),
            'mean_risk': np.mean(list(symptom_scores.values()))
        }


class KinematicsAnalyzer:
    """
    Analyze body kinematics and postural slumping.
    ADV2: Detects physical withdrawal characteristic of depression.
    """
    
    def analyze_slumping(self, head_poses: List[Dict]) -> Dict:
        """
        Detect postural slumping from head pose trajectory.
        
        Args:
            head_poses: List of {yaw, pitch, roll} dicts
            
        Returns:
            Dict with slump detection and trajectory
        """
        if not head_poses:
            return self._default_result()
        
        pitches = [p.get('pitch', 0) for p in head_poses]
        
        thirds = np.array_split(pitches, 3)
        pitch_trajectory = [np.mean(t) for t in thirds]
        
        slump_slope = np.polyfit(range(len(pitches)), pitches, 1)[0]
        
        pitch_velocities = np.diff(pitches)
        mean_velocity = np.mean(np.abs(pitch_velocities))
        
        return {
            'slump_slope': float(slump_slope),
            'slump_detected': slump_slope > 0.5,  # Threshold
            'pitch_trajectory': pitch_trajectory,
            'head_velocity_mean': float(mean_velocity),
            'head_velocity_std': float(np.std(pitch_velocities)),
            'final_vs_initial_pitch': float(pitches[-1] - pitches[0]) if pitches else 0
        }
    
    def analyze_motion_reduction(self, motion_magnitudes: np.ndarray) -> Dict:
        """
        Detect psychomotor retardation via motion reduction.
        
        Returns:
            Dict with motion trajectory and reduction detection
        """
        if len(motion_magnitudes) < 3:
            return {'motion_reduction': False, 'motion_slope': 0}
        
        slope = np.polyfit(range(len(motion_magnitudes)), motion_magnitudes, 1)[0]
        
        thirds = np.array_split(motion_magnitudes, 3)
        motion_trajectory = [np.mean(t) for t in thirds]
        
        return {
            'motion_reduction': slope < -0.1,
            'motion_slope': float(slope),
            'motion_trajectory': motion_trajectory,
            'initial_motion': float(motion_magnitudes[0]),
            'final_motion': float(motion_magnitudes[-1])
        }
    
    def _default_result(self) -> Dict:
        return {
            'slump_slope': 0, 'slump_detected': False,
            'pitch_trajectory': [0, 0, 0], 'head_velocity_mean': 0,
            'head_velocity_std': 0, 'final_vs_initial_pitch': 0
        }


class AdvancedFusion:
    """Unified advanced fusion features (ADV2, ADV4, ADV6, ADV7)."""
    
    def __init__(self):
        self.congruence = CrossModalCongruence()
        self.trajectory = TemporalTrajectoryEncoder()
        self.symptom = SymptomClusterer()
        self.kinematics = KinematicsAnalyzer()
    
    def extract_all(self, 
                    text_sentiment: float = 0,
                    audio_valence: float = 0,
                    feature_sequences: Dict[str, np.ndarray] = None,
                    all_features: Dict[str, float] = None,
                    head_poses: List[Dict] = None,
                    motion_magnitudes: np.ndarray = None) -> Dict:
        """
        Extract all advanced fusion features.
        
        Returns:
            Dict with congruence, trajectory, symptom, kinematics features
        """
        congruence = self.congruence.calculate(text_sentiment, audio_valence)
        
        trajectories = {}
        if feature_sequences:
            trajectories = self.trajectory.encode_multimodal(feature_sequences)
        
        symptom_risk = {}
        if all_features:
            symptom_risk = self.symptom.cluster(all_features)
        
        slumping = self.kinematics._default_result()
        motion = {'motion_reduction': False}
        
        if head_poses:
            slumping = self.kinematics.analyze_slumping(head_poses)
        if motion_magnitudes is not None and len(motion_magnitudes) > 0:
            motion = self.kinematics.analyze_motion_reduction(motion_magnitudes)
        
        return {
            'cross_modal_congruence': congruence,
            'temporal_trajectories': trajectories,
            'symptom_clustering': symptom_risk,
            'slumping_analysis': slumping,
            'motion_analysis': motion
        }
