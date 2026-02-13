"""
H5-OmniFusion Cross-Modal and Fusion Enhancements
Steps 35-40 + Advanced Innovations (ADV1-ADV9)
"""

import numpy as np
from typing import Dict, List, Optional, Any

class FeatureImputer:
    """Handle missing values per documentation spec.
    
    Enhanced with save/load for deployment: Load pre-computed medians/modes from training set.
    """
    
    def __init__(self, stats_path: Optional[str] = None):
        self.medians = {}
        self.modes = {}
        if stats_path and os.path.exists(stats_path):
            self.load_stats(stats_path)
    
    def fit(self, features_list: List[Dict[str, float]]) -> 'FeatureImputer':
        """Learn medians/modes from training data."""
        from collections import Counter
        
        all_values = {}
        for features in features_list:
            for k, v in features.items():
                if k not in all_values:
                    all_values[k] = []
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    all_values[k].append(v)
        
        for k, values in all_values.items():
            if len(values) == 0:
                continue
            if isinstance(values[0], (int, float)):
                self.medians[k] = float(np.median(values))
            else:
                self.modes[k] = Counter(values).most_common(1)[0][0]
        
        return self
    
    def impute(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Impute missing values."""
        result = features.copy()
        for k, v in result.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                if k in self.medians:
                    result[k] = self.medians[k]
                elif k in self.modes:
                    result[k] = self.modes[k]
                else:
                    result[k] = 0
        return result
    
    def save_stats(self, path: str):
        """Save imputation statistics to JSON for deployment."""
        import json
        with open(path, 'w') as f:
            json.dump({'medians': self.medians, 'modes': self.modes}, f)
    
    def load_stats(self, path: str):
        """Load pre-computed imputation statistics from training."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.medians = data.get('medians', {})
            self.modes = data.get('modes', {})
        print(f"[FeatureImputer] Loaded stats for {len(self.medians)} medians, {len(self.modes)} modes from {path}")


class CategoricalEncoder:
    """Encode categorical variables."""
    
    GENDER_MAP = {'male': 1, 'm': 1, 'female': 0, 'f': 0}
    
    def encode(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Encode categorical features to numeric."""
        result = {}
        
        for k, v in features.items():
            if k.lower() == 'gender':
                result['gender_encoded'] = self.GENDER_MAP.get(str(v).lower(), 0.5)
            elif isinstance(v, str):
                result[f'{k}_encoded'] = hash(v) % 100 / 100
            else:
                result[k] = v
        
        return result


class PHQ8SubScoreAnalyzer:
    """Map features to PHQ-8 clinical sub-scales."""
    
    SOMATIC_ITEMS = [3, 4, 5]  # Sleep, Energy, Appetite
    COGNITIVE_ITEMS = [1, 2, 6, 7, 8]  # Mood, Anhedonia, Guilt, Concentration, Psychomotor
    
    FEATURE_PHQ_MAP = {
        'sleep_words': 3,  # PHQ item 3: Sleep problems
        'energy_trend': 4,  # PHQ item 4: Fatigue
        'anhedonia': 2,  # PHQ item 2: Anhedonia
        'first_person_singular': 6,  # PHQ item 6: Guilt (self-focus)
        'response_latency_mean': 8,  # PHQ item 8: Psychomotor
        'pause_ratio': 8,  # PHQ item 8: Psychomotor
        'negative_emotion': 1,  # PHQ item 1: Depressed mood
        'cognitive': 7,  # PHQ item 7: Concentration
    }
    
    def compute_subscales(self, phq8_scores: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute somatic vs cognitive sub-scores from PHQ-8 item responses."""
        if phq8_scores is None or len(phq8_scores) < 8:
            return {'somatic_score': 0, 'cognitive_score': 0, 'somatic_cognitive_ratio': 0}
        
        somatic = sum(phq8_scores[i-1] for i in self.SOMATIC_ITEMS)
        cognitive = sum(phq8_scores[i-1] for i in self.COGNITIVE_ITEMS)
        
        return {
            'somatic_score': float(somatic),
            'cognitive_score': float(cognitive),
            'somatic_cognitive_ratio': float(somatic / (cognitive + 1e-8))
        }
    
    def classify_depression_subtype(self, somatic: float, cognitive: float) -> str:
        """Classify depression subtype based on subscale dominance."""
        if somatic > cognitive * 1.5:
            return 'somatic_predominant'
        elif cognitive > somatic * 1.5:
            return 'cognitive_predominant'
        else:
            return 'mixed'


class QualityGatedFusion:
    """Dynamically weight modalities based on quality scores."""
    
    MODALITY_DIMS = {
        'audio': 768,
        'text': 768,
        'video': 768,
        'face': 768,
        'tabular': 768
    }
    
    def fuse(self, embeddings: Dict[str, np.ndarray], 
             quality_scores: Dict[str, float]) -> np.ndarray:
        """Fuse modality embeddings weighted by quality scores.
        
        Args:
            embeddings: Dict of modality -> 768-dim embedding
            quality_scores: Dict of modality -> quality score (0-1)
        
        Returns:
            768-dim fused embedding
        """
        weighted_sum = np.zeros(768, dtype=np.float32)
        weight_sum = 0
        
        for modality, emb in embeddings.items():
            if emb is not None and len(emb) == 768:
                q = quality_scores.get(modality, 0.5)
                q = max(0.01, min(1.0, q))  # Clamp to valid range
                weighted_sum += q * emb
                weight_sum += q
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        return np.zeros(768, dtype=np.float32)
    
    def compute_overall_quality(self, quality_scores: Dict[str, float]) -> float:
        """Compute overall sample quality from modality scores."""
        if not quality_scores:
            return 0.0
        return float(np.mean(list(quality_scores.values())))


class ModalityImputer:
    """Handle missing modalities gracefully."""
    
    def __init__(self):
        self.fallback_embeddings = {
            'video': np.zeros(768, dtype=np.float32),
            'face': np.zeros(768, dtype=np.float32)
        }
        self.modality_present = {}
    
    def impute(self, embeddings: Dict[str, np.ndarray], 
               available_modalities: List[str]) -> Dict[str, np.ndarray]:
        """Impute missing modality embeddings.
        
        Args:
            embeddings: Available embeddings
            available_modalities: List of available modality names
        """
        result = embeddings.copy()
        all_modalities = ['audio', 'text', 'video', 'face', 'tabular']
        
        for mod in all_modalities:
            if mod not in available_modalities or embeddings.get(mod) is None:
                result[mod] = self.fallback_embeddings.get(mod, np.zeros(768, dtype=np.float32))
                result[f'{mod}_is_imputed'] = True
            else:
                result[f'{mod}_is_imputed'] = False
        
        return result
    
    def get_imputation_mask(self, embeddings: Dict[str, Any]) -> Dict[str, bool]:
        """Get mask indicating which modalities were imputed."""
        return {k.replace('_is_imputed', ''): v 
                for k, v in embeddings.items() if k.endswith('_is_imputed')}


class CrossModalCongruenceScorer:
    """Detect cross-modal alignment for masking/smiling depression detection."""
    
    def score(self, audio_feats: Dict, text_feats: Dict, face_feats: Dict) -> Dict[str, float]:
        """Compute cross-modal congruence scores.
        
        Detects incongruence between verbal content and paralinguistic cues.
        """
        scores = {}
        
        sad_audio = (audio_feats.get('f0_cv', 0) < 0.2 or 
                     audio_feats.get('pause_ratio', 0) > 0.3 or
                     audio_feats.get('speaking_rate', 10) < 3)
        
        sad_text = (text_feats.get('negative_emotion', 0) > 0.02 or 
                    text_feats.get('sentiment_compound', 0) < -0.3)
        
        sad_face = (face_feats.get('sadness_proxy', 0) > 0.2 or 
                    face_feats.get('gaze_aversion_mean', 0) > 0.3 or
                    face_feats.get('au_depression_indicator', 0) > 0.5)
        
        all_sad = sad_audio and sad_text and sad_face
        none_sad = not sad_audio and not sad_text and not sad_face
        
        scores['congruent_depression'] = 1.0 if all_sad else 0.0
        scores['congruent_non_depression'] = 1.0 if none_sad else 0.0
        scores['incongruent'] = 0.0 if (all_sad or none_sad) else 1.0
        
        scores['audio_text_match'] = 1.0 if sad_audio == sad_text else 0.0
        scores['text_face_match'] = 1.0 if sad_text == sad_face else 0.0
        scores['audio_face_match'] = 1.0 if sad_audio == sad_face else 0.0
        
        scores['overall_congruence'] = (scores['audio_text_match'] + 
                                        scores['text_face_match'] + 
                                        scores['audio_face_match']) / 3
        
        verbal_positive = text_feats.get('sentiment_compound', 0) > 0.3
        cues_negative = sad_audio or sad_face
        scores['masking_detected'] = 1.0 if verbal_positive and cues_negative else 0.0
        
        return scores


class TemporalTrajectoryEncoder:
    """Encode feature changes over session duration."""
    
    def encode(self, time_series: List[float], name_prefix: str = '') -> Dict[str, float]:
        """Compute slope, curvature, and volatility of features over time.
        
        Args:
            time_series: Feature values over time (e.g., session quintiles)
            name_prefix: Prefix for feature names
        """
        if len(time_series) < 3:
            return {
                f'{name_prefix}slope': 0.0,
                f'{name_prefix}curvature': 0.0,
                f'{name_prefix}volatility': 0.0,
                f'{name_prefix}early_late_diff': 0.0
            }
        
        y = np.array(time_series)
        x = np.arange(len(y))
        
        try:
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
        except:
            slope = 0.0
        
        try:
            if len(y) >= 3:
                coeffs2 = np.polyfit(x, y, 2)
                curvature = coeffs2[0]  # Quadratic coefficient
            else:
                curvature = 0.0
        except:
            curvature = 0.0
        
        return {
            f'{name_prefix}slope': float(slope),
            f'{name_prefix}curvature': float(curvature),
            f'{name_prefix}volatility': float(np.std(y)),
            f'{name_prefix}early_late_diff': float(y[-1] - y[0])
        }


class ResponseLatencyExtractor:
    """Extract response latency as psychomotor retardation biomarker."""
    
    SLOW_THRESHOLD = 2.0  # seconds
    
    def extract(self, turns: List[Dict]) -> Dict[str, float]:
        """Extract response latency features from turn data.
        
        Args:
            turns: List of {'speaker': str, 'start': float, 'end': float, ...}
        """
        latencies = []
        prev_ellie_end = None
        
        for turn in turns:
            speaker = turn.get('speaker', '').lower()
            
            if 'ellie' in speaker:
                prev_ellie_end = turn.get('end', turn.get('stop_time'))
            elif 'participant' in speaker and prev_ellie_end is not None:
                start = turn.get('start', turn.get('start_time'))
                if start is not None:
                    latency = start - prev_ellie_end
                    if 0 < latency < 15:  # Valid range
                        latencies.append(latency)
                prev_ellie_end = None
        
        if not latencies:
            return {
                'response_latency_mean': 0,
                'response_latency_std': 0,
                'response_latency_max': 0,
                'response_latency_median': 0,
                'slow_response_ratio': 0
            }
        
        return {
            'response_latency_mean': float(np.mean(latencies)),
            'response_latency_std': float(np.std(latencies)),
            'response_latency_max': float(np.max(latencies)),
            'response_latency_median': float(np.median(latencies)),
            'slow_response_ratio': float(sum(1 for l in latencies if l > self.SLOW_THRESHOLD) / len(latencies))
        }


class ClinicalClusterer:
    """Map extracted biomarkers to PHQ-8 clinical sub-scales.
    
    Enhanced symptom clustering for clinical interpretation.
    Reference: implementation_plan.md ADV4 specification.
    """
    
    def score_symptoms(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Calculate PHQ-8 cluster scores from biomarkers.
        
        Args:
            feats: Dict of extracted features from all modalities
            
        Returns:
            Dict with PHQ-8 subscale estimates
        """
        pos_emo = feats.get('t_positive_emo_pct', feats.get('positive_emo_pct', 0.1))
        au12 = feats.get('v_AU12_mean', feats.get('AU12_mean', 0.5))
        compound = feats.get('t_compound', feats.get('compound', 0))
        
        anhedonia = 1.0 - (pos_emo * 2 + min(au12 / 10, 1) + (compound + 1) / 2) / 4
        anhedonia = max(0, min(1, anhedonia))
        
        pause_ratio = feats.get('a_pause_ratio', feats.get('pause_ratio', 0.3))
        pause_mean = feats.get('a_pause_mean', feats.get('pause_mean', 500)) / 1000
        slump = abs(feats.get('v_slump_delta', feats.get('slump_delta', 0)))
        speaking_rate = feats.get('a_speaking_rate', feats.get('speaking_rate', 3))
        
        fatigue = (pause_ratio + min(pause_mean, 1) + slump * 5 + (1 - min(speaking_rate / 5, 1))) / 4
        fatigue = max(0, min(1, fatigue))
        
        jitter = feats.get('a_jitter', feats.get('jitter', 0.01)) * 50
        blink_rate = feats.get('v_blink_rate', feats.get('blink_rate', 15)) / 30
        head_vel = feats.get('v_head_velocity', feats.get('head_velocity', 0.5))
        
        anxiety = (min(jitter, 1) + min(blink_rate, 1) + min(head_vel, 1)) / 3
        anxiety = max(0, min(1, anxiety))
        
        ttr = feats.get('t_ttr', feats.get('ttr', 0.5))
        first_person = feats.get('t_first_person_pct', feats.get('first_person_pct', 0.1))
        absolutist = feats.get('t_absolutist_pct', feats.get('absolutist_pct', 0.05))
        
        cognitive = ((1 - ttr) + first_person * 3 + absolutist * 5) / 3
        cognitive = max(0, min(1, cognitive))
        
        neg_emo = feats.get('t_negative_emo_pct', feats.get('negative_emo_pct', 0.1))
        neg_compound = max(0, -compound) if compound < 0 else 0
        sigh_count = feats.get('a_sigh_count', feats.get('sigh_count', 0)) / 5
        
        negative_affect = (neg_emo * 3 + neg_compound + min(sigh_count, 1)) / 3
        negative_affect = max(0, min(1, negative_affect))
        
        return {
            'PHQ8_Anhedonia': round(anhedonia * 6, 2),      # Scale 0-6 (items 1-2)
            'PHQ8_Fatigue': round(fatigue * 6, 2),          # Scale 0-6 (items 3-4)
            'PHQ8_Anxiety': round(anxiety * 3, 2),          # Scale 0-3
            'PHQ8_Cognitive': round(cognitive * 6, 2),      # Scale 0-6 (items 5-8)
            'PHQ8_NegativeAffect': round(negative_affect * 3, 2),
            'PHQ8_TotalEstimate': round((anhedonia + fatigue + cognitive + negative_affect) * 6, 2)
        }


class WordLevelAligner:
    """Word-level alignment and temporal synchronization (R55).
    
    Merged from final_4_enhancements.py AlignmentEngine class.
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def align_words(self, waveform: np.ndarray, transcript_df) -> List[Dict]:
        """Map acoustic features to word timestamps.
        
        Args:
            waveform: Audio waveform array
            transcript_df: DataFrame with start_time, stop_time, value columns
            
        Returns:
            List of word-level feature dicts
        """
        try:
            import librosa
        except ImportError:
            return []
        
        word_features = []
        if transcript_df is None or len(transcript_df) < 1:
            return word_features
        
        for _, row in transcript_df.iterrows():
            try:
                start_s = float(row.iloc[0]) if len(row) > 0 else 0
                end_s = float(row.iloc[1]) if len(row) > 1 else start_s + 0.1
                start_idx, end_idx = int(start_s * self.sr), int(end_s * self.sr)
                
                if start_idx >= len(waveform) or end_idx > len(waveform):
                    continue
                segment = waveform[start_idx:end_idx]
                if len(segment) < 100:
                    continue
                
                energy = np.mean(segment ** 2)
                pitch = 0
                try:
                    pitches, _ = librosa.piptrack(y=segment.astype(float), sr=self.sr)
                    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
                except:
                    pass
                
                word_features.append({
                    'start': start_s,
                    'end': end_s,
                    'duration': end_s - start_s,
                    'energy': float(energy),
                    'pitch': float(pitch),
                    'text': str(row.iloc[3]) if len(row) > 3 else ''
                })
            except:
                continue
        
        return word_features
    
    def aggregate_word_features(self, word_features: List[Dict]) -> Dict[str, float]:
        """Aggregate word-level features to session-level."""
        if not word_features:
            return {}
        
        energies = [w['energy'] for w in word_features]
        pitches = [w['pitch'] for w in word_features if w['pitch'] > 0]
        durations = [w['duration'] for w in word_features]
        
        return {
            'word_energy_mean': float(np.mean(energies)) if energies else 0,
            'word_energy_var': float(np.var(energies)) if energies else 0,
            'word_pitch_mean': float(np.mean(pitches)) if pitches else 0,
            'word_duration_mean': float(np.mean(durations)) if durations else 0,
            'word_count_aligned': len(word_features)
        }
    
    def synchronize_modalities(self, audio_emb_seq: np.ndarray, video_emb_seq: np.ndarray, 
                                target_hz: int = 30) -> tuple:
        """Interpolate embeddings to common temporal grid for early fusion.
        
        Args:
            audio_emb_seq: Audio embedding sequence (T1, D)
            video_emb_seq: Video embedding sequence (T2, D)
            target_hz: Target temporal resolution
            
        Returns:
            Tuple of synchronized (audio_emb, video_emb)
        """
        try:
            import scipy.ndimage as ndimage
        except ImportError:
            return audio_emb_seq, video_emb_seq
        
        if audio_emb_seq is None or video_emb_seq is None:
            return None, None
        
        if len(audio_emb_seq.shape) == 1:
            audio_emb_seq = audio_emb_seq.reshape(1, -1)
        if len(video_emb_seq.shape) == 1:
            video_emb_seq = video_emb_seq.reshape(1, -1)
        
        target_len = max(len(audio_emb_seq), len(video_emb_seq))
        
        if len(audio_emb_seq) != target_len:
            zoom_factor = target_len / len(audio_emb_seq)
            audio_emb_seq = ndimage.zoom(audio_emb_seq, (zoom_factor, 1), order=1)
        
        if len(video_emb_seq) != target_len:
            zoom_factor = target_len / len(video_emb_seq)
            video_emb_seq = ndimage.zoom(video_emb_seq, (zoom_factor, 1), order=1)
        
        return audio_emb_seq, video_emb_seq
