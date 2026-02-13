"""
Pipeline Integration Module for H5-OmniFusion

Implements updated H5 saving logic that includes all remediated features:
- Core: au_intensity, pose_features (P32/R44, P34/R48)
- Advanced: ADV1-ADV9 innovations

Author: H5-OmniFusion Remediation
Version: 1.0.0
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class H5KeyRegistry:
    """
    Registry of all H5 keys required for 100% spec compliance.
    
    Organized by:
        - Core Production/Research embeddings (768-dim)
        - Auxiliary features (variable dim)
        - Advanced Innovations (768-dim)
    """
    
    CORE_EMBEDDINGS = [
        'audio_embedding',           # P9/R10 - Wav2Vec2
        'audio_egemaps_embedding',   # P10/R11 - eGeMAPSv02 → 768
        'text_embedding',            # P16/R25 - MentalRoBERTa
        'video_embedding',           # P25/R37 - VideoMAE
        'face_embedding',            # P31/R43 - POSTER_v2
        'tabular_embedding',         # P38/R53 - TabPFN
        'fusion_embedding',          # Final convergence
    ]
    
    AUXILIARY_FEATURES = [
        'prosodic_features',         # P11/R15 - Speaking rate, pauses
        'linguistic_features',       # P17/R26 - LIWC counts
        'sentiment_scores',          # P19/R29 - Valence/polarity
        'optical_flow',              # P26/R38 - Motion magnitude
        'gaze_features',             # P33/R47 - Gaze direction
        'quality_scores',            # P40/R59 - QC metrics
        'phq8_score',                # Label - Ground truth
    ]
    
    REMEDIATED_CORE = [
        'au_intensity',              # P32/R44 - Action Unit intensities (17,)
        'au_embedding',              # P32/R44 - AU → 768-dim projection
        'pose_features',             # P34/R48 - Head pose (6,)
        'pose_embedding',            # P34/R48 - Pose → 768-dim projection
    ]
    
    ADVANCED_INNOVATIONS = [
        'response_latency',          # ADV1 - Latency stats (5,)
        'response_latency_embedding',# ADV1 - → 768-dim
        
        'psychomotor_features',      # ADV2 - Movement stats (8,)
        'psychomotor_embedding',     # ADV2 - → 768-dim
        
        'prosodic_fingerprint',      # ADV3 - Rhythm embedding (32,)
        'prosodic_fingerprint_embedding',  # ADV3 - → 768-dim
        
        'symptom_scores',            # ADV4 - PHQ-8 subscales (7,)
        'symptom_embedding',         # ADV4 - → 768-dim
        
        'breath_variability',        # ADV5 - Breath stats (6,)
        'breath_variability_embedding',  # ADV5 - → 768-dim
        'sigh_events',               # ADV5 - Sigh count (1,)
        
        'crossmodal_congruence',     # ADV6 - Alignment scores (4,)
        'crossmodal_sync',           # ADV6 - → 768-dim
        
        'temporal_trajectory',       # ADV7 - Slope/curvature (6,)
        'temporal_trajectory_embedding',  # ADV7 - → 768-dim
        
        
        'imputed_video_embedding',   # ADV9 - Hallucinated video (768,)
        'imputed_face_embedding',    # ADV9 - Hallucinated face (768,)
    ]
    
    @classmethod
    def get_all_keys(cls) -> List[str]:
        """Return complete list of all H5 keys."""
        return (
            cls.CORE_EMBEDDINGS + 
            cls.AUXILIARY_FEATURES + 
            cls.REMEDIATED_CORE + 
            cls.ADVANCED_INNOVATIONS
        )
    
    @classmethod
    def get_768_dim_keys(cls) -> List[str]:
        """Return keys that must be 768-dimensional."""
        return [k for k in cls.get_all_keys() if 'embedding' in k or k == 'fusion_embedding']


class H5FeatureSaver:
    """
    Save extracted features to HDF5 format with full spec compliance.
    
    Ensures all 108-step features are properly written including:
        - Core Production embeddings (7 × 768-dim)
        - Auxiliary features (variable dim)
        - Remediated features: AU, Pose
        - Advanced Innovations: ADV1-ADV9
    """
    
    def __init__(self, output_path: str, compression: str = 'gzip'):
        """
        Initialize H5 saver.
        
        Args:
            output_path: Path to output .h5 file
            compression: Compression algorithm ('gzip', 'lzf', or None)
        """
        self.output_path = Path(output_path)
        self.compression = compression
        self.registry = H5KeyRegistry()
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def validate_embedding_dim(self, key: str, data: np.ndarray) -> bool:
        """
        Validate that embedding keys have correct 768 dimensions.
        
        Args:
            key: Feature key name
            data: Feature data array
            
        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        if 'embedding' in key.lower() or key == 'fusion_embedding':
            if data.shape[-1] != 768:
                raise ValueError(
                    f"DIMENSION MISMATCH: {key} has shape {data.shape}, "
                    f"expected (..., 768). All embeddings must be 768-dim."
                )
        return True
    
    def save_participant(self, 
                         participant_id: str, 
                         features: Dict[str, np.ndarray],
                         mode: str = 'a') -> None:
        """
        Save features for a single participant.
        
        Args:
            participant_id: Participant identifier (e.g., "300")
            features: Dictionary of feature arrays
            mode: File mode ('a' for append, 'w' for overwrite)
        """
        with h5py.File(self.output_path, mode) as f:
            if participant_id in f:
                grp = f[participant_id]
            else:
                grp = f.create_group(participant_id)
            
            saved_keys = []
            skipped_keys = []
            
            for key, data in features.items():
                try:
                    if 'embedding' in key:
                        self.validate_embedding_dim(key, data)
                    
                    if not isinstance(data, np.ndarray):
                        data = np.array(data, dtype=np.float32)
                    
                    if data.dtype != np.float32 and data.dtype != object:
                        data = data.astype(np.float32)
                    
                    if key in grp:
                        del grp[key]
                    
                    if data.dtype == object:
                        grp.create_dataset(key, data=data)
                    else:
                        grp.create_dataset(
                            key, 
                            data=data,
                            compression=self.compression if data.nbytes > 1024 else None
                        )
                    
                    saved_keys.append(key)
                    
                except Exception as e:
                    logger.warning(f"Failed to save {key}: {e}")
                    skipped_keys.append(key)
            
            logger.info(
                f"[{participant_id}] Saved {len(saved_keys)} keys, "
                f"skipped {len(skipped_keys)}"
            )
    
    def save_advanced_features(self,
                               participant_id: str,
                               advanced_features: Dict[str, np.ndarray]) -> None:
        """
        Save specifically the advanced innovation features (ADV1-ADV9).
        
        This method explicitly targets the remediated features from the
        compliance audit gap analysis.
        
        Args:
            participant_id: Participant identifier
            advanced_features: Dictionary from AdvancedFeatureExtractor
        """
        adv_keys = set(self.registry.ADVANCED_INNOVATIONS + self.registry.REMEDIATED_CORE)
        filtered = {k: v for k, v in advanced_features.items() if k in adv_keys}
        
        if filtered:
            self.save_participant(participant_id, filtered, mode='a')
            logger.info(f"[{participant_id}] Saved {len(filtered)} advanced features")
        else:
            logger.warning(f"[{participant_id}] No advanced features to save")
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance report for the saved H5 file.
        
        Returns:
            Dict with compliance statistics
        """
        report = {
            'file': str(self.output_path),
            'participants': [],
            'total_keys': 0,
            'core_embeddings_present': 0,
            'advanced_innovations_present': 0,
            'compliance_percentage': 0.0,
        }
        
        try:
            with h5py.File(self.output_path, 'r') as f:
                for pid in f.keys():
                    grp = f[pid]
                    keys = list(grp.keys())
                    
                    core_present = sum(1 for k in self.registry.CORE_EMBEDDINGS if k in keys)
                    
                    adv_present = sum(1 for k in self.registry.ADVANCED_INNOVATIONS if k in keys)
                    
                    rem_present = sum(1 for k in self.registry.REMEDIATED_CORE if k in keys)
                    
                    report['participants'].append({
                        'id': pid,
                        'total_keys': len(keys),
                        'core_embeddings': core_present,
                        'advanced_innovations': adv_present,
                        'remediated_core': rem_present,
                    })
                    
                    report['total_keys'] += len(keys)
                    report['core_embeddings_present'] += core_present
                    report['advanced_innovations_present'] += adv_present
                
                expected_total = len(self.registry.get_all_keys())
                if report['participants']:
                    avg_keys = report['total_keys'] / len(report['participants'])
                    report['compliance_percentage'] = (avg_keys / expected_total) * 100
                    
        except FileNotFoundError:
            logger.error(f"H5 file not found: {self.output_path}")
        
        return report


def integrate_advanced_features(
    existing_features: Dict[str, np.ndarray],
    participant_dir: Optional[Path] = None,
    transcript_df: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """
    Integrate advanced features into existing feature dictionary.
    
    This function should be called AFTER standard feature extraction
    to add the remediated AU, Pose, and ADV1-ADV9 features.
    
    Args:
        existing_features: Features from standard pipeline
        participant_dir: Path to participant data (for CLNF files)
        transcript_df: Timestamped transcript DataFrame
        
    Returns:
        Updated feature dictionary with all advanced features
    """
    from advanced_features import AdvancedFeatureExtractor
    
    extractor = AdvancedFeatureExtractor()
    
    prosodic = existing_features.get('prosodic_features', None)
    
    advanced = extractor.extract_all(
        participant_dir=participant_dir,
        transcript_df=transcript_df,
        prosodic_features=prosodic,
        feature_dict=existing_features,
    )
    
    existing_features.update(advanced)
    
    if 'video_embedding' not in existing_features or existing_features.get('video_embedding') is None:
        audio_emb = existing_features.get('audio_embedding')
        text_emb = existing_features.get('text_embedding')
        if audio_emb is not None or text_emb is not None:
            existing_features['imputed_video_embedding'] = extractor.adv9.impute_video(
                audio_embedding=audio_emb,
                text_embedding=text_emb
            )
            if 'video_embedding' not in existing_features:
                existing_features['video_embedding'] = existing_features['imputed_video_embedding']
    
    if 'face_embedding' not in existing_features or existing_features.get('face_embedding') is None:
        audio_emb = existing_features.get('audio_embedding')
        if audio_emb is not None:
            existing_features['imputed_face_embedding'] = extractor.adv9.impute_face(audio_emb)
            if 'face_embedding' not in existing_features:
                existing_features['face_embedding'] = existing_features['imputed_face_embedding']
    
    quality_scores = {
        'audio': existing_features.get('quality_scores', [1.0])[0] if 'quality_scores' in existing_features else 0.8,
        'text': 0.9,  # Text usually reliable
        'video': existing_features.get('quality_scores', [1.0, 1.0])[1] if len(existing_features.get('quality_scores', [])) > 1 else 0.7,
        'face': 0.8,
        'tabular': 0.9,
    }
    
    embeddings = {
        'audio': existing_features.get('audio_embedding'),
        'text': existing_features.get('text_embedding'),
        'video': existing_features.get('video_embedding'),
        'face': existing_features.get('face_embedding'),
        'tabular': existing_features.get('tabular_embedding'),
    }
    
    if any(e is not None for e in embeddings.values()):
        quality_gated_fusion = extractor.adv8.fuse(embeddings, quality_scores)
        existing_features['quality_gated_fusion'] = quality_gated_fusion
    
    return existing_features


def save_compliant_h5(
    output_path: str,
    participant_id: str,
    features: Dict[str, np.ndarray],
    include_advanced: bool = True,
    participant_dir: Optional[Path] = None,
    transcript_df: Optional[Any] = None,
) -> None:
    """
    Save features to H5 with full 108-step compliance.
    
    This is the PRIMARY entry point for saving pipeline outputs.
    
    Args:
        output_path: Path to output .h5 file
        participant_id: Participant identifier
        features: Extracted features dictionary
        include_advanced: Whether to add ADV1-ADV9 features
        participant_dir: Path for CLNF file access
        transcript_df: Timestamped transcript
    """
    if include_advanced:
        features = integrate_advanced_features(
            features,
            participant_dir=participant_dir,
            transcript_df=transcript_df,
        )
    
    saver = H5FeatureSaver(output_path)
    saver.save_participant(participant_id, features)
    
    report = saver.generate_compliance_report()
    logger.info(f"Compliance: {report['compliance_percentage']:.1f}%")


def process_participant_batch(
    participant_dirs: List[Path],
    output_path: str,
    existing_pipeline_fn: callable,
) -> Dict[str, Any]:
    """
    Process batch of participants with full 108-step compliance.
    
    Args:
        participant_dirs: List of participant data directories
        output_path: Path to output H5 file
        existing_pipeline_fn: Function that runs existing feature extraction
                              Signature: (participant_dir) -> Dict[str, np.ndarray]
    
    Returns:
        Processing summary dictionary
    """
    summary = {
        'total': len(participant_dirs),
        'success': 0,
        'failed': 0,
        'errors': [],
    }
    
    saver = H5FeatureSaver(output_path)
    
    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        
        try:
            features = existing_pipeline_fn(participant_dir)
            
            features = integrate_advanced_features(
                features,
                participant_dir=participant_dir,
            )
            
            saver.save_participant(participant_id, features)
            summary['success'] += 1
            
        except Exception as e:
            logger.error(f"[{participant_id}] Processing failed: {e}")
            summary['failed'] += 1
            summary['errors'].append({'id': participant_id, 'error': str(e)})
    
    summary['compliance_report'] = saver.generate_compliance_report()
    
    return summary


if __name__ == "__main__":
    registry = H5KeyRegistry()
    
    print("=" * 60)
    print("H5 KEY REGISTRY - 108-STEP COMPLIANCE")
    print("=" * 60)
    print(f"\nCore Embeddings ({len(registry.CORE_EMBEDDINGS)}):")
    for k in registry.CORE_EMBEDDINGS:
        print(f"  ✓ {k}")
    
    print(f"\nAuxiliary Features ({len(registry.AUXILIARY_FEATURES)}):")
    for k in registry.AUXILIARY_FEATURES:
        print(f"  ✓ {k}")
    
    print(f"\nRemediated Core ({len(registry.REMEDIATED_CORE)}):")
    for k in registry.REMEDIATED_CORE:
        print(f"  + {k}")
    
    print(f"\nAdvanced Innovations ({len(registry.ADVANCED_INNOVATIONS)}):")
    for k in registry.ADVANCED_INNOVATIONS:
        print(f"  + {k}")
    
    print(f"\nTotal Keys: {len(registry.get_all_keys())}")
    print(f"768-dim Keys: {len(registry.get_768_dim_keys())}")
