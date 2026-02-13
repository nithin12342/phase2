"""
H5-OmniFusion Main Pipeline
Integrates all modules for complete preprocessing and feature extraction.
"""
import os
import h5py
import numpy as np
import json
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .config import Config, CFG
from .utils import DEVICE, clear_memory, robust_transcript_load
from .model_loader import MODEL_LOADER

from .audio.preprocessing import AudioPreprocessor
from .audio.feature_extraction import AudioFeatureExtractor
from .audio.diarization import UnifiedDiarizer
from .audio.advanced import AdvancedAudioExtractor

from .text.preprocessing import TextPreprocessor
from .text.feature_extraction import TextFeatureExtractor
from .text.chinese_support import LanguageDetector

from .video.preprocessing import VideoPreprocessor
from .video.feature_extraction import VideoFeatureExtractor

from .face.detection import FacePreprocessor
from .face.feature_extraction import FaceFeatureExtractor

from .tabular.preprocessing import TabularPreprocessor, QualityScorer

from .fusion.cross_modal import CrossModalProcessor
from .fusion.advanced import AdvancedFusion


class H5OmniFusionPipeline:
    """
    Main pipeline integrating all 40 Production, 59 Research, and 9 Advanced steps.
    
    Outputs all embeddings as 768-dim for fusion compatibility.
    Supports both DAIC-WOZ (English) and EATD-Corpus (Mandarin Chinese).
    """
    
    def __init__(self, config: Config = None, device: str = 'auto'):
        self.config = config or CFG
        self.device = DEVICE if device == 'auto' else device
        
        self.lang_detector = LanguageDetector()
        
        self.audio_preprocessor = AudioPreprocessor(self.config)
        self.audio_extractor = AudioFeatureExtractor()
        self.diarizer = UnifiedDiarizer()
        self.audio_advanced = AdvancedAudioExtractor()
        
        self.text_preprocessor = TextPreprocessor()
        self.text_extractor = TextFeatureExtractor()
        
        self.video_preprocessor = VideoPreprocessor(self.config)
        self.video_extractor = VideoFeatureExtractor()
        
        self.face_preprocessor = FacePreprocessor(self.config)
        self.face_extractor = FaceFeatureExtractor()
        
        self.tabular_preprocessor = TabularPreprocessor()
        self.quality_scorer = QualityScorer(self.config)
        
        self.cross_modal = CrossModalProcessor()
        self.advanced_fusion = AdvancedFusion()
        
        self.checkpoint_path = os.path.join(self.config.OUTPUT_PATH, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        print(f"H5OmniFusionPipeline initialized on {self.device}")
    
    def process_participant(self, 
                           audio_path: str = None,
                           video_path: str = None,
                           transcript_path: str = None,
                           participant_id: str = 'unknown') -> Dict:
        """
        Process a single participant's data through all modalities.
        
        Args:
            audio_path: Path to audio file
            video_path: Path to video file (None for EATD-Corpus)
            transcript_path: Path to transcript
            participant_id: Participant identifier
            
        Returns:
            Dict with all embeddings and features
        """
        print(f"\n{'='*50}")
        print(f"Processing participant: {participant_id}")
        print(f"{'='*50}")
        
        results = {
            'participant_id': participant_id,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'errors': []
        }
        
        language = 'english'
        if transcript_path:
            text, _ = robust_transcript_load(transcript_path)
            language = self.lang_detector.detect(text)
        results['language'] = language
        
        print("\n[1/5] Processing audio...")
        audio_results = self._process_audio(audio_path, transcript_path)
        results['audio'] = audio_results
        
        print("\n[2/5] Processing text...")
        text_results = self._process_text(transcript_path, language)
        results['text'] = text_results
        
        print("\n[3/5] Processing video...")
        if video_path and os.path.exists(video_path):
            video_results = self._process_video(video_path)
        else:
            video_results = {'success': False, 'video_embedding': np.zeros(768)}
            print("  No video available (EATD-Corpus mode)")
        results['video'] = video_results
        
        print("\n[4/5] Processing face...")
        if video_path and os.path.exists(video_path):
            face_results = self._process_face(video_path)
        else:
            face_results = {'success': False, 'face_embedding': np.zeros(768)}
            print("  No video for face analysis")
        results['face'] = face_results
        
        print("\n[5/5] Fusing modalities...")
        fusion_results = self._fuse_modalities(results)
        results['fusion'] = fusion_results
        
        results['quality'] = self._calculate_quality(results)
        
        clear_memory()
        
        print(f"\n✓ Participant {participant_id} complete")
        return results
    
    def _process_audio(self, audio_path: str, transcript_path: str = None) -> Dict:
        """Process audio modality."""
        if not audio_path or not os.path.exists(audio_path):
            return {'success': False, 'wav2vec2_embedding': np.zeros(768)}
        
        try:
            preproc = self.audio_preprocessor.process(audio_path)
            if not preproc['success']:
                return {'success': False, 'wav2vec2_embedding': np.zeros(768)}
            
            diarization = self.diarizer.diarize(
                preproc['waveform'], preproc['sr'], transcript_path
            )
            participant_audio = diarization['participant_audio']
            
            features = self.audio_extractor.extract_all(
                participant_audio, preproc['sr'], audio_path
            )
            
            transcript_df = None
            if transcript_path:
                from .audio.diarization import TranscriptDiarizer
                transcript_df = TranscriptDiarizer().parse_transcript(transcript_path)
            
            advanced = self.audio_advanced.extract_all(
                participant_audio, preproc['sr'], transcript_df
            )
            
            return {
                'success': True,
                'wav2vec2_embedding': features['wav2vec2_embedding'],
                'egemaps_embedding': features['egemaps_embedding'],
                'pitch': features['pitch'],
                'jitter_shimmer': features['jitter_shimmer'],
                'formants': features['formants'],
                'pauses': features['pauses'],
                'speaking_rate': features['speaking_rate'],
                'response_latency': advanced['response_latency'],
                'prosody_fingerprint': advanced['prosody_fingerprint'],
                'sigh_detection': advanced['sigh_detection'],
                'diarization': diarization['turn_info'],
                'vad_ratio': preproc['vad_ratio'],
                'duration_sec': preproc['duration_sec']
            }
            
        except Exception as e:
            print(f"  Audio error: {e}")
            return {'success': False, 'wav2vec2_embedding': np.zeros(768), 'error': str(e)}
    
    def _process_text(self, transcript_path: str, language: str = 'english') -> Dict:
        """Process text modality."""
        if not transcript_path or not os.path.exists(transcript_path):
            return {'success': False, 'text_embedding': np.zeros(768)}
        
        try:
            self.text_preprocessor.language = language
            preproc = self.text_preprocessor.process(transcript_path=transcript_path)
            
            if not preproc['success']:
                return {'success': False, 'text_embedding': np.zeros(768)}
            
            self.text_extractor.language = language
            features = self.text_extractor.extract_all(
                preproc['text_for_embedding'],
                turn_info=preproc.get('disfluency_info'),
                language=language
            )
            
            return {
                'success': True,
                'text_embedding': features['text_embedding'],
                'linguistic': features['linguistic'],
                'lexical': features['lexical'],
                'readability': features['readability'],
                'sentiment': features['sentiment'],
                'emotion': features['emotion'],
                'dynamics': features['dynamics'],
                'disfluency': preproc['disfluency_info'],
                'annotations': preproc['annotation_info']
            }
            
        except Exception as e:
            print(f"  Text error: {e}")
            return {'success': False, 'text_embedding': np.zeros(768), 'error': str(e)}
    
    def _process_video(self, video_path: str) -> Dict:
        """Process video modality."""
        try:
            preproc = self.video_preprocessor.process(video_path)
            
            if not preproc['success']:
                return {'success': False, 'video_embedding': np.zeros(768)}
            
            features = self.video_extractor.extract_all(
                preproc['frames'],
                preproc['normalized_frames']
            )
            
            return {
                'success': True,
                'video_embedding': features['video_embedding'],
                'optical_flow': features['optical_flow'],
                'motion_trajectory': features['motion_trajectory'],
                'quality_info': preproc['quality_info']
            }
            
        except Exception as e:
            print(f"  Video error: {e}")
            return {'success': False, 'video_embedding': np.zeros(768), 'error': str(e)}
    
    def _process_face(self, video_path: str) -> Dict:
        """Process face modality."""
        try:
            preproc = self.video_preprocessor.process(video_path, filter_quality=False, normalize=False)
            
            if not preproc['success']:
                return {'success': False, 'face_embedding': np.zeros(768)}
            
            face_preproc = self.face_preprocessor.process(preproc['frames'])
            
            features = self.face_extractor.extract_all(
                face_preproc['face_crops'],
                fps=self.config.TARGET_FPS
            )
            
            return {
                'success': True,
                'face_embedding': features['face_embedding'],
                'action_units': features['action_units'],
                'blink': features['blink'],
                'gaze': features['gaze'],
                'head_pose': features['head_pose'],
                'detection_rate': face_preproc['detection_rate']
            }
            
        except Exception as e:
            print(f"  Face error: {e}")
            return {'success': False, 'face_embedding': np.zeros(768), 'error': str(e)}
    
    def _fuse_modalities(self, results: Dict) -> Dict:
        """Fuse all modality embeddings."""
        try:
            embeddings = {}
            
            if results['audio'].get('success'):
                embeddings['audio'] = results['audio']['wav2vec2_embedding']
            if results['text'].get('success'):
                embeddings['text'] = results['text']['text_embedding']
            if results['video'].get('success'):
                embeddings['video'] = results['video']['video_embedding']
            if results['face'].get('success'):
                embeddings['face'] = results['face']['face_embedding']
            
            qualities = {
                'audio': results.get('quality', {}).get('audio_quality', 0.5),
                'text': 1.0 if results['text'].get('success') else 0.0,
                'video': results.get('quality', {}).get('video_quality', 0.5),
                'face': results['face'].get('detection_rate', 0.5) if results['face'].get('success') else 0.0
            }
            
            fusion = self.cross_modal.process(embeddings, qualities, impute_missing=True)
            
            text_sentiment = results['text'].get('sentiment', {}).get('compound', 0)
            audio_valence = 0  # Would come from audio emotion model
            
            advanced = self.advanced_fusion.extract_all(
                text_sentiment=text_sentiment,
                audio_valence=audio_valence
            )
            
            return {
                'fused_embedding': fusion['fused_embedding_quality'],
                'concat_embedding': fusion['fused_embedding_concat'],
                'imputed': fusion['imputed'],
                'congruence': advanced['cross_modal_congruence'],
                'symptom_risk': advanced['symptom_clustering']
            }
            
        except Exception as e:
            print(f"  Fusion error: {e}")
            return {'fused_embedding': np.zeros(768), 'error': str(e)}
    
    def _calculate_quality(self, results: Dict) -> Dict:
        """Calculate quality scores."""
        audio_quality = 0.5
        video_quality = 0.5
        
        if results['audio'].get('success'):
            vad_ratio = results['audio'].get('vad_ratio', 0.5)
            audio_quality = min(1.0, vad_ratio / 0.4)
        
        if results['video'].get('success'):
            quality_info = results['video'].get('quality_info', {})
            video_quality = quality_info.get('pass_rate', 0.5)
        
        return self.quality_scorer.score_multimodal(
            audio_quality, video_quality,
            text_length=results['text'].get('disfluency', {}).get('word_count', 0)
        )
    
    def save_to_h5(self, results: Dict, output_path: str):
        """Save results to HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            f.attrs['participant_id'] = results['participant_id']
            f.attrs['timestamp'] = results['timestamp']
            f.attrs['language'] = results['language']
            
            emb = f.create_group('embeddings')
            
            if results['audio'].get('success'):
                emb.create_dataset('audio_wav2vec2', data=results['audio']['wav2vec2_embedding'])
                emb.create_dataset('audio_egemaps', data=results['audio']['egemaps_embedding'])
            
            if results['text'].get('success'):
                emb.create_dataset('text', data=results['text']['text_embedding'])
            
            if results['video'].get('success'):
                emb.create_dataset('video', data=results['video']['video_embedding'])
            
            if results['face'].get('success'):
                emb.create_dataset('face', data=results['face']['face_embedding'])
            
            if 'fused_embedding' in results.get('fusion', {}):
                emb.create_dataset('fused', data=results['fusion']['fused_embedding'])
            
            f.attrs['overall_quality'] = results.get('quality', {}).get('overall_quality', 0)
        
        print(f"Saved to {output_path}")
    
    def process_dataset(self, dataset: str = 'daic_woz', split: str = 'all',
                        output_dir: str = None) -> List[Dict]:
        """
        Process entire dataset.
        
        Args:
            dataset: 'daic_woz' or 'eatd_corpus'
            split: 'train', 'dev', 'test', or 'all'
            output_dir: Directory to save H5 files
        """
        output_dir = output_dir or self.config.OUTPUT_PATH
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        if dataset == 'daic_woz':
            participants = self._get_daic_woz_participants(split)
        else:
            participants = self._get_eatd_participants()
        
        print(f"Processing {len(participants)} participants from {dataset}")
        
        for i, (pid, paths) in enumerate(participants):
            print(f"\n[{i+1}/{len(participants)}] ", end='')
            
            try:
                result = self.process_participant(
                    audio_path=paths.get('audio'),
                    video_path=paths.get('video'),
                    transcript_path=paths.get('transcript'),
                    participant_id=pid
                )
                
                h5_path = os.path.join(output_dir, f"{pid}.h5")
                self.save_to_h5(result, h5_path)
                
                all_results.append(result)
                
                if (i + 1) % self.config.CHECKPOINT_FREQUENCY == 0:
                    self._save_checkpoint(all_results, i + 1)
                
            except Exception as e:
                print(f"Error processing {pid}: {e}")
                all_results.append({'participant_id': pid, 'success': False, 'error': str(e)})
        
        return all_results
    
    def _get_daic_woz_participants(self, split: str) -> List[tuple]:
        """Get DAIC-WOZ participant paths."""
        return []
    
    def _get_eatd_participants(self) -> List[tuple]:
        """Get EATD-Corpus participant paths."""
        return []
    
    def _save_checkpoint(self, results: List[Dict], count: int):
        """Save checkpoint."""
        checkpoint_file = os.path.join(self.checkpoint_path, f'checkpoint_{count}.json')
        summary = [{'pid': r['participant_id'], 'success': r['success']} for r in results]
        with open(checkpoint_file, 'w') as f:
            json.dump(summary, f)
        print(f"Checkpoint saved: {checkpoint_file}")


def run_pipeline(audio_path: str = None, video_path: str = None,
                 transcript_path: str = None, participant_id: str = 'test') -> Dict:
    """Quick function to process a single participant."""
    pipeline = H5OmniFusionPipeline()
    return pipeline.process_participant(audio_path, video_path, transcript_path, participant_id)
