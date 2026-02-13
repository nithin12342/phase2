"""
Audio Diarization Module
Implements Step 3, R4 - Speaker diarization for separating participant from interviewer.
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import robust_transcript_load


class TranscriptDiarizer:
    """
    Extract participant speech segments using transcript timestamps.
    Step 3, R4.
    
    DAIC-WOZ transcripts have format: "start_time end_time SPEAKER text"
    """
    
    def __init__(self, participant_labels: List[str] = None):
        self.participant_labels = participant_labels or [
            'participant', 'PARTICIPANT', 'Participant',
            'man', 'woman', 'user', 'subject'
        ]
        self.interviewer_labels = ['ellie', 'ELLIE', 'Ellie', 'interviewer', 'INTERVIEWER']
    
    def parse_transcript(self, transcript_path: str) -> pd.DataFrame:
        """
        Parse transcript with timestamps.
        
        Returns:
            DataFrame with columns: start, end, speaker, text
        """
        content, success = robust_transcript_load(transcript_path)
        if not success:
            return pd.DataFrame(columns=['start', 'end', 'speaker', 'text'])
        
        records = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'^(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\w+)\s+(.*)$', line)
            if match:
                records.append({
                    'start': float(match.group(1)),
                    'end': float(match.group(2)),
                    'speaker': match.group(3),
                    'text': match.group(4)
                })
        
        return pd.DataFrame(records)
    
    def get_participant_segments(self, transcript_path: str) -> List[Tuple[float, float]]:
        """
        Extract participant speech time ranges.
        
        Returns:
            List of (start_time, end_time) in seconds
        """
        df = self.parse_transcript(transcript_path)
        if df.empty:
            return []
        
        mask = df['speaker'].apply(
            lambda s: any(label.lower() in s.lower() for label in self.participant_labels)
        )
        participant_df = df[mask]
        
        return [(row['start'], row['end']) for _, row in participant_df.iterrows()]
    
    def extract_participant_audio(self, waveform: np.ndarray, sr: int,
                                   transcript_path: str) -> np.ndarray:
        """
        Extract only participant speech from audio.
        
        Returns:
            Concatenated participant audio segments
        """
        segments = self.get_participant_segments(transcript_path)
        
        if not segments:
            return waveform  # Return full audio if no segments found
        
        audio_segments = []
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if end_sample <= len(waveform):
                audio_segments.append(waveform[start_sample:end_sample])
        
        if audio_segments:
            return np.concatenate(audio_segments)
        return waveform
    
    def get_conversation_turns(self, transcript_path: str) -> Dict:
        """
        Analyze conversation turn-taking patterns.
        
        Returns:
            Dict with turn counts, talk ratio, mean turn duration
        """
        df = self.parse_transcript(transcript_path)
        if df.empty:
            return self._default_turn_metrics()
        
        df['is_participant'] = df['speaker'].apply(
            lambda s: any(label.lower() in s.lower() for label in self.participant_labels)
        )
        
        participant_df = df[df['is_participant']]
        interviewer_df = df[~df['is_participant']]
        
        participant_duration = (participant_df['end'] - participant_df['start']).sum()
        interviewer_duration = (interviewer_df['end'] - interviewer_df['start']).sum()
        total_duration = df['end'].max() - df['start'].min() if not df.empty else 1.0
        
        return {
            'participant_turn_count': len(participant_df),
            'interviewer_turn_count': len(interviewer_df),
            'participant_talk_ratio': participant_duration / total_duration if total_duration > 0 else 0.5,
            'participant_mean_turn_duration': participant_duration / len(participant_df) if len(participant_df) > 0 else 0,
            'total_duration': total_duration
        }
    
    def _default_turn_metrics(self) -> Dict:
        return {
            'participant_turn_count': 0, 'interviewer_turn_count': 0,
            'participant_talk_ratio': 0.5, 'participant_mean_turn_duration': 0,
            'total_duration': 0
        }


class VADDiarizer:
    """
    Fallback diarization using Voice Activity Detection.
    Used when transcripts are not available (e.g., EATD-Corpus).
    """
    
    def __init__(self, silero_available: bool = False):
        self.silero_available = silero_available
        self.silero_model = None
    
    def load_silero(self) -> bool:
        """Attempt to load Silero VAD model."""
        if self.silero_model is not None:
            return True
        
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            self.silero_model = model
            self.get_speech_timestamps = utils[0]
            self.silero_available = True
            return True
        except Exception as e:
            print(f"Silero VAD not available: {e}")
            return False
    
    def detect_speech_segments(self, waveform: np.ndarray, sr: int = 16000) -> List[Tuple[float, float]]:
        """
        Detect speech segments using VAD.
        
        Returns:
            List of (start_sec, end_sec) tuples
        """
        if self.silero_available and self.load_silero():
            return self._silero_detect(waveform, sr)
        else:
            return self._librosa_detect(waveform, sr)
    
    def _silero_detect(self, waveform: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Use Silero VAD for detection."""
        try:
            import torch
            audio_tensor = torch.tensor(waveform).float()
            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, self.silero_model, sampling_rate=sr
            )
            
            return [(ts['start'] / sr, ts['end'] / sr) for ts in speech_timestamps]
        except Exception:
            return self._librosa_detect(waveform, sr)
    
    def _librosa_detect(self, waveform: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Fallback using librosa."""
        try:
            import librosa
            intervals = librosa.effects.split(waveform, top_db=30)
            return [(s / sr, e / sr) for s, e in intervals]
        except:
            return [(0, len(waveform) / sr)]
    
    def extract_speech(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract speech portions of audio."""
        segments = self.detect_speech_segments(waveform, sr)
        
        audio_parts = []
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            if end_sample <= len(waveform):
                audio_parts.append(waveform[start_sample:end_sample])
        
        return np.concatenate(audio_parts) if audio_parts else waveform


class UnifiedDiarizer:
    """
    Combined diarizer using transcript when available, VAD as fallback.
    """
    
    def __init__(self):
        self.transcript_diarizer = TranscriptDiarizer()
        self.vad_diarizer = VADDiarizer()
    
    def diarize(self, waveform: np.ndarray, sr: int = 16000,
                transcript_path: str = None) -> Dict:
        """
        Extract participant audio with best available method.
        
        Returns:
            Dict with participant_audio, method_used, and segment_info
        """
        if transcript_path:
            segments = self.transcript_diarizer.get_participant_segments(transcript_path)
            if segments:
                participant_audio = self.transcript_diarizer.extract_participant_audio(
                    waveform, sr, transcript_path
                )
                turn_info = self.transcript_diarizer.get_conversation_turns(transcript_path)
                
                return {
                    'participant_audio': participant_audio,
                    'method': 'transcript',
                    'segments': segments,
                    'turn_info': turn_info
                }
        
        segments = self.vad_diarizer.detect_speech_segments(waveform, sr)
        participant_audio = self.vad_diarizer.extract_speech(waveform, sr)
        
        return {
            'participant_audio': participant_audio,
            'method': 'vad',
            'segments': segments,
            'turn_info': self.transcript_diarizer._default_turn_metrics()
        }
