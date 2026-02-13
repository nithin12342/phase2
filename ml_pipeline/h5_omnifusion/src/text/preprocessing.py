"""
Text Preprocessing Module
Implements Steps 12-15 and R18-R24 from H5-OmniFusion specification.
"""
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import robust_transcript_load


class TranscriptCleaner:
    """
    Clean raw transcripts by removing timestamps and speaker tags.
    Steps 12, R19.
    """
    
    def __init__(self):
        self.timestamp_patterns = [
            r'^\d+\.?\d*\s+\d+\.?\d*\s+',  # "0.0 1.5 "
            r'\[\d+:\d+:\d+\.\d+\]',  # [00:01:23.456]
            r'\(\d+:\d+\)',  # (01:23)
        ]
        
        self.speaker_patterns = [
            r'^(ELLIE|Ellie|ellie)[:\s]+',
            r'^(Participant|PARTICIPANT|participant)[:\s]+',
            r'^(Speaker\s*\d*)[:\s]+',
            r'^(Interviewer|INTERVIEWER)[:\s]+',
        ]
    
    def clean(self, text: str) -> str:
        """
        Remove timestamps and speaker tags from transcript.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        lines = text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in self.timestamp_patterns:
                line = re.sub(pattern, '', line)
            
            for pattern in self.speaker_patterns:
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
            
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def load_and_clean(self, transcript_path: str) -> Tuple[str, bool]:
        """Load and clean transcript from file."""
        content, success = robust_transcript_load(transcript_path)
        if not success:
            return "", False
        
        cleaned = self.clean(content)
        return cleaned, True


class AnnotationRemover:
    """
    Remove non-verbal annotations like [laughter], [sigh], [pause].
    Steps 13, R21.
    
    Tracks counts before removal for diagnostic purposes.
    """
    
    def __init__(self):
        self.annotation_pattern = r'\[([^\]]*)\]'
        
        self.tracked_annotations = [
            'laughter', 'laugh', 'laughing',
            'sigh', 'sighing',
            'pause', 'long pause',
            'silence',
            'crying', 'cry',
            'cough', 'coughing',
            'breath', 'breathing'
        ]
    
    def remove(self, text: str) -> Tuple[str, Dict]:
        """
        Remove annotations and return counts.
        
        Returns:
            (cleaned_text, annotation_counts)
        """
        annotations = re.findall(self.annotation_pattern, text)
        
        counts = {ann: 0 for ann in self.tracked_annotations}
        for ann in annotations:
            ann_lower = ann.lower()
            for tracked in self.tracked_annotations:
                if tracked in ann_lower:
                    counts[tracked] += 1
                    break
        
        cleaned = re.sub(self.annotation_pattern, '', text)
        
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned, {
            'annotation_counts': counts,
            'total_annotations': len(annotations)
        }


class ContractionExpander:
    """
    Expand English contractions for consistent text processing.
    Step R20.
    """
    
    def __init__(self):
        self.contractions = {
            "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
            "you're": "you are", "you've": "you have", "you'll": "you will",
            "he's": "he is", "she's": "she is", "it's": "it is",
            "we're": "we are", "we've": "we have", "we'll": "we will",
            "they're": "they are", "they've": "they have", "they'll": "they will",
            "that's": "that is", "there's": "there is", "here's": "here is",
            "what's": "what is", "who's": "who is", "how's": "how is",
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "wasn't": "was not",
            "weren't": "were not", "isn't": "is not", "aren't": "are not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
            "let's": "let us", "that'll": "that will", "who'll": "who will",
            "'cause": "because", "gonna": "going to", "wanna": "want to",
            "gotta": "got to", "kinda": "kind of", "sorta": "sort of"
        }
    
    def expand(self, text: str) -> str:
        """
        Expand contractions in text.
        
        Args:
            text: Input text with contractions
            
        Returns:
            Text with expanded contractions
        """
        words = text.split()
        expanded = []
        
        for word in words:
            word_lower = word.lower()
            punct = ''
            if word_lower and word_lower[-1] in '.,!?;:':
                punct = word_lower[-1]
                word_lower = word_lower[:-1]
            
            if word_lower in self.contractions:
                expanded.append(self.contractions[word_lower] + punct)
            else:
                expanded.append(word)
        
        return ' '.join(expanded)


class DisfluencyAnalyzer:
    """
    Analyze speech disfluencies (fillers like "um", "uh").
    Steps 14, R23.
    
    Preserves filler counts for diagnostic features but removes for embeddings.
    """
    
    def __init__(self, fillers: List[str] = None):
        self.fillers = fillers or CFG.FILLERS
    
    def analyze(self, text: str) -> Dict:
        """
        Count disfluencies in text.
        
        Returns:
            Dict with filler counts and rates
        """
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return self._default_result()
        
        filler_counts = {}
        total_fillers = 0
        
        for filler in self.fillers:
            if ' ' in filler:
                count = text.lower().count(filler)
            else:
                count = words.count(filler)
            
            filler_counts[filler] = count
            total_fillers += count
        
        return {
            'filler_counts': filler_counts,
            'total_fillers': total_fillers,
            'filler_rate': total_fillers / total_words,
            'word_count': total_words
        }
    
    def remove_fillers(self, text: str) -> str:
        """Remove filler words for embedding extraction."""
        words = text.split()
        filtered = []
        
        for word in words:
            if word.lower().strip('.,!?') not in self.fillers:
                filtered.append(word)
        
        return ' '.join(filtered)
    
    def _default_result(self) -> Dict:
        return {
            'filler_counts': {}, 'total_fillers': 0,
            'filler_rate': 0.0, 'word_count': 0
        }


class WhitespaceNormalizer:
    """Normalize whitespace in text. Step R22."""
    
    def normalize(self, text: str) -> str:
        """Normalize all whitespace to single spaces."""
        return re.sub(r'\s+', ' ', text).strip()


class TextTokenizer:
    """
    Tokenize text for transformer models.
    Steps 15, R24.
    
    Uses RoBERTa BPE for English, BertTokenizer for Chinese.
    """
    
    def __init__(self, max_length: int = 512, language: str = 'english'):
        self.max_length = max_length
        self.language = language
        self.tokenizer = None
    
    def _ensure_loaded(self, language: str = None):
        """Lazy load tokenizer."""
        lang = language or self.language
        
        if self.tokenizer is not None:
            return
        
        try:
            from transformers import AutoTokenizer
            
            if lang == 'chinese':
                model_name = 'hfl/chinese-roberta-wwm-ext'
            else:
                model_name = 'roberta-base'
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Tokenizer load error: {e}")
    
    def tokenize(self, text: str, language: str = None) -> Dict:
        """
        Tokenize text for transformer input.
        
        Returns:
            Dict with input_ids, attention_mask, token_count
        """
        self._ensure_loaded(language)
        
        if self.tokenizer is None:
            return self._fallback_tokenize(text)
        
        try:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_count': encoding['attention_mask'].sum().item()
            }
        except Exception as e:
            return self._fallback_tokenize(text)
    
    def _fallback_tokenize(self, text: str) -> Dict:
        """Simple word-based fallback tokenization."""
        words = text.split()[:self.max_length]
        return {
            'input_ids': None,
            'attention_mask': None,
            'token_count': len(words)
        }


class TextPreprocessor:
    """
    Unified text preprocessing pipeline (Steps 12-15, R18-R24).
    """
    
    def __init__(self, language: str = 'english'):
        self.cleaner = TranscriptCleaner()
        self.annotation_remover = AnnotationRemover()
        self.contraction_expander = ContractionExpander()
        self.disfluency_analyzer = DisfluencyAnalyzer()
        self.whitespace_normalizer = WhitespaceNormalizer()
        self.tokenizer = TextTokenizer(language=language)
        self.language = language
    
    def process(self, text: str = None, transcript_path: str = None) -> Dict:
        """
        Run complete text preprocessing pipeline.
        
        Returns:
            Dict with cleaned_text, for_embedding, tokens, and diagnostic features
        """
        if transcript_path and text is None:
            text, success = self.cleaner.load_and_clean(transcript_path)
            if not success:
                return self._failure_result()
        elif text is None:
            return self._failure_result()
        
        cleaned = self.cleaner.clean(text)
        
        cleaned, annotation_info = self.annotation_remover.remove(cleaned)
        
        if self.language == 'english':
            cleaned = self.contraction_expander.expand(cleaned)
        
        cleaned = self.whitespace_normalizer.normalize(cleaned)
        
        disfluency_info = self.disfluency_analyzer.analyze(cleaned)
        
        text_for_embedding = self.disfluency_analyzer.remove_fillers(cleaned)
        
        tokens = self.tokenizer.tokenize(text_for_embedding, self.language)
        
        return {
            'success': True,
            'cleaned_text': cleaned,
            'text_for_embedding': text_for_embedding,
            'tokens': tokens,
            'annotation_info': annotation_info,
            'disfluency_info': disfluency_info,
            'language': self.language
        }
    
    def _failure_result(self) -> Dict:
        return {
            'success': False,
            'cleaned_text': '',
            'text_for_embedding': '',
            'tokens': {'input_ids': None, 'attention_mask': None, 'token_count': 0},
            'annotation_info': {'annotation_counts': {}, 'total_annotations': 0},
            'disfluency_info': {'filler_counts': {}, 'total_fillers': 0, 'filler_rate': 0, 'word_count': 0},
            'language': self.language
        }
