"""
Chinese Text Support Module
Implements Chinese language adaptations for EATD-Corpus processing.
"""
import re
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import DEVICE, SNOWNLP_AVAILABLE

if SNOWNLP_AVAILABLE:
    from snownlp import SnowNLP


class ChineseTextProcessor:
    """
    Specialized text processing for Mandarin Chinese (EATD-Corpus).
    
    Key differences from English:
    - No contractions
    - Different tokenization (character-level)
    - Different sentiment tools (SnowNLP)
    - Different linguistic markers
    """
    
    def __init__(self):
        self.first_person_chinese = {'我', '我的', '我们', '咱', '咱们'}
        self.negative_emotion_chinese = {
            '难过', '悲伤', '痛苦', '绝望', '抑郁', '沮丧',
            '焦虑', '害怕', '孤独', '无助', '疲惫'
        }
        self.absolutist_chinese = {
            '总是', '从不', '一定', '必须', '完全', '绝对',
            '永远', '任何', '所有', '全部'
        }
    
    def process(self, text: str) -> Dict:
        """
        Process Chinese text.
        
        Returns:
            Dict with cleaned text, character count, word segments
        """
        text = re.sub(r'\[.*?\]', '', text)
        
        text = re.sub(r'\s+', '', text)
        
        char_count = len(text)
        
        words = self._segment_words(text)
        
        return {
            'cleaned_text': text,
            'char_count': char_count,
            'word_count': len(words),
            'words': words
        }
    
    def _segment_words(self, text: str) -> List[str]:
        """Segment Chinese text into words."""
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            return list(text)
    
    def analyze_linguistic(self, text: str) -> Dict:
        """
        Analyze Chinese linguistic features.
        
        Returns:
            Dict with category counts and ratios
        """
        chars = list(text)
        words = self._segment_words(text)
        
        first_person_count = sum(1 for w in words if w in self.first_person_chinese)
        negative_count = sum(1 for w in words if w in self.negative_emotion_chinese)
        absolutist_count = sum(1 for w in words if w in self.absolutist_chinese)
        
        total = len(words) or 1
        
        return {
            'first_person': first_person_count,
            'negative_emotion': negative_count,
            'absolutist': absolutist_count,
            'first_person_ratio': first_person_count / total,
            'negative_emotion_ratio': negative_count / total,
            'absolutist_ratio': absolutist_count / total,
            'word_count': len(words),
            'char_count': len(chars)
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze Chinese sentiment using SnowNLP.
        
        Returns:
            Dict with sentiment score
        """
        if not SNOWNLP_AVAILABLE:
            return {'compound': 0, 'positive': 0.5, 'negative': 0.5}
        
        try:
            s = SnowNLP(text)
            sentiment = s.sentiments  # 0-1 scale
            
            return {
                'compound': (sentiment - 0.5) * 2,  # -1 to 1
                'positive': sentiment,
                'negative': 1 - sentiment,
                'raw_score': sentiment
            }
        except:
            return {'compound': 0, 'positive': 0.5, 'negative': 0.5}
    
    def get_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from Chinese text."""
        if not SNOWNLP_AVAILABLE:
            return []
        
        try:
            s = SnowNLP(text)
            return s.keywords(top_n)
        except:
            return []
    
    def summarize(self, text: str, num_sentences: int = 3) -> List[str]:
        """Extract summary sentences from Chinese text."""
        if not SNOWNLP_AVAILABLE:
            return []
        
        try:
            s = SnowNLP(text)
            return s.summary(num_sentences)
        except:
            return []


class LanguageDetector:
    """
    Simple language detection to route to appropriate processor.
    """
    
    def __init__(self):
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    def detect(self, text: str) -> str:
        """
        Detect if text is Chinese or English.
        
        Returns:
            'chinese' or 'english'
        """
        if not text:
            return 'english'
        
        chinese_chars = len(self.chinese_pattern.findall(text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'english'
        
        chinese_ratio = chinese_chars / total_chars
        
        return 'chinese' if chinese_ratio > 0.3 else 'english'
    
    def is_chinese(self, text: str) -> bool:
        """Check if text is Chinese."""
        return self.detect(text) == 'chinese'


class BilingualTextProcessor:
    """
    Process text with automatic language detection and routing.
    """
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.chinese_processor = ChineseTextProcessor()
    
    def process(self, text: str) -> Dict:
        """
        Process text with automatic language handling.
        
        Returns:
            Dict with processed text and detected language
        """
        language = self.language_detector.detect(text)
        
        if language == 'chinese':
            result = self.chinese_processor.process(text)
            linguistic = self.chinese_processor.analyze_linguistic(text)
            sentiment = self.chinese_processor.analyze_sentiment(text)
        else:
            result = {
                'cleaned_text': text,
                'char_count': len(text),
                'word_count': len(text.split())
            }
            linguistic = {}
            sentiment = {}
        
        return {
            'language': language,
            **result,
            'linguistic': linguistic,
            'sentiment': sentiment
        }
