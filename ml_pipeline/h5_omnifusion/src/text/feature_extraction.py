"""
Text Feature Extraction Module
Implements Steps 16-20 and R25-R31 from H5-OmniFusion specification.
"""
import re
import numpy as np
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from ..config import CFG
from ..utils import (
    DEVICE, VADER_AVAILABLE, TRANSFORMERS_AVAILABLE,
    ensure_768_dim, safe_embedding
)
from ..model_loader import MODEL_LOADER

if VADER_AVAILABLE:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TextEmbeddingExtractor:
    """
    Extract 768-dim text embeddings using transformer models.
    Steps 16, R25.
    
    Uses mental-roberta-base for English, chinese-roberta-wwm-ext for Chinese.
    """
    
    def __init__(self, language: str = 'english', device=DEVICE):
        self.language = language
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def _ensure_loaded(self, language: str = None):
        """Lazy load model."""
        lang = language or self.language
        if self.model is None:
            self.model, self.tokenizer = MODEL_LOADER.get_text_encoder(lang)
    
    def extract(self, text: str, language: str = None) -> np.ndarray:
        """
        Extract [CLS] token embedding.
        
        Args:
            text: Input text
            language: 'english' or 'chinese'
            
        Returns:
            768-dim embedding
        """
        lang = language or self.language
        self._ensure_loaded(lang)
        
        if self.model is None or self.tokenizer is None:
            return np.zeros(768, dtype=np.float32)
        
        try:
            inputs = self.tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)
            
            if hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                pass  # inputs are handled by model
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            return safe_embedding(cls_embedding.cpu().float().numpy().squeeze())
            
        except Exception as e:
            print(f"Text embedding error: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def extract_mean_pooled(self, text: str, language: str = None) -> np.ndarray:
        """Extract mean-pooled embedding (alternative to [CLS])."""
        lang = language or self.language
        self._ensure_loaded(lang)
        
        if self.model is None or self.tokenizer is None:
            return np.zeros(768, dtype=np.float32)
        
        try:
            inputs = self.tokenizer(
                text, max_length=512, truncation=True,
                padding='max_length', return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention = inputs['attention_mask'].unsqueeze(-1)
                embeddings = outputs.last_hidden_state * attention
                mean_emb = embeddings.sum(dim=1) / attention.sum(dim=1)
            
            return safe_embedding(mean_emb.cpu().float().numpy().squeeze())
            
        except Exception as e:
            return np.zeros(768, dtype=np.float32)


class LinguisticAnalyzer:
    """
    Extract LIWC-style linguistic features.
    Steps 17, R26.
    
    Features: first-person pronouns, absolutist words, negative emotion terms.
    """
    
    def __init__(self, config=None):
        cfg = config or CFG
        self.first_person = set(cfg.FIRST_PERSON)
        self.absolutist = set(cfg.ABSOLUTIST)
        self.negative_emotion = set(cfg.NEGATIVE_EMOTION)
        
        self.positive_emotion = {
            'happy', 'joy', 'love', 'wonderful', 'great', 'good',
            'excellent', 'amazing', 'beautiful', 'glad'
        }
        self.cognitive = {
            'think', 'know', 'believe', 'feel', 'understand',
            'realize', 'consider', 'remember', 'guess'
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text for linguistic marker categories.
        
        Returns:
            Dict with category counts and ratios
        """
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return self._default_result()
        
        counts = {
            'first_person': sum(1 for w in words if w in self.first_person),
            'absolutist': sum(1 for w in words if w in self.absolutist),
            'negative_emotion': sum(1 for w in words if w in self.negative_emotion),
            'positive_emotion': sum(1 for w in words if w in self.positive_emotion),
            'cognitive': sum(1 for w in words if w in self.cognitive)
        }
        
        ratios = {f'{k}_ratio': v / total_words for k, v in counts.items()}
        
        return {
            **counts,
            **ratios,
            'total_words': total_words
        }
    
    def _default_result(self) -> Dict:
        return {
            'first_person': 0, 'absolutist': 0, 'negative_emotion': 0,
            'positive_emotion': 0, 'cognitive': 0,
            'first_person_ratio': 0, 'absolutist_ratio': 0,
            'negative_emotion_ratio': 0, 'positive_emotion_ratio': 0,
            'cognitive_ratio': 0, 'total_words': 0
        }


class LexicalDiversity:
    """
    Calculate lexical diversity metrics.
    Steps 18, R27.
    """
    
    def __init__(self, mattr_window: int = 50):
        self.mattr_window = mattr_window
    
    def analyze(self, text: str) -> Dict:
        """
        Calculate lexical diversity metrics.
        
        Returns:
            Dict with TTR, MATTR, hapax ratio
        """
        words = text.lower().split()
        total_words = len(words)
        
        if total_words < 2:
            return self._default_result()
        
        unique_words = set(words)
        
        ttr = len(unique_words) / total_words
        
        mattr = self._calculate_mattr(words)
        
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        hapax = sum(1 for c in word_counts.values() if c == 1)
        hapax_ratio = hapax / total_words
        
        return {
            'ttr': ttr,
            'mattr': mattr,
            'hapax_count': hapax,
            'hapax_ratio': hapax_ratio,
            'unique_words': len(unique_words),
            'total_words': total_words
        }
    
    def _calculate_mattr(self, words: List[str]) -> float:
        """Calculate Moving Average Type-Token Ratio."""
        if len(words) < self.mattr_window:
            return len(set(words)) / len(words)
        
        ttr_values = []
        for i in range(len(words) - self.mattr_window + 1):
            window = words[i:i + self.mattr_window]
            ttr_values.append(len(set(window)) / self.mattr_window)
        
        return np.mean(ttr_values)
    
    def _default_result(self) -> Dict:
        return {
            'ttr': 0, 'mattr': 0, 'hapax_count': 0,
            'hapax_ratio': 0, 'unique_words': 0, 'total_words': 0
        }


class ReadabilityAnalyzer:
    """
    Calculate readability scores.
    Steps 18, R28.
    """
    
    def analyze(self, text: str) -> Dict:
        """
        Calculate readability metrics.
        
        Returns:
            Dict with Flesch-Kincaid, Gunning Fog, avg sentence/word length
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        
        if not sentences or not words:
            return self._default_result()
        
        syllables = sum(self._count_syllables(w) for w in words)
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        avg_word_length = np.mean([len(w) for w in words])
        
        fk_grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        gunning_fog = 0.4 * (avg_words_per_sentence + 100 * complex_words / len(words))
        
        return {
            'flesch_kincaid_grade': max(0, fk_grade),
            'gunning_fog': max(0, gunning_fog),
            'avg_sentence_length': avg_words_per_sentence,
            'avg_word_length': avg_word_length,
            'sentence_count': len(sentences)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count."""
        word = word.lower().strip('.,!?;:')
        if len(word) <= 3:
            return 1
        
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        if word.endswith('e'):
            count = max(1, count - 1)
        
        return max(1, count)
    
    def _default_result(self) -> Dict:
        return {
            'flesch_kincaid_grade': 0, 'gunning_fog': 0,
            'avg_sentence_length': 0, 'avg_word_length': 0, 'sentence_count': 0
        }


class SentimentAnalyzer:
    """
    Extract sentiment valence and polarity.
    Steps 19, R29.
    
    Uses VADER for English, SnowNLP for Chinese.
    """
    
    def __init__(self):
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str, language: str = 'english') -> Dict:
        """
        Analyze sentiment.
        
        Returns:
            Dict with compound, positive, negative, neutral scores
        """
        if language == 'chinese':
            return self._analyze_chinese(text)
        else:
            return self._analyze_english(text)
    
    def _analyze_english(self, text: str) -> Dict:
        """VADER sentiment analysis."""
        if self.vader is None:
            return self._default_result()
        
        try:
            scores = self.vader.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except:
            return self._default_result()
    
    def _analyze_chinese(self, text: str) -> Dict:
        """SnowNLP sentiment analysis."""
        try:
            from snownlp import SnowNLP
            s = SnowNLP(text)
            sentiment = s.sentiments  # 0-1 scale
            
            return {
                'compound': (sentiment - 0.5) * 2,  # Convert to -1 to 1
                'positive': sentiment,
                'negative': 1 - sentiment,
                'neutral': 0.0
            }
        except:
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}


class EmotionLabeler:
    """
    Classify text into discrete emotion categories.
    Step R30.
    """
    
    def __init__(self):
        self.emotion_keywords = {
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'hopeless', 'lonely'],
            'fear': ['afraid', 'scared', 'anxious', 'worried', 'nervous', 'terrified'],
            'joy': ['happy', 'joyful', 'excited', 'pleased', 'delighted', 'glad'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled'],
            'disgust': ['disgusted', 'repulsed', 'revolted', 'sick']
        }
    
    def label(self, text: str) -> Dict:
        """
        Label text with emotion category.
        
        Returns:
            Dict with dominant_emotion and emotion_scores
        """
        words = set(text.lower().split())
        
        scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in words)
            scores[emotion] = score
        
        total = sum(scores.values()) or 1
        probabilities = {e: s / total for e, s in scores.items()}
        
        dominant = max(scores, key=scores.get) if max(scores.values()) > 0 else 'neutral'
        
        return {
            'dominant_emotion': dominant,
            'emotion_scores': scores,
            'emotion_probabilities': probabilities
        }


class ConversationDynamics:
    """
    Analyze conversation dynamics from transcript.
    Steps 20, R31.
    """
    
    def analyze(self, turn_info: Dict = None, transcript_df=None) -> Dict:
        """
        Calculate conversation dynamics metrics.
        
        Returns:
            Dict with talk_ratio, engagement_trajectory, etc.
        """
        if turn_info is None and transcript_df is None:
            return self._default_result()
        
        if turn_info:
            return self._from_turn_info(turn_info)
        else:
            return self._from_transcript(transcript_df)
    
    def _from_turn_info(self, turn_info: Dict) -> Dict:
        """Extract dynamics from pre-computed turn info."""
        return {
            'talk_ratio': turn_info.get('participant_talk_ratio', 0.5),
            'turn_count': turn_info.get('participant_turn_count', 0),
            'mean_turn_duration': turn_info.get('participant_mean_turn_duration', 0),
            'total_duration': turn_info.get('total_duration', 0),
            'engagement_slope': 0  # Would need temporal data
        }
    
    def _from_transcript(self, df) -> Dict:
        """Calculate dynamics from transcript DataFrame."""
        if df is None or df.empty:
            return self._default_result()
        
        try:
            participant_mask = df['speaker'].str.lower().str.contains('participant|man|woman|user')
            participant_df = df[participant_mask]
            
            if participant_df.empty:
                return self._default_result()
            
            total_duration = df['end'].max() - df['start'].min()
            participant_duration = (participant_df['end'] - participant_df['start']).sum()
            
            thirds = np.array_split(participant_df, 3)
            words_per_third = []
            for third in thirds:
                if len(third) > 0:
                    words = sum(len(str(t).split()) for t in third['text'])
                    words_per_third.append(words / len(third))
                else:
                    words_per_third.append(0)
            
            if len(words_per_third) >= 2:
                slope = np.polyfit(range(len(words_per_third)), words_per_third, 1)[0]
            else:
                slope = 0
            
            return {
                'talk_ratio': participant_duration / total_duration if total_duration > 0 else 0.5,
                'turn_count': len(participant_df),
                'mean_turn_duration': participant_duration / len(participant_df),
                'total_duration': total_duration,
                'engagement_slope': slope,
                'words_per_third': words_per_third
            }
        except:
            return self._default_result()
    
    def _default_result(self) -> Dict:
        return {
            'talk_ratio': 0.5, 'turn_count': 0, 'mean_turn_duration': 0,
            'total_duration': 0, 'engagement_slope': 0
        }


class TextFeatureExtractor:
    """
    Unified text feature extraction (Steps 16-20, R25-R31).
    """
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.embedding_extractor = TextEmbeddingExtractor(language)
        self.linguistic = LinguisticAnalyzer()
        self.lexical = LexicalDiversity()
        self.readability = ReadabilityAnalyzer()
        self.sentiment = SentimentAnalyzer()
        self.emotion = EmotionLabeler()
        self.dynamics = ConversationDynamics()
    
    def extract_all(self, text: str, turn_info: Dict = None,
                    language: str = None) -> Dict:
        """
        Extract all text features.
        
        Returns:
            Dict with embedding and all linguistic/semantic features
        """
        lang = language or self.language
        
        embedding = self.embedding_extractor.extract(text, lang)
        
        linguistic = self.linguistic.analyze(text)
        
        lexical = self.lexical.analyze(text)
        
        readability = self.readability.analyze(text)
        
        sentiment = self.sentiment.analyze(text, lang)
        
        emotion = self.emotion.label(text)
        
        dynamics = self.dynamics.analyze(turn_info)
        
        return {
            'text_embedding': embedding,  # 768-dim
            'linguistic': linguistic,
            'lexical': lexical,
            'readability': readability,
            'sentiment': sentiment,
            'emotion': emotion,
            'dynamics': dynamics
        }
