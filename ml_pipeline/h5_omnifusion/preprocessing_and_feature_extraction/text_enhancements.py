"""
H5-OmniFusion Text Pipeline Enhancements
Steps 12-20 from 40-Step Production Pipeline + Chinese Support for EATD-Corpus
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

class TranscriptCleaner:
    """Clean transcript text per production pipeline steps 12-14."""
    
    DISFLUENCIES = ['um', 'uh', 'er', 'ah', 'hm', 'hmm', 'mm', 'mhm', 'uh-huh']
    
    def __init__(self, preserve_disfluencies: bool = False):
        self.preserve_disfluencies = preserve_disfluencies
    
    def clean(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Clean text and return disfluency counts."""
        original_words = text.lower().split()
        word_count = len(original_words) + 1e-8
        
        disfluency_count = sum(1 for w in original_words if w.strip('.,!?') in self.DISFLUENCIES)
        disfluency_rate = disfluency_count / word_count
        
        text = re.sub(r'^\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(ELLIE|Participant|Ellie|PARTICIPANT):\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        text = re.sub(r'\[.*?\]', '', text)  # [laughter], [sigh], etc.
        text = re.sub(r'\(.*?\)', '', text)  # (pause), (inaudible)
        text = re.sub(r'<.*?>', '', text)    # <breath>, <noise>
        
        if not self.preserve_disfluencies:
            pattern = r'\b(' + '|'.join(self.DISFLUENCIES) + r')\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = ContractionExpander.expand(text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        features = {
            'disfluency_count': disfluency_count,
            'disfluency_rate': float(disfluency_rate),
            'cleaned_text': text
        }
        
        return features
    
    def load_transcript(self, path: str) -> str:
        """Load transcript from file (DAIC-WOZ CSV or plain text)."""
        import os
        import pandas as pd
        
        if not path or not os.path.exists(path):
            return ""
        
        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path, sep='\t', header=None, 
                                 names=['start', 'end', 'speaker', 'text'],
                                 on_bad_lines='skip')
                df_p = df[~df['speaker'].str.lower().str.contains('ellie', na=False)]
                text = ' '.join(df_p['text'].dropna().astype(str).tolist())
            else:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            return text.strip()
        except Exception as e:
            print(f"Error loading transcript {path}: {e}")
            return ""


class ContractionExpander:
    """Expand English contractions (Step R20)."""
    
    CONTRACTIONS = {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
        "he's": "he is", "he'll": "he will", "he'd": "he would",
        "she's": "she is", "she'll": "she will", "she'd": "she would",
        "it's": "it is", "it'll": "it will",
        "we're": "we are", "we've": "we have", "we'll": "we will", "we'd": "we would",
        "they're": "they are", "they've": "they have", "they'll": "they will", "they'd": "they would",
        "that's": "that is", "what's": "what is", "where's": "where is", "who's": "who is",
        "can't": "cannot", "won't": "will not", "shan't": "shall not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
        "mustn't": "must not", "mightn't": "might not", "needn't": "need not"
    }
    
    @staticmethod
    def expand(text: str) -> str:
        """Expand contractions using library or fallback map."""
        if not text: return ""
        
        try:
            import contractions
            return contractions.fix(text)
        except ImportError:
            text_lower = text.lower()
            
            def replace(match):
                word = match.group(0)
                return ContractionExpander.CONTRACTIONS.get(word.lower(), word)
            
            pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in ContractionExpander.CONTRACTIONS.keys()) + r')\b', re.IGNORECASE)
            return pattern.sub(replace, text)


class PsycholinguisticExtractor:
    """Extract evidence-based psycholinguistic features for depression."""
    
    FIRST_PERSON_SING = ['i', 'me', 'my', 'myself', 'mine']
    FIRST_PERSON_PLUR = ['we', 'us', 'our', 'ourselves', 'ours']
    ABSOLUTIST = ['always', 'never', 'nothing', 'everything', 'completely', 
                  'totally', 'absolutely', 'constantly', 'entire', 'whole']
    NEGATIVE_EMOTION = ['sad', 'depressed', 'hopeless', 'worthless', 'tired',
                        'empty', 'lonely', 'anxious', 'afraid', 'angry',
                        'frustrated', 'guilty', 'ashamed', 'hurt', 'miserable',
                        'terrible', 'awful', 'horrible', 'hate', 'pain']
    COGNITIVE = ['think', 'know', 'believe', 'feel', 'understand',
                 'realize', 'remember', 'wonder', 'guess', 'suppose']
    PAST_MARKERS = ['was', 'were', 'had', 'did', 'used', 'ago', 'before', 'yesterday']
    FUTURE_MARKERS = ['will', 'going', 'gonna', 'tomorrow', 'soon', 'later', 'plan']
    SOCIAL = ['friend', 'family', 'mother', 'father', 'people', 'talk',
              'share', 'together', 'relationship', 'love']
    SLEEP = ['sleep', 'insomnia', 'tired', 'exhausted', 'rest', 'awake', 'bed']
    ANHEDONIA = ['boring', 'bored', 'pointless', 'meaningless', 'interest',
                 'enjoy', 'pleasure', 'fun', 'happy', 'excited']
    
    def extract(self, text: str) -> Dict[str, float]:
        """Extract all psycholinguistic features."""
        words = text.lower().split()
        wc = len(words) + 1e-8
        
        return {
            'first_person_singular': sum(1 for w in words if w in self.FIRST_PERSON_SING) / wc,
            'first_person_plural': sum(1 for w in words if w in self.FIRST_PERSON_PLUR) / wc,
            'absolutist': sum(1 for w in words if w in self.ABSOLUTIST) / wc,
            'negative_emotion': sum(1 for w in words if w in self.NEGATIVE_EMOTION) / wc,
            'cognitive': sum(1 for w in words if w in self.COGNITIVE) / wc,
            'past_focus': sum(1 for w in words if w in self.PAST_MARKERS) / wc,
            'future_focus': sum(1 for w in words if w in self.FUTURE_MARKERS) / wc,
            'social': sum(1 for w in words if w in self.SOCIAL) / wc,
            'sleep_words': sum(1 for w in words if w in self.SLEEP) / wc,
            'anhedonia': sum(1 for w in words if w in self.ANHEDONIA) / wc,
            'word_count': len(words),
            'lexical_diversity': len(set(words)) / wc
        }


class ComplexityAnalyzer:
    """Text complexity and readability metrics."""
    
    def extract(self, text: str) -> Dict[str, float]:
        """Extract readability and complexity features."""
        try:
            import textstat
            return {
                'flesch_reading_ease': float(textstat.flesch_reading_ease(text)),
                'flesch_kincaid_grade': float(textstat.flesch_kincaid_grade(text)),
                'gunning_fog': float(textstat.gunning_fog(text)),
                'automated_readability': float(textstat.automated_readability_index(text)),
                'syllable_count': float(textstat.syllable_count(text)),
                'sentence_count': max(1, len(re.split(r'[.!?]+', text)))
            }
        except Exception:
            return {
                'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
                'gunning_fog': 0, 'automated_readability': 0,
                'syllable_count': 0, 'sentence_count': 1
            }


class MultilingualSentimentAnalyzer:
    """Sentiment analysis for English (VADER) and Chinese (SnowNLP)."""
    
    def __init__(self):
        self._en_analyzer = None
        self._zh_analyzer = None
    
    @property
    def en_analyzer(self):
        if self._en_analyzer is None:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self._en_analyzer = SentimentIntensityAnalyzer()
        return self._en_analyzer
    
    def analyze_english(self, text: str) -> Dict[str, float]:
        """VADER sentiment for English."""
        scores = self.en_analyzer.polarity_scores(text)
        return {
            'sentiment_neg': scores['neg'],
            'sentiment_neu': scores['neu'],
            'sentiment_pos': scores['pos'],
            'sentiment_compound': scores['compound']
        }
    
    def analyze_chinese(self, text: str) -> Dict[str, float]:
        """SnowNLP sentiment for Chinese."""
        try:
            from snownlp import SnowNLP
            s = SnowNLP(text)
            sentiment = s.sentiments  # 0-1, higher = more positive
            return {
                'sentiment_neg': 1 - sentiment,
                'sentiment_neu': 0.5,  # SnowNLP doesn't provide neutral
                'sentiment_pos': sentiment,
                'sentiment_compound': sentiment * 2 - 1  # Scale to [-1, 1]
            }
        except ImportError:
            return {'sentiment_neg': 0, 'sentiment_neu': 1, 'sentiment_pos': 0, 'sentiment_compound': 0}
    
    def analyze(self, text: str, lang: str = 'english') -> Dict[str, float]:
        """Unified analyze method - dispatches to language-specific method."""
        if lang.lower() in ['chinese', 'mandarin', 'zh']:
            return self.analyze_chinese(text)
        else:
            return self.analyze_english(text)


class ConversationDynamicsAnalyzer:
    """Analyze turn-taking and conversation patterns."""
    
    def analyze(self, turns: List[Dict]) -> Dict[str, float]:
        """Analyze conversation dynamics from turn list.
        
        Args:
            turns: List of {'speaker': str, 'text': str, 'start': float, 'end': float}
        """
        participant_turns = [t for t in turns if 'participant' in t.get('speaker', '').lower()]
        ellie_turns = [t for t in turns if 'ellie' in t.get('speaker', '').lower()]
        
        if not participant_turns:
            return {k: 0 for k in ['turn_count', 'turn_length_mean', 'turn_length_std',
                                   'turn_length_trend', 'talk_ratio', 'engagement_change']}
        
        p_lens = [len(t.get('text', '').split()) for t in participant_turns]
        e_lens = [len(t.get('text', '').split()) for t in ellie_turns]
        
        p_words = sum(p_lens)
        e_words = sum(e_lens)
        talk_ratio = p_words / (p_words + e_words + 1e-8)
        
        n = len(p_lens)
        if n >= 6:
            early = np.mean(p_lens[:n//3])
            late = np.mean(p_lens[2*n//3:])
            trend = late - early
        else:
            trend = 0
        
        latencies = []
        for i, e_turn in enumerate(ellie_turns):
            for p_turn in participant_turns:
                if p_turn.get('start', 0) > e_turn.get('end', 0):
                    lat = p_turn['start'] - e_turn['end']
                    if 0 < lat < 15:
                        latencies.append(lat)
                    break
        
        return {
            'turn_count': len(participant_turns),
            'turn_length_mean': float(np.mean(p_lens)),
            'turn_length_std': float(np.std(p_lens)),
            'turn_length_min': float(np.min(p_lens)),
            'turn_length_max': float(np.max(p_lens)),
            'turn_length_trend': float(trend),
            'talk_ratio': float(talk_ratio),
            'engagement_change': float(trend),
            'response_latency_mean': float(np.mean(latencies)) if latencies else 0,
            'response_latency_std': float(np.std(latencies)) if latencies else 0,
            'response_latency_max': float(np.max(latencies)) if latencies else 0,
            'slow_response_ratio': float(sum(1 for l in latencies if l > 2) / (len(latencies) + 1e-8))
        }


class LanguageDetector:
    """Detect language for model switching."""
    
    @staticmethod
    def detect(text: str) -> str:
        """Return 'zh' for Chinese, 'en' for English."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ratio = chinese_chars / (len(text) + 1e-8)
        return 'zh' if ratio > 0.3 else 'en'


class ChinesePsycholinguisticExtractor:
    """Chinese-specific psycholinguistic features for EATD-Corpus."""
    
    FIRST_PERSON = ['我', '我的', '我自己']
    NEGATIVE = ['难过', '难受', '悲伤', '绝望', '焦虑', '害怕', '生气', '孤独', '空虚']
    ABSOLUTIST = ['总是', '永远', '从不', '一直', '完全', '绝对']
    
    def extract(self, text: str) -> Dict[str, float]:
        """Extract Chinese psycholinguistic features."""
        char_count = len(text) + 1e-8
        
        return {
            'cn_first_person': sum(text.count(w) for w in self.FIRST_PERSON) / char_count,
            'cn_negative': sum(text.count(w) for w in self.NEGATIVE) / char_count,
            'cn_absolutist': sum(text.count(w) for w in self.ABSOLUTIST) / char_count
        }
