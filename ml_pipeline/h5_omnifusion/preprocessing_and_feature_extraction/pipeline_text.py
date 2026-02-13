"""
H5-OmniFusion Text Pipeline
Steps 12-20, R18-R31
"""
import os, re
import numpy as np
from typing import Dict, List, Optional
import torch
import torch.nn as nn

try:
    from text_enhancements import (
        TranscriptCleaner as EnhancedTranscriptCleaner,
        PsycholinguisticExtractor as EnhancedPsycholinguistic,
        ComplexityAnalyzer,
        MultilingualSentimentAnalyzer as EnhancedSentiment,
        ConversationDynamicsAnalyzer as EnhancedDynamics,
        LanguageDetector as EnhancedLanguageDetector
    )
    TEXT_ENHANCEMENTS_OK = True
except ImportError:
    TEXT_ENHANCEMENTS_OK = False
    print("Warning: text_enhancements module not found, using built-in classes")

try:
    from research_layer_extensions import CategoricalEmotionLabeler, TextAugmenter
    R30_R58_OK = True
except ImportError:
    R30_R58_OK = False
    print("Warning: CategoricalEmotionLabeler (R30) / TextAugmenter (R58) not found")


class TranscriptCleaner:
    """Steps 12-14, R18-R23: Transcript cleaning."""
    FILLERS = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean']
    
    def clean(self, text: str) -> Dict:
        if not text: return {'cleaned_text': '', 'filler_count': 0, 'annotation_count': 0}
        
        annotations = len(re.findall(r'\[.*?\]', text))
        
        text = re.sub(r'^\d+\.?\d*\s+\d+\.?\d*\s+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'^(ELLIE|Participant|Speaker\s*\d*)[:\s]+', '', text, flags=re.IGNORECASE|re.MULTILINE)
        
        text = re.sub(r'\[.*?\]', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        text_lower = text.lower()
        filler_count = sum(text_lower.count(f) for f in self.FILLERS)
        
        return {'cleaned_text': text, 'filler_count': filler_count, 'annotation_count': annotations}
    
    def load_transcript(self, path: str) -> str:
        if not path or not os.path.exists(path): return ''
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except: continue
        return ''

class LanguageDetector:
    """Detect English vs Chinese text."""
    @staticmethod
    def detect(text: str) -> str:
        if not text: return 'english'
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return 'chinese' if chinese_chars > len(text) * 0.1 else 'english'

class PsycholinguisticExtractor:
    """Step 17, R26: LIWC-style features."""
    FIRST_PERSON = ['i', 'me', 'my', 'myself', 'mine']
    NEGATIVE = ['sad', 'angry', 'afraid', 'anxious', 'depressed', 'hopeless', 'worthless', 'terrible', 'awful']
    POSITIVE = ['happy', 'joy', 'love', 'excited', 'good', 'great', 'wonderful', 'amazing']
    ABSOLUTIST = ['always', 'never', 'nothing', 'everything', 'completely', 'totally', 'absolutely']
    COGNITIVE = ['think', 'know', 'consider', 'because', 'reason', 'understand']
    
    def extract(self, text: str) -> Dict:
        if not text: return {'first_person_ratio':0, 'negative_ratio':0, 'positive_ratio':0, 'absolutist_ratio':0}
        words = text.lower().split()
        n = len(words) or 1
        return {
            'first_person_ratio': sum(1 for w in words if w in self.FIRST_PERSON) / n,
            'negative_ratio': sum(1 for w in words if w in self.NEGATIVE) / n,
            'positive_ratio': sum(1 for w in words if w in self.POSITIVE) / n,
            'absolutist_ratio': sum(1 for w in words if w in self.ABSOLUTIST) / n,
            'cognitive_ratio': sum(1 for w in words if w in self.COGNITIVE) / n,
        }

class LexicalDiversityAnalyzer:
    """R27: Vocabulary richness."""
    def analyze(self, text: str) -> Dict:
        if not text: return {'ttr':0, 'word_count':0, 'unique_words':0}
        words = text.lower().split()
        unique = set(words)
        n = len(words) or 1
        return {'ttr': len(unique)/n, 'word_count': len(words), 'unique_words': len(unique)}

class ReadabilityExtractor:
    """Step 18, R28: Readability scores."""
    def extract(self, text: str) -> Dict:
        if not text: return {'avg_word_length':0, 'avg_sentence_length':0}
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        n_words = len(words) or 1
        n_sents = len(sentences) or 1
        return {
            'avg_word_length': sum(len(w) for w in words) / n_words,
            'avg_sentence_length': n_words / n_sents,
        }

class MultilingualSentimentAnalyzer:
    """Step 19, R29: Sentiment analysis (Non-Proximal Transformer)."""
    def __init__(self):
        self.vader = None
        self.transformer_pipeline = None
        
        try:
            from transformers import pipeline
            self.transformer_pipeline = pipeline("sentiment-analysis", 
                                               model="distilbert-base-uncased-finetuned-sst-2-english",
                                               device=0 if torch.cuda.is_available() else -1)
            print("Loaded Transformer Analysis (Non-Proximal)")
        except Exception as e:
            print(f"Transformer Sentiment failed: {e}")
        
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except: pass
    
    def analyze(self, text: str, lang: str = 'english') -> Dict:
        if not text: return {'sentiment_compound':0, 'sentiment_pos':0, 'sentiment_neg':0, 'sentiment_neu':0}
        
        if self.transformer_pipeline and lang == 'english':
            try:
                trunc_text = text[:512] 
                result = self.transformer_pipeline(trunc_text)[0]
                score = result['score']
                label = result['label']
                compound = score if label == 'POSITIVE' else -score
                
                return {
                    'sentiment_compound': compound, 
                    'sentiment_pos': score if label == 'POSITIVE' else (1-score), 
                    'sentiment_neg': score if label == 'NEGATIVE' else (1-score), 
                    'sentiment_neu': 0.0 # Binary model lacks neutral
                }
            except Exception as e:
                print(f"Transformer inference error: {e}")
        
        if lang == 'chinese':
            try:
                from snownlp import SnowNLP
                s = SnowNLP(text)
                score = s.sentiments
                return {'sentiment_compound': score*2-1, 'sentiment_pos': score, 'sentiment_neg': 1-score, 'sentiment_neu': 0.0}
            except: pass
        
        if self.vader:
            scores = self.vader.polarity_scores(text)
            return {'sentiment_compound': scores['compound'], 'sentiment_pos': scores['pos'], 'sentiment_neg': scores['neg'], 'sentiment_neu': scores['neu']}
        
        return {'sentiment_compound':0, 'sentiment_pos':0, 'sentiment_neg':0, 'sentiment_neu':0}

class EmotionLabeler:
    """R30: Categorical emotion."""
    EMOTION_WORDS = {
        'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'grief', 'sorrow'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated'],
        'fear': ['afraid', 'scared', 'anxious', 'worried', 'nervous'],
        'joy': ['happy', 'joyful', 'excited', 'glad', 'pleased'],
    }
    
    def label(self, text: str) -> Dict:
        if not text: return {'dominant_emotion': 'neutral', 'emotion_scores': {}}
        text_lower = text.lower()
        scores = {emo: sum(1 for w in words if w in text_lower) for emo, words in self.EMOTION_WORDS.items()}
        dominant = max(scores, key=scores.get) if max(scores.values()) > 0 else 'neutral'
        return {'dominant_emotion': dominant, 'emotion_scores': scores}

class ConversationDynamicsAnalyzer:
    """Step 20, R31: Turn-taking analysis."""
    def analyze(self, df) -> Dict:
        if df is None or len(df) == 0:
            return {'talk_ratio':0, 'turn_count':0, 'words_per_turn':0, 'engagement_slope':0}
        
        try:
            participant_turns = df[~df['speaker'].str.lower().str.contains('ellie', na=False)]
            all_turns = len(df)
            p_turns = len(participant_turns)
            
            p_words = participant_turns['text'].fillna('').str.split().str.len().sum()
            total_words = df['text'].fillna('').str.split().str.len().sum() or 1
            
            if len(participant_turns) >= 3:
                thirds = np.array_split(participant_turns['text'].fillna('').str.split().str.len().values, 3)
                means = [np.mean(t) if len(t)>0 else 0 for t in thirds]
                slope = (means[2] - means[0]) / 2 if len(means) == 3 else 0
            else:
                slope = 0
            
            return {
                'talk_ratio': p_words / total_words,
                'turn_count': p_turns,
                'words_per_turn': p_words / max(p_turns, 1),
                'engagement_slope': slope
            }
        except:
            return {'talk_ratio':0, 'turn_count':0, 'words_per_turn':0, 'engagement_slope':0}

class TextPreprocessor:
    """Complete text pipeline: Steps 12-20, R18-R31."""
    def __init__(self, models, embed_dim=768, device='cuda'):
        self.models = models
        self.embed_dim = embed_dim
        self.device = device
        self.embed_dim = embed_dim
        self.device = device
        
        if TEXT_ENHANCEMENTS_OK:
            self.cleaner = EnhancedTranscriptCleaner()
            self.lang_detect = EnhancedLanguageDetector()
            self.psycho = EnhancedPsycholinguistic()
            self.readability = ComplexityAnalyzer()  # Enhanced readability
            self.sentiment = EnhancedSentiment()
            self.dynamics = EnhancedDynamics()
            self.lexical = LexicalDiversityAnalyzer() # Use local as fallback or update if enhanced exists
            self.emotion = EmotionLabeler() # Local fallback
            if 'Chinese' in str(EnhancedPsycholinguistic): # Duck typing check or attribute check if needed
                 self.chinese_psycho = EnhancedPsycholinguistic # Or separate class if defined
        else:
            self.cleaner = TranscriptCleaner()
            self.lang_detect = LanguageDetector()
            self.psycho = PsycholinguisticExtractor()
            self.lexical = LexicalDiversityAnalyzer()
            self.readability = ReadabilityExtractor()
            self.sentiment = MultilingualSentimentAnalyzer()
            self.emotion = EmotionLabeler()
            self.dynamics = ConversationDynamicsAnalyzer()
        
        if R30_R58_OK:
            self.cat_emotion = CategoricalEmotionLabeler()
            self.text_augmenter = TextAugmenter()
        else:
            self.cat_emotion = None
            self.text_augmenter = None
    
    @torch.no_grad()
    def get_text_embedding(self, text: str, lang: str = 'english') -> np.ndarray:
        model_key = f'text_{lang}'
        if model_key not in self.models.models:
            model_key = 'text_english'
        if model_key not in self.models.models:
            return np.zeros(self.embed_dim)
        
        try:
            tokenizer = self.models.processors[model_key]
            model = self.models.models[model_key]
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        except:
            return np.zeros(self.embed_dim)
    
    def process_text(self, text: str = None, transcript_path: str = None, language: str = None, transcript_df=None, augment: bool = False) -> Dict:
        result = {'text_embedding': np.zeros(self.embed_dim), 'quality_score': 0.0}
        
        if text is None and transcript_path:
            text = self.cleaner.load_transcript(transcript_path)
        
        if not text:
            return result
        
        char_count_raw = len(text)
        
        if augment and self.text_augmenter:
            try:
                text = self.text_augmenter.augment(text)
                result['augmentation_applied'] = True
            except Exception as e:
                print(f"TextAugmenter error: {e}")
        
        cleaned = self.cleaner.clean(text)
        text = cleaned['cleaned_text']
        result.update(cleaned)
        
        char_count_clean = len(text)
        disfluencies_removed = cleaned.get('filler_count', 0)
        
        if not text:
            result['metadata'] = {
                'char_count_raw': char_count_raw,
                'char_count_clean': 0,
                'disfluencies_removed': disfluencies_removed,
                'sentiment_polarity_score': 0.0,
                'lexical_diversity_ttr': 0.0
            }
            return result
        
        if language:
            lang = language
        else:
            lang = self.lang_detect.detect(text)
        result['language'] = lang
        
        result['text_embedding'] = self.get_text_embedding(text, lang)
        
        result.update(self.psycho.extract(text))
        
        result.update(self.lexical.analyze(text))
        
        result.update(self.readability.extract(text))
        
        result.update(self.sentiment.analyze(text, lang))
        
        emotion = self.emotion.label(text)
        result['dominant_emotion'] = emotion['dominant_emotion']
        
        if transcript_df is not None:
            result.update(self.dynamics.analyze(transcript_df))
        
        result['quality_score'] = min(1.0, len(text.split()) / 100)
        
        if self.cat_emotion:
            cat_emotion = self.cat_emotion.label(text)
            result.update({f'cat_{k}': v for k, v in cat_emotion.items()})
        
        result['metadata'] = {
            'char_count_raw': char_count_raw,
            'char_count_clean': char_count_clean,
            'disfluencies_removed': disfluencies_removed,
            'sentiment_polarity_score': float(result.get('sentiment_compound', 0.0)),
            'lexical_diversity_ttr': float(result.get('ttr', 0.0))
        }
        
        return result

print("Text Pipeline loaded: 8 classes + TextPreprocessor")
