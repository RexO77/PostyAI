"""
Advanced Feature Engineering for Social Media Post Engagement Prediction

This module extracts comprehensive features from post content including:
- Text statistics and readability
- Linguistic patterns
- Emotional indicators
- Structural features
- Content type indicators
- Advanced semantic features
- Temporal and contextual features
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Tuple
import textstat
from collections import Counter
import hashlib
from datetime import datetime


class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering for social media posts with enhanced features
    """
    
    def __init__(self):
        self.emoji_pattern = re.compile(r'[😀-🙏🌀-🗿🚀-🛿🇀-🇿✂-➰Ⓜ-🔽]')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Enhanced patterns for better feature extraction
        self.cashtag_pattern = re.compile(r'\$[A-Z]{1,5}')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[\d]{1,14}')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Engagement words and phrases
        self.engagement_words = {
            'call_to_action': ['comment', 'share', 'like', 'follow', 'subscribe', 'click', 'join', 'participate'],
            'curiosity': ['secret', 'revealed', 'discover', 'hidden', 'unknown', 'mystery', 'surprise'],
            'urgency': ['now', 'today', 'limited', 'hurry', 'urgent', 'deadline', 'expires'],
            'social_proof': ['everyone', 'popular', 'trending', 'viral', 'thousands', 'millions'],
            'emotional_triggers': ['amazing', 'incredible', 'shocking', 'unbelievable', 'stunning', 'powerful']
        }
        
        # Industry-specific keywords
        self.industry_keywords = {
            'tech': ['AI', 'ML', 'blockchain', 'crypto', 'software', 'developer', 'coding', 'programming'],
            'business': ['strategy', 'growth', 'revenue', 'profit', 'marketing', 'sales', 'leadership'],
            'career': ['job', 'hiring', 'interview', 'resume', 'career', 'opportunity', 'promotion'],
            'personal': ['motivation', 'inspiration', 'success', 'goals', 'achievement', 'mindset']
        }
    
    def extract_all_features(self, text: str, tags: List[str] = None, language: str = 'English') -> Dict[str, float]:
        """
        Extract all features from a post including new enhanced features
        
        Args:
            text: Post content
            tags: List of post tags
            language: Post language
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Original features
        features.update(self._extract_text_stats(text))
        features.update(self._extract_readability_features(text))
        features.update(self._extract_linguistic_features(text))
        features.update(self._extract_emotional_features(text))
        features.update(self._extract_structural_features(text))
        features.update(self._extract_social_features(text))
        features.update(self._extract_content_type_features(text, tags))
        features.update(self._extract_language_features(text, language))
        
        # NEW ENHANCED FEATURES
        features.update(self._extract_engagement_features(text))
        features.update(self._extract_semantic_features(text))
        features.update(self._extract_timing_features(text))
        features.update(self._extract_visual_features(text))
        features.update(self._extract_complexity_features(text))
        features.update(self._extract_industry_features(text, tags))
        
        return features
    
    def _extract_text_stats(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'char_count_no_spaces': len(text.replace(' ', '')),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_word_ratio': len(set(words)) / max(len(words), 1) if words else 0,
        }
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability and complexity features"""
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'syllable_count': textstat.syllable_count(text),
            }
        except:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'automated_readability_index': 0,
                'coleman_liau_index': 0,
                'syllable_count': 0,
            }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic patterns"""
        # Punctuation analysis
        punct_count = sum(1 for char in text if char in string.punctuation)
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        # Case analysis
        upper_count = sum(1 for char in text if char.isupper())
        lower_count = sum(1 for char in text if char.islower())
        
        # Word patterns
        words = text.split()
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        
        return {
            'punctuation_ratio': punct_count / max(len(text), 1),
            'question_mark_count': question_marks,
            'exclamation_mark_count': exclamation_marks,
            'upper_case_ratio': upper_count / max(len(text), 1),
            'capitalized_word_ratio': capitalized_words / max(len(words), 1),
            'digit_count': sum(1 for char in text if char.isdigit()),
        }
    
    def _extract_emotional_features(self, text: str) -> Dict[str, float]:
        """Extract emotional indicators"""
        # Positive emotion words
        positive_words = ['great', 'awesome', 'amazing', 'fantastic', 'excellent', 'love', 'happy', 'excited', 'wonderful', 'brilliant']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'frustrated', 'disappointed', 'worst', 'horrible']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Emphasis patterns
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        repeated_punct = len(re.findall(r'[!?]{2,}', text))
        
        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'emotion_ratio': (positive_count - negative_count) / max(len(text.split()), 1),
            'caps_word_count': caps_words,
            'repeated_punctuation': repeated_punct,
        }
    
    def _extract_structural_features(self, text: str) -> Dict[str, float]:
        """Extract content structure features"""
        lines = text.split('\n')
        
        # Paragraph analysis
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # List indicators
        bullet_indicators = len(re.findall(r'^\s*[•\-\*]\s', text, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
        
        return {
            'line_count': len([line for line in lines if line.strip()]),
            'paragraph_count': len(paragraphs),
            'avg_line_length': np.mean([len(line) for line in lines if line.strip()]) if lines else 0,
            'bullet_point_count': bullet_indicators,
            'numbered_list_count': numbered_lists,
            'newline_ratio': text.count('\n') / max(len(text), 1),
        }
    
    def _extract_social_features(self, text: str) -> Dict[str, float]:
        """Extract social media specific features"""
        # Emojis
        emojis = self.emoji_pattern.findall(text)
        
        # Hashtags and mentions
        hashtags = self.hashtag_pattern.findall(text)
        mentions = self.mention_pattern.findall(text)
        
        # URLs
        urls = self.url_pattern.findall(text)
        
        return {
            'emoji_count': len(emojis),
            'hashtag_count': len(hashtags),
            'mention_count': len(mentions),
            'url_count': len(urls),
            'social_element_ratio': (len(emojis) + len(hashtags) + len(mentions)) / max(len(text.split()), 1),
        }
    
    def _extract_content_type_features(self, text: str, tags: List[str] = None) -> Dict[str, float]:
        """Extract content type indicators"""
        text_lower = text.lower()
        
        # Content type keywords
        question_indicators = ['how', 'what', 'why', 'when', 'where', 'which', 'who']
        tip_indicators = ['tip', 'tips', 'advice', 'guide', 'tutorial', 'steps']
        story_indicators = ['story', 'experience', 'journey', 'happened', 'remember']
        motivation_indicators = ['motivat', 'inspir', 'succeed', 'achieve', 'dream', 'goal']
        
        question_score = sum(1 for word in question_indicators if word in text_lower)
        tip_score = sum(1 for word in tip_indicators if word in text_lower)
        story_score = sum(1 for word in story_indicators if word in text_lower)
        motivation_score = sum(1 for word in motivation_indicators if word in text_lower)
        
        # Tag diversity
        tag_count = len(tags) if tags else 0
        
        return {
            'question_content_score': question_score,
            'tip_content_score': tip_score,
            'story_content_score': story_score,
            'motivation_content_score': motivation_score,
            'tag_count': tag_count,
            'is_question_post': 1 if text.strip().endswith('?') else 0,
        }
    
    def _extract_language_features(self, text: str, language: str) -> Dict[str, float]:
        """Extract language-specific features"""
        return {
            'is_english': 1 if language.lower() == 'english' else 0,
            'is_hinglish': 1 if language.lower() == 'hinglish' else 0,
            'is_multilingual': 1 if language.lower() not in ['english', 'hinglish'] else 0,
        }
    
    def _extract_engagement_features(self, text: str) -> Dict[str, float]:
        """Extract features specifically designed to predict engagement"""
        text_lower = text.lower()
        features = {}
        
        # Call-to-action detection
        cta_score = sum(1 for word in self.engagement_words['call_to_action'] if word in text_lower)
        features['call_to_action_score'] = cta_score
        
        # Curiosity indicators
        curiosity_score = sum(1 for word in self.engagement_words['curiosity'] if word in text_lower)
        features['curiosity_score'] = curiosity_score
        
        # Urgency indicators
        urgency_score = sum(1 for word in self.engagement_words['urgency'] if word in text_lower)
        features['urgency_score'] = urgency_score
        
        # Social proof indicators
        social_proof_score = sum(1 for word in self.engagement_words['social_proof'] if word in text_lower)
        features['social_proof_score'] = social_proof_score
        
        # Emotional triggers
        emotional_trigger_score = sum(1 for word in self.engagement_words['emotional_triggers'] if word in text_lower)
        features['emotional_trigger_score'] = emotional_trigger_score
        
        # Question engagement potential
        questions = text.count('?')
        features['question_engagement_potential'] = min(questions, 3) * 2  # Cap at 3 questions
        
        # Direct address indicators
        direct_address = len(re.findall(r'\byou\b|\byour\b', text_lower))
        features['direct_address_count'] = direct_address
        
        # Lists and actionable content
        list_indicators = len(re.findall(r'^\s*[\d\-\•]\s', text, re.MULTILINE))
        features['list_structure_score'] = min(list_indicators, 10)  # Cap for normalization
        
        return features
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic and content-meaning features"""
        words = text.split()
        text_lower = text.lower()
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / max(total_words, 1)
        
        # Lexical density (content words vs function words)
        function_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        content_words = [word for word in words if word.lower() not in function_words]
        lexical_density = len(content_words) / max(total_words, 1)
        
        # Expertise indicators
        expertise_words = ['expert', 'experience', 'years', 'professional', 'specialist', 'certified']
        expertise_score = sum(1 for word in expertise_words if word in text_lower)
        
        # Personal experience indicators
        personal_words = ['i', 'my', 'me', 'myself', 'personal', 'experience', 'learned']
        personal_score = sum(1 for word in personal_words if word in text_lower)
        
        # Abstract vs concrete language
        abstract_words = ['concept', 'idea', 'thought', 'principle', 'theory', 'philosophy']
        concrete_words = ['example', 'step', 'action', 'result', 'outcome', 'method']
        abstract_score = sum(1 for word in abstract_words if word in text_lower)
        concrete_score = sum(1 for word in concrete_words if word in text_lower)
        
        return {
            'type_token_ratio': ttr,
            'lexical_density': lexical_density,
            'expertise_indicators': expertise_score,
            'personal_experience_score': personal_score,
            'abstract_language_score': abstract_score,
            'concrete_language_score': concrete_score,
            'abstract_concrete_ratio': abstract_score / max(concrete_score, 1)
        }
    
    def _extract_timing_features(self, text: str) -> Dict[str, float]:
        """Extract temporal and time-sensitive features"""
        text_lower = text.lower()
        
        # Time-sensitive words
        time_words = ['today', 'yesterday', 'tomorrow', 'now', 'soon', 'recently', 'just', 'currently']
        time_sensitivity = sum(1 for word in time_words if word in text_lower)
        
        # Trend indicators
        trend_words = ['trending', 'viral', 'popular', 'hot', 'breaking', 'new', 'latest', 'update']
        trend_score = sum(1 for word in trend_words if word in text_lower)
        
        # Future vs past orientation
        future_words = ['will', 'going', 'plan', 'future', 'upcoming', 'next', 'soon']
        past_words = ['was', 'were', 'had', 'did', 'happened', 'before', 'ago', 'previous']
        future_score = sum(1 for word in future_words if word in text_lower)
        past_score = sum(1 for word in past_words if word in text_lower)
        
        return {
            'time_sensitivity_score': time_sensitivity,
            'trend_indicators': trend_score,
            'future_orientation': future_score,
            'past_orientation': past_score,
            'temporal_focus_ratio': future_score / max(past_score, 1)
        }
    
    def _extract_visual_features(self, text: str) -> Dict[str, float]:
        """Extract features related to visual appeal and formatting"""
        # Emoji diversity
        emojis = self.emoji_pattern.findall(text)
        emoji_diversity = len(set(emojis)) / max(len(emojis), 1) if emojis else 0
        
        # Visual separators
        visual_separators = len(re.findall(r'[-=*]{3,}', text))
        
        # Formatted text (bold, italic indicators)
        bold_text = len(re.findall(r'\*\*.*?\*\*', text))
        italic_text = len(re.findall(r'\*.*?\*', text))
        
        # Special characters for emphasis
        special_chars = len(re.findall(r'[►→⭐✨💡🔥]', text))
        
        # Line breaks for readability
        line_breaks = text.count('\n')
        words = len(text.split())
        line_break_ratio = line_breaks / max(words, 1)
        
        return {
            'emoji_diversity': emoji_diversity,
            'visual_separators': visual_separators,
            'bold_text_indicators': bold_text,
            'italic_text_indicators': italic_text,
            'special_emphasis_chars': special_chars,
            'line_break_ratio': line_break_ratio,
            'visual_appeal_score': emoji_diversity + (special_chars * 0.1) + (visual_separators * 0.2)
        }
    
    def _extract_complexity_features(self, text: str) -> Dict[str, float]:
        """Extract features related to content complexity and cognitive load"""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Sentence complexity
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        
        # Vocabulary complexity (longer words indicate higher complexity)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        long_words = len([word for word in words if len(word) > 6])
        long_word_ratio = long_words / max(len(words), 1)
        
        # Punctuation complexity
        complex_punct = len(re.findall(r'[;:()[\]{}"]', text))
        punct_complexity = complex_punct / max(len(text), 1)
        
        # Information density
        unique_concepts = len(set(word.lower() for word in words if len(word) > 3))
        info_density = unique_concepts / max(len(words), 1)
        
        return {
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_word_length': avg_word_length,
            'long_word_ratio': long_word_ratio,
            'punctuation_complexity': punct_complexity,
            'information_density': info_density,
            'cognitive_load_score': (avg_words_per_sentence / 20) + long_word_ratio + punct_complexity
        }
    
    def _extract_industry_features(self, text: str, tags: List[str] = None) -> Dict[str, float]:
        """Extract industry and domain-specific features"""
        text_lower = text.lower()
        features = {}
        
        # Industry classification scores
        for industry, keywords in self.industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            features[f'{industry}_industry_score'] = score
        
        # Professional vs casual tone
        professional_words = ['strategy', 'implement', 'optimize', 'analyze', 'evaluate', 'professional']
        casual_words = ['awesome', 'cool', 'amazing', 'love', 'fun', 'great']
        prof_score = sum(1 for word in professional_words if word in text_lower)
        casual_score = sum(1 for word in casual_words if word in text_lower)
        
        features['professional_tone_score'] = prof_score
        features['casual_tone_score'] = casual_score
        features['tone_formality_ratio'] = prof_score / max(casual_score, 1)
        
        # Tag diversity and relevance
        if tags:
            features['tag_diversity'] = len(set(tags))
            # Check tag relevance to content
            tag_content_overlap = sum(1 for tag in tags if tag.lower() in text_lower)
            features['tag_content_relevance'] = tag_content_overlap / max(len(tags), 1)
        else:
            features['tag_diversity'] = 0
            features['tag_content_relevance'] = 0
        
        return features

    def create_feature_dataframe(self, posts_data: List[Dict]) -> pd.DataFrame:
        """
        Create a complete feature dataframe from posts data
        
        Args:
            posts_data: List of post dictionaries
            
        Returns:
            DataFrame with all engineered features
        """
        features_list = []
        
        for post in posts_data:
            # Extract features
            features = self.extract_all_features(
                text=post['text'],
                tags=post.get('tags', []),
                language=post.get('language', 'English')
            )
            
            # Add target variable and metadata
            features['engagement'] = post['engagement']
            features['line_count'] = post.get('line_count', 0)
            features['language'] = post.get('language', 'English')
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Create a dummy text to extract feature names
        dummy_features = self.extract_all_features("Sample text for feature extraction.", ["sample"], "English")
        return list(dummy_features.keys())


def create_feature_importance_report(feature_importance: Dict[str, float], top_n: int = 20) -> str:
    """
    Create a formatted feature importance report
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_n: Number of top features to include
        
    Returns:
        Formatted string report
    """
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    report = f"\n🔍 TOP {top_n} MOST IMPORTANT FEATURES FOR ENGAGEMENT PREDICTION\n"
    report += "=" * 70 + "\n\n"
    
    for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
        # Format feature name
        feature_display = feature.replace('_', ' ').title()
        bar_length = int(importance * 50)  # Scale for visual bar
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        report += f"{i:2d}. {feature_display:<30} │{bar}│ {importance:.4f}\n"
    
    return report


if __name__ == "__main__":
    # Demo usage
    fe = AdvancedFeatureEngineer()
    
    sample_text = """
    🎯 5 Tips for Better UX Design:
    
    1. Keep it simple
    2. Test with real users
    3. Focus on accessibility
    4. Use consistent patterns
    
    What's your favorite UX principle? 🤔
    
    #UXDesign #DesignTips
    """
    
    features = fe.extract_all_features(sample_text, ["UX Design", "Tips"], "English")
    print("Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")