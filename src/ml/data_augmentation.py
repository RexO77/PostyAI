"""
Data Augmentation System for PostyAI

This module generates synthetic training data to improve model performance
by creating variations of existing posts while preserving engagement patterns.
"""

import json
import random
import re
import numpy as np
from typing import List, Dict, Tuple
import copy
from collections import defaultdict


class DataAugmentor:
    """
    Generate synthetic training data through various augmentation techniques
    """
    
    def __init__(self):
        # Synonym dictionaries for text variation
        self.synonyms = {
            'amazing': ['incredible', 'fantastic', 'awesome', 'outstanding', 'remarkable'],
            'great': ['excellent', 'wonderful', 'fantastic', 'superb', 'brilliant'],
            'good': ['nice', 'fine', 'decent', 'solid', 'quality'],
            'bad': ['poor', 'terrible', 'awful', 'horrible', 'dreadful'],
            'important': ['crucial', 'vital', 'essential', 'significant', 'critical'],
            'big': ['large', 'huge', 'massive', 'enormous', 'giant'],
            'small': ['tiny', 'little', 'minor', 'minimal', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'immediate'],
            'slow': ['gradual', 'steady', 'measured', 'deliberate', 'careful'],
            'new': ['fresh', 'recent', 'latest', 'modern', 'current'],
            'old': ['traditional', 'classic', 'established', 'proven', 'time-tested']
        }
        
        # Emoji variations
        self.emoji_groups = {
            'positive': ['😊', '🎉', '✨', '👍', '🔥', '💪', '🚀', '⭐'],
            'thinking': ['🤔', '💭', '🧠', '💡'],
            'professional': ['💼', '📊', '📈', '🎯', '📋'],
            'celebration': ['🎊', '🥳', '🎈', '🏆', '🥇']
        }
        
        # Phrase templates for content generation
        self.question_starters = [
            "What's your take on",
            "How do you handle",
            "What would you do if",
            "Have you ever wondered",
            "What's the best way to"
        ]
        
        self.conclusion_phrases = [
            "What are your thoughts?",
            "Share your experience below!",
            "Let me know in the comments!",
            "What would you add to this list?",
            "Tag someone who needs to see this!"
        ]
    
    def augment_dataset(self, posts: List[Dict], target_size: int = 300) -> List[Dict]:
        """
        Augment the dataset to reach target size while preserving distribution
        
        Args:
            posts: Original posts list
            target_size: Target number of posts after augmentation
            
        Returns:
            Augmented posts list
        """
        print(f"🔄 Augmenting dataset from {len(posts)} to {target_size} posts...")
        
        if len(posts) >= target_size:
            print("✅ Dataset already meets target size")
            return posts
        
        augmented_posts = posts.copy()
        posts_to_generate = target_size - len(posts)
        
        # Analyze original distribution
        distribution = self._analyze_distribution(posts)
        
        # Generate new posts while maintaining distribution
        for i in range(posts_to_generate):
            # Select a random post as template
            template_post = random.choice(posts)
            
            # Apply augmentation technique
            technique = random.choice([
                'synonym_replacement',
                'emoji_variation',
                'structure_modification',
                'style_transfer',
                'content_expansion'
            ])
            
            try:
                new_post = self._apply_augmentation(template_post, technique)
                
                # Adjust engagement based on changes made
                new_post['engagement'] = self._estimate_engagement(new_post, template_post)
                
                augmented_posts.append(new_post)
                
                if (i + 1) % 50 == 0:
                    print(f"  Generated {i + 1}/{posts_to_generate} posts...")
                    
            except Exception as e:
                print(f"  ⚠️ Error generating post {i+1}: {e}")
                continue
        
        print(f"✅ Dataset augmented to {len(augmented_posts)} posts")
        return augmented_posts
    
    def _analyze_distribution(self, posts: List[Dict]) -> Dict:
        """Analyze the distribution of posts by various attributes"""
        distribution = {
            'language': defaultdict(int),
            'line_count': defaultdict(int),
            'tags': defaultdict(int),
            'engagement_ranges': defaultdict(int)
        }
        
        for post in posts:
            distribution['language'][post['language']] += 1
            distribution['line_count'][post['line_count']] += 1
            
            # Engagement ranges
            engagement = post['engagement']
            if engagement < 100:
                range_key = 'low'
            elif engagement < 500:
                range_key = 'medium'
            elif engagement < 1000:
                range_key = 'high'
            else:
                range_key = 'very_high'
            distribution['engagement_ranges'][range_key] += 1
            
            # Tags
            for tag in post.get('tags', []):
                distribution['tags'][tag] += 1
        
        return distribution
    
    def _apply_augmentation(self, template_post: Dict, technique: str) -> Dict:
        """Apply specific augmentation technique to create new post"""
        new_post = copy.deepcopy(template_post)
        text = new_post['text']
        
        if technique == 'synonym_replacement':
            new_post['text'] = self._replace_synonyms(text)
            
        elif technique == 'emoji_variation':
            new_post['text'] = self._vary_emojis(text)
            
        elif technique == 'structure_modification':
            new_post['text'] = self._modify_structure(text)
            
        elif technique == 'style_transfer':
            new_post['text'] = self._transfer_style(text, new_post)
            
        elif technique == 'content_expansion':
            new_post['text'] = self._expand_content(text)
        
        # Update metadata
        new_post['line_count'] = len([line for line in new_post['text'].split('\n') if line.strip()])
        
        return new_post
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms"""
        words = text.split()
        modified_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            
            # Check if word has synonyms
            if word_lower in self.synonyms:
                # 30% chance to replace with synonym
                if random.random() < 0.3:
                    synonym = random.choice(self.synonyms[word_lower])
                    # Preserve capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    modified_words.append(word.replace(word_lower, synonym))
                else:
                    modified_words.append(word)
            else:
                modified_words.append(word)
        
        return ' '.join(modified_words)
    
    def _vary_emojis(self, text: str) -> str:
        """Modify emojis while maintaining emotional context"""
        # Find existing emojis
        emoji_pattern = re.compile(r'[😀-🙏🌀-🗿🚀-🛿🇀-🇿✂-➰Ⓜ-🔽]')
        emojis = emoji_pattern.findall(text)
        
        if not emojis:
            # Add some emojis if none exist
            if random.random() < 0.4:  # 40% chance to add emojis
                emoji_group = random.choice(list(self.emoji_groups.keys()))
                emoji = random.choice(self.emoji_groups[emoji_group])
                
                # Add at the end
                if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
                    text += f" {emoji}"
                else:
                    text = text[:-1] + f" {emoji}" + text[-1]
        else:
            # Replace existing emojis with similar ones
            for emoji in emojis:
                if random.random() < 0.5:  # 50% chance to replace
                    # Find appropriate replacement
                    for group_name, group_emojis in self.emoji_groups.items():
                        if emoji in group_emojis:
                            new_emoji = random.choice(group_emojis)
                            text = text.replace(emoji, new_emoji, 1)
                            break
        
        return text
    
    def _modify_structure(self, text: str) -> str:
        """Modify the structure of the post"""
        lines = text.split('\n')
        
        # Randomly apply structural changes
        if random.random() < 0.3:  # Add line breaks
            sentences = text.split('. ')
            if len(sentences) > 2:
                # Break long paragraphs
                mid_point = len(sentences) // 2
                return '. '.join(sentences[:mid_point]) + '.\n\n' + '. '.join(sentences[mid_point:])
        
        elif random.random() < 0.3:  # Add enumeration
            if len(lines) >= 3 and not any(line.strip().startswith(('1.', '2.', '-', '•')) for line in lines):
                # Convert to numbered list
                new_lines = [lines[0]]  # Keep first line as intro
                for i, line in enumerate(lines[1:], 1):
                    if line.strip():
                        new_lines.append(f"{i}. {line.strip()}")
                return '\n'.join(new_lines)
        
        elif random.random() < 0.4:  # Add question at the end
            if not text.endswith('?'):
                question = random.choice(self.conclusion_phrases)
                return text + f"\n\n{question}"
        
        return text
    
    def _transfer_style(self, text: str, post: Dict) -> str:
        """Transfer style between different types of posts"""
        language = post.get('language', 'English')
        
        # Professional to casual style transfer
        if 'professional' in text.lower() or 'strategy' in text.lower():
            # Make it more casual
            casual_replacements = {
                'implement': 'try out',
                'utilize': 'use',
                'optimize': 'make better',
                'furthermore': 'also',
                'therefore': 'so'
            }
            
            for formal, casual in casual_replacements.items():
                text = text.replace(formal, casual)
        
        # Add language-specific variations
        if language == 'Hinglish':
            # Add some Hindi words
            hinglish_additions = {
                'really': 'bilkul',
                'very': 'bahut',
                'good': 'accha',
                'right': 'sahi'
            }
            
            for eng, hindi in hinglish_additions.items():
                if random.random() < 0.2:  # 20% chance
                    text = text.replace(eng, hindi)
        
        return text
    
    def _expand_content(self, text: str) -> str:
        """Expand content with additional context"""
        expansion_templates = [
            "Here's why this matters:",
            "My experience with this:",
            "The key insight:",
            "What I've learned:",
            "Pro tip:"
        ]
        
        insights = [
            "It's all about consistency and patience.",
            "Small steps lead to big changes.",
            "Focus on progress, not perfection.",
            "Learn from failures and keep moving forward.",
            "Surround yourself with supportive people."
        ]
        
        if random.random() < 0.4:  # 40% chance to expand
            template = random.choice(expansion_templates)
            insight = random.choice(insights)
            return text + f"\n\n{template}\n{insight}"
        
        return text
    
    def _estimate_engagement(self, new_post: Dict, template_post: Dict) -> int:
        """Estimate engagement for synthetic post based on template"""
        base_engagement = template_post['engagement']
        
        # Apply variations based on changes made
        variation_factor = random.uniform(0.8, 1.2)  # ±20% variation
        
        # Adjust based on content features
        new_text = new_post['text']
        
        # More emojis typically increase engagement
        emoji_count = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿🇀-🇿✂-➰Ⓜ-🔽]', new_text))
        if emoji_count > 0:
            variation_factor *= 1.1
        
        # Questions increase engagement
        if '?' in new_text:
            variation_factor *= 1.15
        
        # Lists and structure increase engagement
        if any(new_text.count(marker) > 0 for marker in ['1.', '2.', '-', '•']):
            variation_factor *= 1.1
        
        # Apply randomness to avoid overfitting
        final_engagement = int(base_engagement * variation_factor)
        
        # Ensure reasonable bounds
        return max(1, min(final_engagement, base_engagement * 2))
    
    def save_augmented_data(self, posts: List[Dict], output_path: str = "data/augmented_posts.json"):
        """Save augmented dataset"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(posts)} augmented posts to {output_path}")


def main():
    """Demonstrate data augmentation"""
    print("🚀 PostyAI Data Augmentation Pipeline")
    print("=" * 40)
    
    # Load original data
    with open('data/processed_posts.json', 'r', encoding='utf-8') as f:
        original_posts = json.load(f)
    
    print(f"📥 Loaded {len(original_posts)} original posts")
    
    # Initialize augmentor
    augmentor = DataAugmentor()
    
    # Augment dataset
    augmented_posts = augmentor.augment_dataset(original_posts, target_size=300)
    
    # Save augmented data
    augmentor.save_augmented_data(augmented_posts)
    
    print("✅ Data augmentation completed successfully!")


if __name__ == "__main__":
    main()
