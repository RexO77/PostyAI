#!/usr/bin/env python3
"""
Model Training Script for PostyAI Engagement Prediction

This script trains multiple ML models on the post engagement data,
evaluates their performance, and saves the best models for production use.
"""

import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from src.ml.feature_engineering import AdvancedFeatureEngineer, create_feature_importance_report
from src.ml.engagement_predictor import EngagementPredictor


def load_training_data(data_path: str = "data/processed_posts.json") -> list:
    """
    Load training data from JSON file
    
    Args:
        data_path: Path to the processed posts JSON file
        
    Returns:
        List of post dictionaries
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        
        print(f"✅ Loaded {len(posts)} posts from {data_path}")
        return posts
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {data_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return []


def analyze_data(posts: list) -> dict:
    """
    Perform basic data analysis
    
    Args:
        posts: List of post dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    if not posts:
        return {}
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(posts)
    
    analysis = {
        'total_posts': len(posts),
        'engagement_stats': {
            'mean': df['engagement'].mean(),
            'median': df['engagement'].median(),
            'std': df['engagement'].std(),
            'min': df['engagement'].min(),
            'max': df['engagement'].max(),
        },
        'language_distribution': df['language'].value_counts().to_dict(),
        'line_count_stats': {
            'mean': df['line_count'].mean(),
            'median': df['line_count'].median(),
            'std': df['line_count'].std(),
        }
    }
    
    # Analyze tags
    all_tags = []
    for post in posts:
        all_tags.extend(post.get('tags', []))
    
    tag_counts = pd.Series(all_tags).value_counts()
    analysis['top_tags'] = tag_counts.head(10).to_dict()
    analysis['unique_tags'] = len(tag_counts)
    
    return analysis


def print_data_analysis(analysis: dict):
    """Print formatted data analysis"""
    print("\n" + "="*60)
    print("📊 DATA ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📈 Dataset Overview:")
    print(f"  Total Posts: {analysis['total_posts']}")
    print(f"  Unique Tags: {analysis['unique_tags']}")
    
    print(f"\n🎯 Engagement Statistics:")
    eng_stats = analysis['engagement_stats']
    print(f"  Mean: {eng_stats['mean']:.1f}")
    print(f"  Median: {eng_stats['median']:.1f}")
    print(f"  Std Dev: {eng_stats['std']:.1f}")
    print(f"  Range: {eng_stats['min']} - {eng_stats['max']}")
    
    print(f"\n🌍 Language Distribution:")
    for lang, count in analysis['language_distribution'].items():
        print(f"  {lang}: {count} posts ({count/analysis['total_posts']*100:.1f}%)")
    
    print(f"\n🏷️ Top Tags:")
    for tag, count in analysis['top_tags'].items():
        print(f"  {tag}: {count} posts")


def main():
    """Main training pipeline"""
    print("🚀 PostyAI ML Training Pipeline")
    print("="*50)
    
    # Step 1: Load data
    print("\n1️⃣ Loading training data...")
    posts = load_training_data()
    
    if not posts:
        print("❌ No data available for training. Exiting.")
        return
    
    # Step 2: Data analysis
    print("\n2️⃣ Analyzing data...")
    analysis = analyze_data(posts)
    print_data_analysis(analysis)
    
    # Check if we have enough data
    if len(posts) < 20:
        print("⚠️ Warning: Very small dataset. Results may not be reliable.")
    
    # Step 3: Feature engineering
    print("\n3️⃣ Extracting features...")
    feature_engineer = AdvancedFeatureEngineer()
    
    # Create feature dataframe
    df = feature_engineer.create_feature_dataframe(posts)
    print(f"✅ Extracted {df.shape[1]} features from {df.shape[0]} posts")
    
    # Show feature summary
    feature_names = [col for col in df.columns if col not in ['engagement', 'line_count', 'language']]
    print(f"📋 Features extracted: {len(feature_names)}")
    print(f"   Text Stats: word_count, char_count, sentence_count, etc.")
    print(f"   Readability: flesch_reading_ease, gunning_fog, etc.")
    print(f"   Linguistic: punctuation_ratio, caps_word_count, etc.")
    print(f"   Social Media: emoji_count, hashtag_count, etc.")
    print(f"   Content Type: question_content_score, tip_content_score, etc.")
    
    # Step 4: Train models
    print("\n4️⃣ Training ML models...")
    predictor = EngagementPredictor()
    
    # Prepare data
    X, y = predictor.prepare_data(posts)
    print(f"📊 Training set: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎯 Target range: {y.min():.0f} - {y.max():.0f} engagement points")
    
    # Train all models
    results = predictor.train_all_models(X, y, test_size=0.2)
    
    # Step 5: Model comparison
    print("\n5️⃣ Model Performance Results:")
    comparison_report = predictor.create_model_comparison_report()
    print(comparison_report)
    
    # Step 6: Feature importance analysis
    print("\n6️⃣ Feature Importance Analysis:")
    if predictor.best_model_name and predictor.results[predictor.best_model_name]['feature_importance']:
        importance = predictor.results[predictor.best_model_name]['feature_importance']
        importance_report = create_feature_importance_report(importance, top_n=15)
        print(importance_report)
    else:
        print("❌ No feature importance available for the best model")
    
    # Step 7: Save models
    print("\n7️⃣ Saving trained models...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    model_filepath = models_dir / "engagement_predictor.pkl"
    predictor.save_models(str(model_filepath))
    
    # Save feature engineering pipeline
    feature_engineer_path = models_dir / "feature_engineer.pkl"
    import joblib
    joblib.dump(feature_engineer, feature_engineer_path)
    print(f"✅ Feature engineer saved to {feature_engineer_path}")
    
    # Step 8: Test prediction on sample posts
    print("\n8️⃣ Testing predictions on sample posts...")
    
    sample_tests = [
        {
            'text': "🎯 Top 5 tips for career growth:\n1. Network actively\n2. Learn continuously\n3. Seek feedback\n4. Take initiative\n5. Build personal brand\nWhat's your #1 career tip? 💼",
            'tags': ['Career', 'Tips', 'Professional Growth'],
            'language': 'English'
        },
        {
            'text': "Just completed my first machine learning project! 🤖 Learned so much about data preprocessing and model evaluation. Excited to apply these skills in real-world scenarios!",
            'tags': ['Machine Learning', 'Achievement'],
            'language': 'English'
        },
        {
            'text': "LinkedIn influencers be like: 'I was rejected 100 times, now I'm CEO' 😅",
            'tags': ['Humor', 'LinkedIn'],
            'language': 'English'
        }
    ]
    
    for i, test_post in enumerate(sample_tests, 1):
        try:
            prediction = predictor.predict_engagement(
                test_post['text'], 
                test_post['tags'], 
                test_post['language']
            )
            
            print(f"\n📝 Sample Post {i}:")
            print(f"   Text: {test_post['text'][:100]}...")
            print(f"   Predicted Engagement: {prediction['predicted_engagement']:.1f}")
            print(f"   Confidence Interval: {prediction['confidence_interval'][0]:.1f} - {prediction['confidence_interval'][1]:.1f}")
            print(f"   Model Used: {prediction['model_used']}")
            
        except Exception as e:
            print(f"❌ Error predicting engagement for sample {i}: {str(e)}")
    
    # Step 9: Create summary
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✅ Trained {len(predictor.models)} different algorithms")
    print(f"✅ Best model: {predictor.best_model_name}")
    print(f"✅ Best R² score: {predictor.results[predictor.best_model_name]['test_metrics']['r2']:.4f}")
    print(f"✅ Models saved to: {models_dir}")
    print(f"✅ Ready for production deployment!")
    
    # Create integration instructions
    print(f"\n🔧 INTEGRATION INSTRUCTIONS:")
    print(f"1. Install dependencies: pip install -r requirements.txt")
    print(f"2. Models are saved in: {models_dir}/")
    print(f"3. Use the Flask endpoints to make predictions")
    print(f"4. Monitor model performance in production")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 