#!/usr/bin/env python3
"""
Enhanced Model Training Script for PostyAI Engagement Prediction

This script trains multiple ML models with advanced hyperparameter tuning,
evaluates their performance, and saves the best models for production use.
"""

import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
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
    Perform comprehensive data analysis
    
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
            'q25': df['engagement'].quantile(0.25),
            'q75': df['engagement'].quantile(0.75),
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
    analysis['top_tags'] = tag_counts.head(15).to_dict()
    analysis['unique_tags'] = len(tag_counts)
    
    # Engagement distribution analysis
    engagement_bins = pd.cut(df['engagement'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    analysis['engagement_distribution'] = engagement_bins.value_counts().to_dict()
    
    return analysis


def print_enhanced_analysis(analysis: dict):
    """Print comprehensive data analysis"""
    print("\n" + "="*70)
    print("📊 ENHANCED DATA ANALYSIS REPORT")
    print("="*70)
    
    print(f"\n📈 Dataset Overview:")
    print(f"  Total Posts: {analysis['total_posts']}")
    print(f"  Unique Tags: {analysis['unique_tags']}")
    
    print(f"\n💹 Engagement Statistics:")
    stats = analysis['engagement_stats']
    print(f"  Mean: {stats['mean']:.1f}")
    print(f"  Median: {stats['median']:.1f}")
    print(f"  Std Dev: {stats['std']:.1f}")
    print(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}")
    print(f"  Q25-Q75: {stats['q25']:.1f} - {stats['q75']:.1f}")
    
    print(f"\n🌍 Language Distribution:")
    for lang, count in analysis['language_distribution'].items():
        percentage = (count / analysis['total_posts']) * 100
        print(f"  {lang}: {count} posts ({percentage:.1f}%)")
    
    print(f"\n📏 Line Count Statistics:")
    line_stats = analysis['line_count_stats']
    print(f"  Average: {line_stats['mean']:.1f} lines")
    print(f"  Median: {line_stats['median']:.1f} lines")
    print(f"  Std Dev: {line_stats['std']:.1f}")
    
    print(f"\n🏷️ Top Tags:")
    for i, (tag, count) in enumerate(list(analysis['top_tags'].items())[:10], 1):
        print(f"  {i:2d}. {tag}: {count} posts")
    
    print(f"\n📊 Engagement Distribution:")
    for category, count in analysis['engagement_distribution'].items():
        print(f"  {category}: {count} posts")


def save_enhanced_models(predictor: EngagementPredictor, results: dict, analysis: dict):
    """
    Save trained models and comprehensive metadata
    
    Args:
        predictor: Trained EngagementPredictor instance
        results: Training results dictionary
        analysis: Data analysis results
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save individual models
    for name, result in results.items():
        if result and 'model' in result:
            model_path = f'models/engagement_predictor_{name}.pkl'
            joblib.dump(result['model'], model_path)
            print(f"💾 Saved {name} model to {model_path}")
    
    # Save scalers
    scalers_path = 'models/engagement_predictor_scalers.pkl'
    joblib.dump(predictor.scalers, scalers_path)
    print(f"💾 Saved scalers to {scalers_path}")
    
    # Save feature engineer
    feature_engineer_path = 'models/feature_engineer.pkl'
    joblib.dump(predictor.feature_engineer, feature_engineer_path)
    print(f"💾 Saved feature engineer to {feature_engineer_path}")
    
    # Create comprehensive metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_model_name': predictor.best_model_name,
        'data_analysis': analysis,
        'selected_features': getattr(predictor, 'selected_features', []),
        'total_features': len(getattr(predictor, 'selected_features', [])),
        'results': {}
    }
    
    # Process results for metadata
    for name, result in results.items():
        if result:
            # Convert numpy types to native Python types for JSON serialization
            processed_result = {}
            for key, value in result.items():
                if key == 'model':
                    continue  # Skip model object
                elif key == 'predictions':
                    processed_result[key] = value
                elif isinstance(value, dict):
                    # Handle nested dictionaries (metrics)
                    processed_result[key] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                        for k, v in value.items()
                    }
                elif isinstance(value, (np.integer, np.floating)):
                    processed_result[key] = float(value)
                else:
                    processed_result[key] = value
            
            metadata['results'][name] = processed_result
    
    # Save metadata
    metadata_path = 'models/engagement_predictor_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to {metadata_path}")


def print_model_comparison(results: dict):
    """Print detailed model comparison"""
    print("\n" + "="*70)
    print("🏆 MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    # Sort models by test R² score
    sorted_results = sorted(
        [(name, result) for name, result in results.items() if result],
        key=lambda x: x[1]['test_metrics']['r2'],
        reverse=True
    )
    
    print(f"\n{'Rank':<4} {'Model':<25} {'R² Score':<10} {'RMSE':<10} {'MAE':<10} {'CV Score':<10}")
    print("-" * 70)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        r2 = result['test_metrics']['r2']
        rmse = result['test_metrics']['rmse']
        mae = result['test_metrics']['mae']
        cv_score = result['cv_score_mean']
        
        print(f"{i:<4} {name:<25} {r2:<10.4f} {rmse:<10.1f} {mae:<10.1f} {cv_score:<10.4f}")
    
    # Print best model details
    if sorted_results:
        best_name, best_result = sorted_results[0]
        print(f"\n🥇 BEST MODEL: {best_name}")
        print(f"  R² Score: {best_result['test_metrics']['r2']:.4f}")
        print(f"  RMSE: {best_result['test_metrics']['rmse']:.2f}")
        print(f"  MAE: {best_result['test_metrics']['mae']:.2f}")
        print(f"  Cross-validation: {best_result['cv_score_mean']:.4f} ± {best_result['cv_score_std']:.4f}")
        
        # Print feature importance if available
        if best_result.get('feature_importance'):
            print(f"\n🔍 TOP 10 FEATURES FOR {best_name}:")
            importance = best_result['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, score) in enumerate(sorted_features[:10], 1):
                feature_display = feature.replace('_', ' ').title()
                print(f"  {i:2d}. {feature_display:<30} {score:.4f}")


def main():
    """Main training function with enhanced pipeline"""
    print("🚀 PostyAI Enhanced Model Training Pipeline")
    print("=" * 50)
    
    # Load and analyze data
    print("\n📥 Loading training data...")
    posts = load_training_data()
    
    if not posts:
        print("❌ No training data available. Exiting.")
        return
    
    # Perform enhanced data analysis
    print("\n🔍 Analyzing dataset...")
    analysis = analyze_data(posts)
    print_enhanced_analysis(analysis)
    
    # Initialize predictor
    print("\n🤖 Initializing enhanced ML predictor...")
    predictor = EngagementPredictor()
    
    # Prepare data
    print("\n⚙️ Preparing features...")
    X, y = predictor.prepare_data(posts)
    print(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} posts")
    
    # Train models with hyperparameter tuning
    print("\n🏋️ Training models with hyperparameter optimization...")
    results = predictor.train_with_hyperparameter_tuning(X, y, test_size=0.2)
    
    # Print comparison
    print_model_comparison(results)
    
    # Save models and results
    print("\n💾 Saving trained models...")
    save_enhanced_models(predictor, results, analysis)
    
    # Generate feature importance report
    if predictor.best_model_name and predictor.best_model_name in results:
        best_result = results[predictor.best_model_name]
        if best_result.get('feature_importance'):
            importance_report = create_feature_importance_report(
                best_result['feature_importance'], 
                top_n=20
            )
            print(importance_report)
    
    print("\n🎉 Enhanced training pipeline completed successfully!")
    print(f"🏆 Best model: {predictor.best_model_name}")
    print("💡 Models are ready for production use!")


if __name__ == "__main__":
    main()
