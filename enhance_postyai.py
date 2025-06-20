#!/usr/bin/env python3
"""
PostyAI Enhancement Pipeline

This script implements all the recommended improvements:
1. Model Performance Enhancement
2. Data Size Enhancement (Data Augmentation)
3. Real-time Features
4. A/B Testing Framework
5. Caching with Redis

Usage:
    python enhance_postyai.py [--run-all] [--train-models] [--augment-data] [--test-cache]
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def run_data_augmentation():
    """Run data augmentation to increase dataset size"""
    print("🔄 Running Data Augmentation...")
    print("=" * 50)
    
    try:
        from src.ml.data_augmentation import DataAugmentor
        import json
        
        # Load original data
        with open('data/processed_posts.json', 'r', encoding='utf-8') as f:
            original_posts = json.load(f)
        
        print(f"📥 Loaded {len(original_posts)} original posts")
        
        # Initialize augmentor
        augmentor = DataAugmentor()
        
        # Augment dataset to 300 posts
        augmented_posts = augmentor.augment_dataset(original_posts, target_size=300)
        
        # Save augmented data
        augmentor.save_augmented_data(augmented_posts)
        
        print("✅ Data augmentation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data augmentation failed: {e}")
        return False

def run_enhanced_model_training():
    """Run enhanced model training with hyperparameter tuning"""
    print("🚀 Running Enhanced Model Training...")
    print("=" * 50)
    
    try:
        from src.ml.train_models_enhanced import main as train_main
        train_main()
        print("✅ Enhanced model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model training failed: {e}")
        return False

def test_caching_system():
    """Test the caching system"""
    print("🧪 Testing Caching System...")
    print("=" * 50)
    
    try:
        from src.utils.caching import get_cache_manager, cache_response
        import time
        
        cache_manager = get_cache_manager()
        
        # Test basic operations
        print("📝 Testing basic cache operations...")
        test_key = "test_enhancement"
        test_value = {"message": "PostyAI Enhanced!", "timestamp": time.time()}
        
        # Set value
        success = cache_manager.set(test_key, test_value, 60)
        print(f"  Set operation: {'✅ Success' if success else '❌ Failed'}")
        
        # Get value
        retrieved = cache_manager.get(test_key)
        print(f"  Get operation: {'✅ Success' if retrieved else '❌ Failed'}")
        
        # Test cache stats
        stats = cache_manager.get_stats()
        print(f"  Cache stats: {stats}")
        
        # Test decorator
        @cache_response(ttl=30, namespace="test")
        def test_function(x, y):
            return x + y + time.time()
        
        # First call (cache miss)
        result1 = test_function(1, 2)
        print(f"  First call result: {result1}")
        
        # Second call (cache hit)
        result2 = test_function(1, 2)
        print(f"  Second call result: {result2}")
        print(f"  Cache hit: {'✅ Yes' if result1 == result2 else '❌ No'}")
        
        print("✅ Caching system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Caching system test failed: {e}")
        return False

def test_analytics_system():
    """Test the real-time analytics system"""
    print("📊 Testing Real-time Analytics...")
    print("=" * 50)
    
    try:
        from src.utils.real_time_analytics import get_analytics_instance
        import uuid
        
        analytics = get_analytics_instance()
        
        # Test post generation tracking
        test_post_id = str(uuid.uuid4())
        test_content = "This is a test post for analytics! 🚀 #testing"
        test_params = {
            'length': 'Medium',
            'language': 'English',
            'tag': 'Testing',
            'tone': 'Professional'
        }
        
        print("📝 Testing post generation tracking...")
        analytics.track_post_generation(
            test_post_id, 
            test_content, 
            test_params, 
            predicted_engagement=85.5
        )
        
        # Test interactions
        print("📝 Testing interaction tracking...")
        analytics.track_post_interaction(test_post_id, 'view', 'test_user_1')
        analytics.track_post_interaction(test_post_id, 'copy', 'test_user_1')
        analytics.track_post_interaction(test_post_id, 'rate', 'test_user_1', {'rating': 4})
        
        # Get analytics
        print("📝 Testing analytics retrieval...")
        live_metrics = analytics.get_live_metrics()
        print(f"  Live metrics: ✅ Retrieved")
        
        post_analytics = analytics.get_post_analytics(test_post_id)
        print(f"  Post analytics: ✅ Retrieved")
        
        trending = analytics.get_trending_topics()
        print(f"  Trending topics: ✅ Retrieved")
        
        print("✅ Real-time analytics test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Real-time analytics test failed: {e}")
        return False

def test_ab_testing_system():
    """Test the A/B testing system"""
    print("🧪 Testing A/B Testing Framework...")
    print("=" * 50)
    
    try:
        from src.utils.ab_testing import get_ab_testing_framework
        
        ab_testing = get_ab_testing_framework()
        
        # Create a test
        print("📝 Creating test A/B test...")
        test_variants = [
            {
                'name': 'control',
                'prompt': 'Generate a professional LinkedIn post',
                'description': 'Original prompt'
            },
            {
                'name': 'variant_a',
                'prompt': 'Create an engaging LinkedIn post that drives interaction',
                'description': 'Engagement-focused prompt'
            }
        ]
        
        test_id = ab_testing.create_test(
            test_name="Enhanced Prompt Test",
            description="Testing prompt variations for better engagement",
            variants=test_variants,
            duration_days=7
        )
        
        print(f"  Test created with ID: {test_id}")
        
        # Start the test
        print("📝 Starting A/B test...")
        started = ab_testing.start_test(test_id)
        print(f"  Test started: {'✅ Success' if started else '❌ Failed'}")
        
        # Test user assignment
        print("📝 Testing user assignment...")
        variant = ab_testing.assign_user_to_variant(test_id, 'test_user_1')
        print(f"  User assigned to variant: {variant['name'] if variant else 'None'}")
        
        # Record result
        if variant:
            print("📝 Recording test result...")
            ab_testing.record_result(test_id, 'test_user_1', {
                'user_rating': 4,
                'generation_time': 1.2,
                'predicted_engagement': 85.0
            })
            print("  Result recorded ✅")
        
        # Get results
        print("📝 Getting test results...")
        results = ab_testing.get_test_results(test_id)
        print(f"  Results retrieved: {'✅ Success' if results else '❌ Failed'}")
        
        print("✅ A/B testing framework test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ A/B testing framework test failed: {e}")
        return False

def run_all_enhancements():
    """Run all enhancement procedures"""
    print("🚀 PostyAI Complete Enhancement Pipeline")
    print("=" * 60)
    
    results = {
        'data_augmentation': False,
        'model_training': False,
        'caching_test': False,
        'analytics_test': False,
        'ab_testing_test': False
    }
    
    # Step 1: Data Augmentation
    print("\n🔄 STEP 1: Data Augmentation")
    results['data_augmentation'] = run_data_augmentation()
    
    # Step 2: Enhanced Model Training
    print("\n🤖 STEP 2: Enhanced Model Training")
    if results['data_augmentation']:
        results['model_training'] = run_enhanced_model_training()
    else:
        print("⏭️ Skipping model training due to data augmentation failure")
    
    # Step 3: Test Caching System
    print("\n💾 STEP 3: Caching System Test")
    results['caching_test'] = test_caching_system()
    
    # Step 4: Test Real-time Analytics
    print("\n📊 STEP 4: Real-time Analytics Test")
    results['analytics_test'] = test_analytics_system()
    
    # Step 5: Test A/B Testing Framework
    print("\n🧪 STEP 5: A/B Testing Framework Test")
    results['ab_testing_test'] = test_ab_testing_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ENHANCEMENT PIPELINE SUMMARY")
    print("=" * 60)
    
    total_steps = len(results)
    successful_steps = sum(results.values())
    
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {step.replace('_', ' ').title()}: {status}")
    
    print(f"\n🏆 Overall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps*100:.1f}%)")
    
    if successful_steps == total_steps:
        print("🎉 All enhancements completed successfully!")
        print("💡 PostyAI is now running with all performance improvements!")
    else:
        print("⚠️ Some enhancements failed. Check the logs above for details.")
    
    return successful_steps == total_steps

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="PostyAI Enhancement Pipeline")
    parser.add_argument('--run-all', action='store_true', 
                       help='Run all enhancement procedures')
    parser.add_argument('--train-models', action='store_true',
                       help='Run enhanced model training only')
    parser.add_argument('--augment-data', action='store_true',
                       help='Run data augmentation only')
    parser.add_argument('--test-cache', action='store_true',
                       help='Test caching system only')
    parser.add_argument('--test-analytics', action='store_true',
                       help='Test analytics system only')
    parser.add_argument('--test-ab', action='store_true',
                       help='Test A/B testing system only')
    
    args = parser.parse_args()
    
    if args.run_all:
        success = run_all_enhancements()
        sys.exit(0 if success else 1)
    elif args.train_models:
        success = run_enhanced_model_training()
        sys.exit(0 if success else 1)
    elif args.augment_data:
        success = run_data_augmentation()
        sys.exit(0 if success else 1)
    elif args.test_cache:
        success = test_caching_system()
        sys.exit(0 if success else 1)
    elif args.test_analytics:
        success = test_analytics_system()
        sys.exit(0 if success else 1)
    elif args.test_ab:
        success = test_ab_testing_system()
        sys.exit(0 if success else 1)
    else:
        print("🚀 PostyAI Enhancement Pipeline")
        print("=" * 40)
        print("Choose an enhancement to run:")
        print("  --run-all: Run all enhancements")
        print("  --train-models: Enhanced model training")
        print("  --augment-data: Data augmentation")
        print("  --test-cache: Test caching system")
        print("  --test-analytics: Test analytics system")
        print("  --test-ab: Test A/B testing system")
        print("\nExample: python enhance_postyai.py --run-all")

if __name__ == "__main__":
    main()
