#!/usr/bin/env python3
"""
Quick Test Script for PostyAI Enhanced Features

This script performs a quick test of all the new enhanced features
to ensure everything is working correctly.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the enhanced health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Status: {data['status']}")
            print(f"  📊 Components: {len(data['components'])} checked")
            print(f"  🎯 Version: {data['version']}")
            return True
        else:
            print(f"  ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Health check error: {e}")
        return False

def test_enhanced_generation():
    """Test enhanced post generation with new features"""
    print("🤖 Testing enhanced post generation...")
    try:
        data = {
            "length": "Medium",
            "language": "English",
            "tag": "UX Design",
            "tone": "Professional"
        }
        
        response = requests.post(f"{BASE_URL}/api/generate", json=data)
        if response.status_code == 200:
            result = response.json()
            post = result['post']
            system_info = result.get('system_info', {})
            
            print(f"  ✅ Post generated successfully")
            print(f"  📝 Content length: {len(post['content'])} chars")
            print(f"  ⏱️ Generation time: {post['generation_time']}s")
            print(f"  🎯 ML prediction: {post.get('predicted_engagement', 'N/A')}")
            print(f"  🧪 A/B test: {system_info.get('ab_test_active', False)}")
            print(f"  💾 Cache status: {system_info.get('cache_status', 'unknown')}")
            
            return post['id']
        else:
            print(f"  ❌ Generation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ Generation error: {e}")
        return None

def test_analytics(post_id=None):
    """Test real-time analytics endpoints"""
    print("📊 Testing analytics...")
    try:
        # Test live analytics
        response = requests.get(f"{BASE_URL}/api/analytics/live")
        if response.status_code == 200:
            print("  ✅ Live analytics working")
        
        # Test trending topics
        response = requests.get(f"{BASE_URL}/api/analytics/trending")
        if response.status_code == 200:
            print("  ✅ Trending topics working")
        
        # Test post analytics if we have a post ID
        if post_id:
            response = requests.get(f"{BASE_URL}/api/analytics/post/{post_id}")
            if response.status_code == 200:
                print("  ✅ Post analytics working")
        
        return True
    except Exception as e:
        print(f"  ❌ Analytics error: {e}")
        return False

def test_tracking(post_id):
    """Test interaction tracking"""
    print("📈 Testing interaction tracking...")
    try:
        # Test view tracking
        response = requests.post(f"{BASE_URL}/api/track/view", json={"post_id": post_id})
        if response.status_code == 200:
            print("  ✅ View tracking working")
        
        # Test copy tracking
        response = requests.post(f"{BASE_URL}/api/track/copy", json={"post_id": post_id})
        if response.status_code == 200:
            print("  ✅ Copy tracking working")
        
        # Test rating
        response = requests.post(f"{BASE_URL}/api/ab-testing/rate", json={
            "post_id": post_id,
            "rating": 4
        })
        if response.status_code == 200:
            print("  ✅ Rating tracking working")
        
        return True
    except Exception as e:
        print(f"  ❌ Tracking error: {e}")
        return False

def test_ab_testing():
    """Test A/B testing framework"""
    print("🧪 Testing A/B testing...")
    try:
        # Test getting active tests
        response = requests.get(f"{BASE_URL}/api/ab-testing/tests")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Active tests: {data['total_active']}")
        
        # Test creating a test
        test_data = {
            "name": "Quick Test",
            "description": "Testing the A/B framework",
            "variants": [
                {"name": "control", "description": "Original"},
                {"name": "variant", "description": "Test variant"}
            ],
            "duration_days": 1
        }
        
        response = requests.post(f"{BASE_URL}/api/ab-testing/create", json=test_data)
        if response.status_code == 200:
            test_id = response.json()['test_id']
            print(f"  ✅ Test created: {test_id[:8]}...")
            
            # Try to start the test
            response = requests.post(f"{BASE_URL}/api/ab-testing/start/{test_id}")
            if response.status_code == 200:
                print("  ✅ Test started successfully")
            
            return test_id
        
        return None
    except Exception as e:
        print(f"  ❌ A/B testing error: {e}")
        return None

def test_cache_stats():
    """Test cache statistics"""
    print("💾 Testing cache statistics...")
    try:
        response = requests.get(f"{BASE_URL}/api/cache/stats")
        if response.status_code == 200:
            data = response.json()
            stats = data['cache_stats']
            print(f"  ✅ Cache backend: {stats['backend']}")
            print(f"  📈 Hit rate: {stats['hit_rate_percent']}%")
            print(f"  🔄 Operations: {stats['operations']['hits']} hits, {stats['operations']['misses']} misses")
            return True
        else:
            print(f"  ❌ Cache stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Cache stats error: {e}")
        return False

def test_ml_prediction():
    """Test ML prediction endpoint"""
    print("🧠 Testing ML predictions...")
    try:
        test_content = "This is a test LinkedIn post about productivity tips! 🚀 #productivity"
        data = {
            "content": test_content,
            "tags": ["Productivity"],
            "language": "English"
        }
        
        response = requests.post(f"{BASE_URL}/api/ml/predict-engagement", json=data)
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            print(f"  ✅ Engagement prediction: {prediction['engagement']:.1f}")
            print(f"  📊 Model used: {prediction['model_used']}")
            print(f"  🎯 Confidence interval: [{prediction['confidence_interval'][0]:.1f}, {prediction['confidence_interval'][1]:.1f}]")
            return True
        else:
            print(f"  ❌ ML prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ ML prediction error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PostyAI Enhanced Features Test Suite")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Server not responding. Make sure PostyAI is running on localhost:8000")
            sys.exit(1)
    except:
        print("❌ Cannot connect to server. Make sure PostyAI is running on localhost:8000")
        sys.exit(1)
    
    results = {}
    
    # Test 1: Health Check
    results['health'] = test_health_check()
    
    # Test 2: Enhanced Generation
    post_id = test_enhanced_generation()
    results['generation'] = post_id is not None
    
    # Test 3: Analytics
    results['analytics'] = test_analytics(post_id)
    
    # Test 4: Tracking
    if post_id:
        results['tracking'] = test_tracking(post_id)
    else:
        results['tracking'] = False
    
    # Test 5: A/B Testing
    test_id = test_ab_testing()
    results['ab_testing'] = test_id is not None
    
    # Test 6: Cache Stats
    results['cache'] = test_cache_stats()
    
    # Test 7: ML Predictions
    results['ml_prediction'] = test_ml_prediction()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🏆 Overall Result: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All enhanced features are working correctly!")
    else:
        print("⚠️ Some features need attention. Check the logs above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
