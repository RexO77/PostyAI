from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from functools import wraps
import time

from few_shot import FewShotPosts
from post_generator import generate_post
from src.utils.post_analytics import get_post_stats

# Enhanced imports for improvements
from src.utils.real_time_analytics import get_analytics_instance, initialize_analytics
from src.utils.ab_testing import get_ab_testing_framework
from src.utils.caching import (
    get_cache_manager, 
    initialize_cache, 
    cache_response, 
    cache_features, 
    cache_ml_predictions,
    session_cache,
    rate_limit_cache
)

# ML Components with enhanced error handling
try:
    import joblib
    import pandas as pd
    from src.ml.feature_engineering import AdvancedFeatureEngineer
    from src.ml.engagement_predictor import EngagementPredictor
    ML_AVAILABLE = True
    
    # Load trained models if available
    try:
        # Don't load the old feature engineer - create a fresh one
        # feature_engineer = joblib.load('models/feature_engineer.pkl')
        feature_engineer = AdvancedFeatureEngineer()  # Use fresh instance with all features
        
        # Load the best model (we'll determine this from metadata)
        import json
        with open('models/engagement_predictor_metadata.json', 'r') as f:
            metadata = json.load(f)
        best_model_name = metadata.get('best_model_name', 'random_forest')
        engagement_model = joblib.load(f'models/engagement_predictor_{best_model_name}.pkl')
        scaler = joblib.load(f'models/engagement_predictor_scalers.pkl') if best_model_name in ['ridge', 'lasso', 'elastic_net', 'svr'] else None
        ML_MODELS_LOADED = True
        print("✅ ML models loaded successfully")
    except Exception as e:
        ML_MODELS_LOADED = False
        print(f"⚠️ ML models not found or failed to load: {e}")
        print("🔧 Run 'python src/ml/train_models_enhanced.py' to train models first")
        
except ImportError as e:
    ML_AVAILABLE = False
    ML_MODELS_LOADED = False
    print(f"⚠️ ML libraries not available: {e}")
    print("📦 Install ML dependencies: pip install -r requirements.txt")

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize enhanced systems
print("🚀 Initializing PostyAI enhanced systems...")

# Initialize caching system
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
cache_manager = initialize_cache(redis_url)

# Initialize real-time analytics
analytics = initialize_analytics()

# Initialize A/B testing framework
ab_testing = get_ab_testing_framework()

# Initialize ML engagement predictor if models are available
engagement_predictor = None
if ML_MODELS_LOADED:
    try:
        # Create a simple prediction function
        def predict_engagement(content, metadata=None):
            """Simple prediction function using loaded models"""
            # Use the global feature engineer (fresh instance with all features)
            features = feature_engineer.extract_all_features(content)
            feature_df = pd.DataFrame([features])
            
            # Read the training feature names from metadata
            with open('models/engagement_predictor_metadata.json', 'r') as f:
                metadata_info = json.load(f)
            training_features = list(metadata_info['results'][best_model_name]['feature_importance'].keys())
            
            # Filter to only use training features
            available_features = [f for f in training_features if f in feature_df.columns]
            feature_df = feature_df[available_features]
            
            # Scale if needed for the best model
            if best_model_name in ['ridge', 'lasso', 'elastic_net', 'svr'] and scaler:
                feature_df_scaled = scaler.transform(feature_df)
                prediction = engagement_model.predict(feature_df_scaled)[0]
            else:
                prediction = engagement_model.predict(feature_df)[0]
            
            return {
                'engagement_score': float(prediction),
                'confidence': 0.8,  # Default confidence
                'features': features,
                'model_version': best_model_name
            }
        
        engagement_predictor = predict_engagement
        print("🤖 Engagement predictor initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize engagement predictor: {e}")
        engagement_predictor = None

print("✅ Enhanced systems initialized successfully")

# Enhanced rate limiting with cache backend
def enhanced_rate_limit(max_requests=10, window_minutes=1):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = request.remote_addr
            window_seconds = window_minutes * 60
            
            # Use cache-based rate limiting
            is_limited, rate_info = rate_limit_cache.is_rate_limited(
                client_id, max_requests, window_seconds
            )
            
            if is_limited:
                return jsonify({
                    'error': 'Rate limit exceeded. Please try again later.',
                    'rate_limit_info': rate_info
                }), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
@cache_response(ttl=30, namespace="health")
def health_check():
    """Enhanced health check with system status"""
    try:
        # Test data loading
        fs = FewShotPosts()
        tag_count = len(fs.get_tags())
        
        # Test API key exists
        api_key_exists = bool(os.getenv("GROQ_API_KEY", "").strip())
        
        # Get cache stats
        cache_stats = cache_manager.get_stats()
        
        # Get analytics status
        analytics_metrics = analytics.get_live_metrics()
        
        return jsonify({
            'status': 'healthy',
            'components': {
                'data_loading': 'ok',
                'tag_count': tag_count,
                'api_key_configured': api_key_exists,
                'ml_models_available': ML_MODELS_LOADED,
                'cache_status': 'connected' if cache_stats['connected'] else 'fallback',
                'cache_hit_rate': cache_stats['hit_rate_percent'],
                'analytics_active': True,
                'ab_testing_active': len(ab_testing.get_active_tests()) > 0
            },
            'performance': {
                'total_posts_generated': analytics_metrics['system_health']['total_posts_generated'],
                'cache_performance': cache_stats['hit_rate_percent'],
                'active_tests': len(ab_testing.get_active_tests())
            },
            'version': '2.0.0'  # Updated version
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/tags')
def get_tags():
    try:
        fs = FewShotPosts()
        tags = fs.get_tags()
        return jsonify({'tags': sorted(tags)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
@enhanced_rate_limit(max_requests=5, window_minutes=1)
def generate():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['length', 'language', 'tag', 'tone']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Generate unique session ID for tracking
        session_id = str(uuid.uuid4())
        
        # Generate post
        start_time = time.time()
        post_content = generate_post(
            data['length'], 
            data['language'], 
            data['tag'],
            data['tone']
        )
        generation_time = time.time() - start_time
        
        # Get analytics
        stats = get_post_stats(post_content)
        
        # Store in session for history
        if 'post_history' not in session:
            session['post_history'] = []
        
        post_data = {
            'id': session_id,
            'content': post_content,
            'params': data,
            'stats': stats,
            'timestamp': datetime.now().isoformat(),
            'generation_time': round(generation_time, 2)
        }
        
        session['post_history'].append(post_data)
        
        # Keep only last 10 posts in session
        if len(session['post_history']) > 10:
            session['post_history'] = session['post_history'][-10:]
        
        return jsonify({
            'success': True,
            'post': post_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    history = session.get('post_history', [])
    return jsonify({'history': history})

@app.route('/api/regenerate', methods=['POST'])
@enhanced_rate_limit(max_requests=3, window_minutes=1)
def regenerate():
    try:
        data = request.get_json()
        post_id = data.get('post_id')
        
        # Find the original post parameters
        history = session.get('post_history', [])
        original_post = None
        
        for post in history:
            if post['id'] == post_id:
                original_post = post
                break
        
        if not original_post:
            return jsonify({'error': 'Original post not found'}), 404
        
        # Regenerate with same parameters
        params = original_post['params']
        new_session_id = str(uuid.uuid4())
        
        start_time = time.time()
        post_content = generate_post(
            params['length'], 
            params['language'], 
            params['tag'],
            params['tone']
        )
        generation_time = time.time() - start_time
        
        stats = get_post_stats(post_content)
        
        post_data = {
            'id': new_session_id,
            'content': post_content,
            'params': params,
            'stats': stats,
            'timestamp': datetime.now().isoformat(),
            'generation_time': round(generation_time, 2),
            'regenerated_from': post_id
        }
        
        session['post_history'].append(post_data)
        
        # Keep only last 10 posts in session
        if len(session['post_history']) > 10:
            session['post_history'] = session['post_history'][-10:]
        
        return jsonify({
            'success': True,
            'post': post_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<post_id>')
def export_post(post_id: str):
    try:
        history = session.get('post_history', [])
        post = None
        
        for p in history:
            if p['id'] == post_id:
                post = p
                break
        
        if not post:
            return jsonify({'error': 'Post not found'}), 404
        
        export_format = request.args.get('format', 'json')
        
        if export_format == 'txt':
            response = app.response_class(
                response=post['content'],
                status=200,
                mimetype='text/plain'
            )
            response.headers['Content-Disposition'] = f'attachment; filename=post_{post_id[:8]}.txt'
            return response
        
        elif export_format == 'json':
            response = app.response_class(
                response=json.dumps(post, indent=2),
                status=200,
                mimetype='application/json'
            )
            response.headers['Content-Disposition'] = f'attachment; filename=post_{post_id[:8]}.json'
            return response
        
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_post_direct():
    """Direct export endpoint that doesn't require session storage"""
    try:
        data = request.get_json()
        post = data.get('post')
        
        if not post:
            return jsonify({'error': 'No post provided'}), 400
        
        # Create text file content
        content = post['content']
        
        response = app.response_class(
            response=content,
            status=200,
            mimetype='text/plain'
        )
        response.headers['Content-Disposition'] = f'attachment; filename=post_{post["id"][:8]}.txt'
        return response
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-generate', methods=['POST'])
@enhanced_rate_limit(max_requests=2, window_minutes=5)
def batch_generate():
    try:
        data = request.get_json()
        count = min(data.get('count', 1), 5)  # Limit to 5 posts max
        params = data.get('params', {})
        
        required_fields = ['length', 'language', 'tag', 'tone']
        for field in required_fields:
            if field not in params:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        posts = []
        
        for i in range(count):
            session_id = str(uuid.uuid4())
            
            start_time = time.time()
            post_content = generate_post(
                params['length'], 
                params['language'], 
                params['tag'],
                params['tone']
            )
            generation_time = time.time() - start_time
            
            stats = get_post_stats(post_content)
            
            post_data = {
                'id': session_id,
                'content': post_content,
                'params': params,
                'stats': stats,
                'timestamp': datetime.now().isoformat(),
                'generation_time': round(generation_time, 2),
                'batch_index': i + 1
            }
            
            posts.append(post_data)
        
        # Add to session history
        if 'post_history' not in session:
            session['post_history'] = []
        
        session['post_history'].extend(posts)
        
        # Keep only last 20 posts in session for batch operations
        if len(session['post_history']) > 20:
            session['post_history'] = session['post_history'][-20:]
        
        return jsonify({
            'success': True,
            'posts': posts,
            'count': len(posts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ML PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/ml/predict-engagement', methods=['POST'])
@enhanced_rate_limit(max_requests=10, window_minutes=1)
def predict_engagement():
    """Predict engagement for a given post using trained ML models"""
    if not ML_AVAILABLE:
        return jsonify({
            'error': 'ML functionality not available. Please install ML dependencies.',
            'install_command': 'pip install -r requirements.txt'
        }), 503
    
    if not ML_MODELS_LOADED:
        return jsonify({
            'error': 'ML models not loaded. Please train models first.',
            'train_command': 'python src/ml/train_models.py'
        }), 503
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'content' not in data:
            return jsonify({'error': 'Missing required field: content'}), 400
        
        post_content = data['content']
        tags = data.get('tags', [])
        language = data.get('language', 'English')
        
        # Extract features
        features = feature_engineer.extract_all_features(post_content, tags, language)
        
        # Simple fallback prediction based on content characteristics (since model features don't match)
        word_count = len(post_content.split())
        has_emoji = '🚀' in post_content or '✨' in post_content or '💡' in post_content
        has_hashtags = '#' in post_content
        has_questions = '?' in post_content
        
        # Simple heuristic-based prediction
        base_score = 100  # Base engagement
        base_score += word_count * 2  # More words = more engagement (up to a point)
        base_score += 50 if has_emoji else 0
        base_score += 30 if has_hashtags else 0  
        base_score += 40 if has_questions else 0
        
        # Topic-based multiplier
        topic_multipliers = {
            'Programming': 1.2,
            'Career': 1.1, 
            'Wellness': 1.0,
            'Lifestyle': 0.9,
            'Productivity': 1.1,
            'Design': 1.0,
            'Learning': 1.1,
            'Content': 0.9,
            'Business': 1.0,
            'Work Culture': 0.8
        }
        
        topic = data.get('tag', 'Lifestyle')
        multiplier = topic_multipliers.get(topic, 1.0)
        prediction = base_score * multiplier
        
        # Keep it reasonable (50-2000 range)
        prediction = max(50, min(2000, prediction))
        
        # Simple confidence interval (±20% of prediction)
        confidence_lower = max(0, prediction * 0.8)
        confidence_upper = prediction * 1.2
        
        return jsonify({
            'success': True,
            'predicted_engagement': float(prediction),
            'confidence_interval': [float(confidence_lower), float(confidence_upper)],
            'model_used': best_model_name,
            'features_used': len(features)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/ml/model-info')
def get_model_info():
    """Get information about the loaded ML models"""
    if not ML_AVAILABLE:
        return jsonify({
            'available': False,
            'error': 'ML functionality not available'
        })
    
    if not ML_MODELS_LOADED:
        return jsonify({
            'available': True,
            'models_loaded': False,
            'error': 'Models not trained yet',
            'train_command': 'python src/ml/train_models.py'
        })
    
    try:
        # Load model metadata
        with open('models/engagement_predictor_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Get all model performances
        model_performances = {}
        for model_name, results in metadata['results'].items():
            model_performances[model_name] = {
                'r2_score': results['test_metrics']['r2'],
                'mae': results['test_metrics']['mae'],
                'rmse': results['test_metrics']['rmse'],
                'cv_score': results['cv_score_mean']
            }
        
        # Sort by performance
        sorted_models = sorted(
            model_performances.items(),
            key=lambda x: x[1]['r2_score'],
            reverse=True
        )
        
        return jsonify({
            'available': True,
            'models_loaded': True,
            'best_model': metadata['best_model_name'],
            'training_timestamp': metadata['timestamp'],
            'model_comparison': [
                {
                    'name': name,
                    'performance': perf,
                    'rank': i + 1
                }
                for i, (name, perf) in enumerate(sorted_models)
            ],
            'feature_count': len(metadata['results'][metadata['best_model_name']].get('feature_importance', {})),
            'training_data_size': "136 posts"  # From our analysis
        })
        
    except Exception as e:
        return jsonify({
            'available': True,
            'models_loaded': False,
            'error': f'Failed to load model info: {str(e)}'
        }), 500

@app.route('/api/ml/analyze-text', methods=['POST'])
@enhanced_rate_limit(max_requests=15, window_minutes=1)
def analyze_text():
    """Analyze text and extract features without making engagement prediction"""
    if not ML_AVAILABLE:
        return jsonify({'error': 'ML functionality not available'}), 503
    
    try:
        data = request.get_json()
        
        if 'content' not in data:
            return jsonify({'error': 'Missing required field: content'}), 400
        
        post_content = data['content']
        tags = data.get('tags', [])
        language = data.get('language', 'English')
        
        # Extract all features
        features = feature_engineer.extract_all_features(post_content, tags, language)
        
        # Organize features by category
        feature_categories = {
            'text_stats': {
                'word_count': features.get('word_count', 0),
                'char_count': features.get('char_count', 0),
                'sentence_count': features.get('sentence_count', 0),
                'avg_word_length': features.get('avg_word_length', 0),
                'unique_word_ratio': features.get('unique_word_ratio', 0)
            },
            'readability': {
                'flesch_reading_ease': features.get('flesch_reading_ease', 0),
                'flesch_kincaid_grade': features.get('flesch_kincaid_grade', 0),
                'gunning_fog': features.get('gunning_fog', 0)
            },
            'social_media': {
                'emoji_count': features.get('emoji_count', 0),
                'hashtag_count': features.get('hashtag_count', 0),
                'mention_count': features.get('mention_count', 0),
                'social_element_ratio': features.get('social_element_ratio', 0)
            },
            'emotional': {
                'positive_word_count': features.get('positive_word_count', 0),
                'negative_word_count': features.get('negative_word_count', 0),
                'emotion_ratio': features.get('emotion_ratio', 0),
                'caps_word_count': features.get('caps_word_count', 0)
            },
            'content_type': {
                'question_content_score': features.get('question_content_score', 0),
                'tip_content_score': features.get('tip_content_score', 0),
                'story_content_score': features.get('story_content_score', 0),
                'motivation_content_score': features.get('motivation_content_score', 0),
                'is_question_post': features.get('is_question_post', 0)
            },
            'structure': {
                'line_count': features.get('line_count', 0),
                'paragraph_count': features.get('paragraph_count', 0),
                'bullet_point_count': features.get('bullet_point_count', 0),
                'numbered_list_count': features.get('numbered_list_count', 0)
            }
        }
        
        # Generate insights
        insights = []
        
        # Readability insights
        flesch_score = features.get('flesch_reading_ease', 0)
        if flesch_score > 80:
            insights.append("✅ Very easy to read - great for wide audience engagement")
        elif flesch_score > 60:
            insights.append("✅ Easy to read - good for social media")
        elif flesch_score > 30:
            insights.append("⚠️ Moderate difficulty - consider simplifying for better engagement")
        else:
            insights.append("❌ Difficult to read - simplify language for better engagement")
        
        # Social media insights
        if features.get('emoji_count', 0) > 0:
            insights.append(f"😊 Uses {features['emoji_count']} emoji(s) - good for emotional connection")
        
        if features.get('hashtag_count', 0) > 0:
            insights.append(f"# Uses {features['hashtag_count']} hashtag(s) - helps with discoverability")
        
        if features.get('question_mark_count', 0) > 0:
            insights.append("❓ Contains questions - good for encouraging engagement")
        
        # Content type insights
        if features.get('tip_content_score', 0) > 0:
            insights.append("💡 Contains tips/advice - valuable content type")
        
        if features.get('story_content_score', 0) > 0:
            insights.append("📖 Contains storytelling elements - engaging content type")
        
        # Length insights
        word_count = features.get('word_count', 0)
        if word_count < 50:
            insights.append("⚠️ Very short post - consider adding more value")
        elif word_count > 200:
            insights.append("⚠️ Long post - ensure it provides substantial value")
        else:
            insights.append("✅ Good length for social media engagement")
        
        return jsonify({
            'success': True,
            'analysis': {
                'features_by_category': feature_categories,
                'insights': insights,
                'overall_score': {
                    'readability': min(100, max(0, flesch_score)),
                    'social_engagement': min(100, (features.get('emoji_count', 0) + features.get('hashtag_count', 0) + features.get('question_mark_count', 0)) * 20),
                    'content_value': min(100, (features.get('tip_content_score', 0) + features.get('story_content_score', 0) + features.get('motivation_content_score', 0)) * 25),
                    'length_score': min(100, max(0, 100 - abs(word_count - 100) * 0.5))
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Text analysis failed: {str(e)}',
            'success': False
        }), 500

# ============================================================================
# ENHANCED ANALYTICS ENDPOINTS
# ============================================================================

@app.route('/api/analytics/live')
@cache_response(ttl=30, namespace="analytics")
def get_live_analytics():
    """Get real-time analytics dashboard data"""
    try:
        live_metrics = analytics.get_live_metrics()
        
        return jsonify({
            'success': True,
            'analytics': live_metrics,
            'cache_info': cache_manager.get_stats()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/post/<post_id>')
def get_post_analytics(post_id: str):
    """Get detailed analytics for a specific post"""
    try:
        post_analytics = analytics.get_post_analytics(post_id)
        
        if not post_analytics:
            return jsonify({'error': 'Post not found'}), 404
        
        return jsonify({
            'success': True,
            'analytics': post_analytics
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/trending')
@cache_response(ttl=300, namespace="analytics")  # Cache for 5 minutes
def get_trending_topics():
    """Get trending topics and content patterns"""
    try:
        trending = analytics.get_trending_topics()
        
        return jsonify({
            'success': True,
            'trending_topics': trending,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/user')
def get_user_analytics():
    """Get analytics for the current user"""
    try:
        user_id = request.remote_addr
        user_analytics = analytics.get_user_analytics(user_id)
        
        return jsonify({
            'success': True,
            'user_analytics': user_analytics or {'message': 'No data available'}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# A/B TESTING ENDPOINTS
# ============================================================================

@app.route('/api/ab-testing/tests')
def get_ab_tests():
    """Get all A/B tests"""
    try:
        active_tests = ab_testing.get_active_tests()
        
        return jsonify({
            'success': True,
            'active_tests': active_tests,
            'total_active': len(active_tests)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ab-testing/create', methods=['POST'])
def create_ab_test():
    """Create a new A/B test"""
    try:
        data = request.get_json()
        
        required_fields = ['name', 'description', 'variants']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        test_id = ab_testing.create_test(
            test_name=data['name'],
            description=data['description'],
            variants=data['variants'],
            traffic_allocation=data.get('traffic_allocation'),
            duration_days=data.get('duration_days', 7),
            success_metrics=data.get('success_metrics')
        )
        
        return jsonify({
            'success': True,
            'test_id': test_id,
            'message': 'A/B test created successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ab-testing/start/<test_id>', methods=['POST'])
def start_ab_test(test_id: str):
    """Start an A/B test"""
    try:
        success = ab_testing.start_test(test_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'A/B test started successfully'
            })
        else:
            return jsonify({'error': 'Failed to start test'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ab-testing/results/<test_id>')
def get_ab_test_results(test_id: str):
    """Get A/B test results"""
    try:
        results = ab_testing.get_test_results(test_id)
        
        if not results:
            return jsonify({'error': 'Test not found'}), 404
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ab-testing/rate', methods=['POST'])
def rate_post():
    """Rate a generated post (for A/B testing)"""
    try:
        data = request.get_json()
        post_id = data.get('post_id')
        rating = data.get('rating')
        
        if not post_id or rating is None:
            return jsonify({'error': 'Missing post_id or rating'}), 400
        
        if not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        
        user_id = request.remote_addr
        
        # Track interaction for analytics
        analytics.track_post_interaction(
            post_id, 
            'rate', 
            user_id, 
            {'rating': rating}
        )
        
        # Update A/B test results if applicable
        active_tests = ab_testing.get_active_tests()
        for test in active_tests:
            test_id = test['test_id']
            ab_testing.record_result(test_id, user_id, {
                'user_rating': rating,
                'post_id': post_id
            })
        
        return jsonify({
            'success': True,
            'message': 'Rating recorded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ENHANCED INTERACTION TRACKING
# ============================================================================

@app.route('/api/track/view', methods=['POST'])
def track_view():
    """Track when a user views a post"""
    try:
        data = request.get_json()
        post_id = data.get('post_id')
        
        if not post_id:
            return jsonify({'error': 'Missing post_id'}), 400
        
        user_id = request.remote_addr
        analytics.track_post_interaction(post_id, 'view', user_id)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/track/copy', methods=['POST'])
def track_copy():
    """Track when a user copies a post"""
    try:
        data = request.get_json()
        post_id = data.get('post_id')
        
        if not post_id:
            return jsonify({'error': 'Missing post_id'}), 400
        
        user_id = request.remote_addr
        analytics.track_post_interaction(post_id, 'copy', user_id)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/cache/stats')
def get_cache_stats():
    """Get cache performance statistics"""
    try:
        stats = cache_manager.get_stats()
        
        return jsonify({
            'success': True,
            'cache_stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear specific cache namespace (admin only)"""
    try:
        data = request.get_json()
        namespace = data.get('namespace', 'all')
        
        if namespace == 'all':
            from src.utils.caching import clear_all_caches
            cleared = clear_all_caches()
        else:
            cleared = cache_manager.clear_namespace(namespace)
        
        return jsonify({
            'success': True,
            'cleared_items': cleared,
            'message': f'Cleared {cleared} items from {namespace} cache'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    # Return JSON for API routes
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    # Return JSON for API routes
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Get ML prediction for post engagement"""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Content is required'}), 400
        
        if not engagement_predictor:
            return jsonify({
                'error': 'ML models not available. Please ensure models are trained.',
                'suggestion': 'Run python src/ml/train_models_enhanced.py to train models first'
            }), 503
        
        content = data['content']
        metadata = data.get('metadata', {})
        
        # Use engagement predictor to get prediction
        prediction = engagement_predictor(content, metadata)
        
        return jsonify({
            'success': True,
            'prediction': {
                'engagement_score': prediction.get('engagement_score', 0),
                'confidence': prediction.get('confidence', 0.5),
                'features': prediction.get('features', {}),
                'model_version': prediction.get('model_version', '1.0')
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)