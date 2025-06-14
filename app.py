from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime, timedelta
from functools import wraps
import time

from few_shot import FewShotPosts
from post_generator import generate_post
from src.utils.post_analytics import get_post_stats

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}

def rate_limit(max_requests=10, window_minutes=1):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = request.remote_addr
            current_time = datetime.now()
            
            if client_id not in rate_limit_storage:
                rate_limit_storage[client_id] = []
            
            # Clean old requests
            rate_limit_storage[client_id] = [
                req_time for req_time in rate_limit_storage[client_id]
                if current_time - req_time < timedelta(minutes=window_minutes)
            ]
            
            if len(rate_limit_storage[client_id]) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded. Please try again later.',
                    'retry_after': window_minutes * 60
                }), 429
            
            rate_limit_storage[client_id].append(current_time)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test data loading
        fs = FewShotPosts()
        tag_count = len(fs.get_tags())
        
        # Test API key exists
        import os
        api_key_exists = bool(os.getenv("GROQ_API_KEY", "").strip())
        
        return jsonify({
            'status': 'healthy',
            'components': {
                'data_loading': 'ok',
                'tag_count': tag_count,
                'api_key_configured': api_key_exists
            },
            'version': '1.0.0'
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
@rate_limit(max_requests=5, window_minutes=1)
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
@rate_limit(max_requests=3, window_minutes=1)
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
def export_post(post_id):
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

@app.route('/api/batch-generate', methods=['POST'])
@rate_limit(max_requests=2, window_minutes=5)
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

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000) 