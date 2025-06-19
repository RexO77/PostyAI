"""
Advanced Caching System for PostyAI

This module provides comprehensive caching solutions using Redis for improved
performance, including response caching, feature caching, and session management.
"""

import json
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
import redis
from functools import wraps
import logging


class CacheManager:
    """
    Advanced caching manager with Redis backend
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            self.connected = True
            print("✅ Connected to Redis cache")
        except (redis.ConnectionError, redis.RedisError) as e:
            print(f"⚠️ Redis not available: {e}")
            print("📝 Falling back to in-memory cache")
            self.redis_client = None
            self.connected = False
            self._memory_cache = {}
            self._cache_expiry = {}
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.cache_prefix = "postyai:"
        
        # Cache hit/miss statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def _get_cache_key(self, key: str, namespace: str = "default") -> str:
        """Generate cache key with namespace"""
        return f"{self.cache_prefix}{namespace}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(data)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "default") -> bool:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Cache namespace
            
        Returns:
            Success status
        """
        cache_key = self._get_cache_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        try:
            if self.connected:
                serialized_value = self._serialize_value(value)
                result = self.redis_client.setex(cache_key, ttl, serialized_value)
                self.stats['sets'] += 1
                return result
            else:
                # Fallback to memory cache
                self._memory_cache[cache_key] = value
                self._cache_expiry[cache_key] = datetime.now() + timedelta(seconds=ttl)
                self.stats['sets'] += 1
                return True
                
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            Cached value or None
        """
        cache_key = self._get_cache_key(key, namespace)
        
        try:
            if self.connected:
                data = self.redis_client.get(cache_key)
                if data:
                    self.stats['hits'] += 1
                    return self._deserialize_value(data)
                else:
                    self.stats['misses'] += 1
                    return None
            else:
                # Check memory cache
                if cache_key in self._memory_cache:
                    # Check expiry
                    if cache_key in self._cache_expiry and datetime.now() > self._cache_expiry[cache_key]:
                        del self._memory_cache[cache_key]
                        del self._cache_expiry[cache_key]
                        self.stats['misses'] += 1
                        return None
                    else:
                        self.stats['hits'] += 1
                        return self._memory_cache[cache_key]
                else:
                    self.stats['misses'] += 1
                    return None
                    
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            self.stats['misses'] += 1
            return None
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a value from cache"""
        cache_key = self._get_cache_key(key, namespace)
        
        try:
            if self.connected:
                result = self.redis_client.delete(cache_key)
                self.stats['deletes'] += 1
                return bool(result)
            else:
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                    if cache_key in self._cache_expiry:
                        del self._cache_expiry[cache_key]
                    self.stats['deletes'] += 1
                    return True
                return False
                
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache"""
        cache_key = self._get_cache_key(key, namespace)
        
        try:
            if self.connected:
                return bool(self.redis_client.exists(cache_key))
            else:
                return cache_key in self._memory_cache
        except Exception as e:
            logging.error(f"Cache exists error: {e}")
            return False
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        try:
            if self.connected:
                pattern = f"{self.cache_prefix}{namespace}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    self.stats['deletes'] += deleted
                    return deleted
                return 0
            else:
                pattern_prefix = f"{self.cache_prefix}{namespace}:"
                keys_to_delete = [k for k in self._memory_cache.keys() if k.startswith(pattern_prefix)]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    if key in self._cache_expiry:
                        del self._cache_expiry[key]
                self.stats['deletes'] += len(keys_to_delete)
                return len(keys_to_delete)
                
        except Exception as e:
            logging.error(f"Cache clear namespace error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_operations = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_operations * 100) if total_operations > 0 else 0
        
        stats = {
            'connected': self.connected,
            'backend': 'Redis' if self.connected else 'Memory',
            'hit_rate_percent': round(hit_rate, 2),
            'operations': self.stats,
            'memory_usage': len(self._memory_cache) if not self.connected else None
        }
        
        if self.connected:
            try:
                info = self.redis_client.info('memory')
                stats['redis_memory_usage'] = info.get('used_memory_human', 'N/A')
                stats['redis_connected_clients'] = self.redis_client.info().get('connected_clients', 0)
            except:
                pass
        
        return stats


# Cache decorators for common use cases
def cache_response(ttl: int = 3600, namespace: str = "responses", 
                  key_func: Optional[callable] = None):
    """
    Decorator to cache function responses
    
    Args:
        ttl: Time to live in seconds
        namespace: Cache namespace
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        key_parts.append(str(arg))
                for k, v in sorted(kwargs.items()):
                    if isinstance(v, (str, int, float, bool)):
                        key_parts.append(f"{k}:{v}")
                
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, namespace)
            
            return result
        return wrapper
    return decorator


def cache_features(ttl: int = 7200):
    """Decorator specifically for caching feature extraction results"""
    def key_func(text: str, tags: List[str] = None, language: str = 'English'):
        # Create deterministic key from inputs
        key_data = f"{text}:{sorted(tags or [])}:{language}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    return cache_response(ttl=ttl, namespace="features", key_func=key_func)


def cache_ml_predictions(ttl: int = 1800):
    """Decorator for caching ML model predictions"""
    def key_func(post_content: str, tags: List[str] = None, language: str = 'English'):
        key_data = f"prediction:{post_content}:{sorted(tags or [])}:{language}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    return cache_response(ttl=ttl, namespace="predictions", key_func=key_func)


class SessionCache:
    """
    Enhanced session management with Redis backend
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.session_ttl = 86400  # 24 hours
        self.namespace = "sessions"
    
    def create_session(self, session_id: str, data: Dict) -> bool:
        """Create a new session"""
        session_data = {
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'data': data
        }
        return self.cache.set(session_id, session_data, self.session_ttl, self.namespace)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        session = self.cache.get(session_id, self.namespace)
        if session:
            # Update last accessed time
            session['last_accessed'] = datetime.now().isoformat()
            self.cache.set(session_id, session, self.session_ttl, self.namespace)
            return session['data']
        return None
    
    def update_session(self, session_id: str, data: Dict) -> bool:
        """Update session data"""
        session = self.cache.get(session_id, self.namespace)
        if session:
            session['data'].update(data)
            session['last_accessed'] = datetime.now().isoformat()
            return self.cache.set(session_id, session, self.session_ttl, self.namespace)
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.cache.delete(session_id, self.namespace)


class RateLimitCache:
    """
    Rate limiting using cache backend
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.namespace = "rate_limits"
    
    def is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, Dict]:
        """
        Check if a key is rate limited
        
        Args:
            key: Rate limit key (e.g., user_id, ip_address)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_limited, info_dict)
        """
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        # Get current request count
        cache_key = f"{key}:{current_time // window_seconds}"
        current_count = self.cache.get(cache_key, self.namespace) or 0
        
        if current_count >= max_requests:
            return True, {
                'limited': True,
                'current_count': current_count,
                'max_requests': max_requests,
                'reset_time': (current_time // window_seconds + 1) * window_seconds,
                'retry_after': window_seconds - (current_time % window_seconds)
            }
        
        # Increment counter
        new_count = current_count + 1
        self.cache.set(cache_key, new_count, window_seconds, self.namespace)
        
        return False, {
            'limited': False,
            'current_count': new_count,
            'max_requests': max_requests,
            'remaining': max_requests - new_count,
            'reset_time': (current_time // window_seconds + 1) * window_seconds
        }


# Global cache manager instance
cache_manager = CacheManager()
session_cache = SessionCache(cache_manager)
rate_limit_cache = RateLimitCache(cache_manager)


def initialize_cache(redis_url: str = None):
    """Initialize the global cache manager"""
    global cache_manager, session_cache, rate_limit_cache
    
    if redis_url:
        cache_manager = CacheManager(redis_url)
    
    session_cache = SessionCache(cache_manager)
    rate_limit_cache = RateLimitCache(cache_manager)
    
    return cache_manager


def get_cache_manager():
    """Get the global cache manager instance"""
    return cache_manager


def clear_all_caches():
    """Clear all caches (use with caution)"""
    namespaces = ["responses", "features", "predictions", "sessions", "rate_limits"]
    total_cleared = 0
    
    for namespace in namespaces:
        cleared = cache_manager.clear_namespace(namespace)
        total_cleared += cleared
        print(f"🧹 Cleared {cleared} items from {namespace} cache")
    
    print(f"✅ Total items cleared: {total_cleared}")
    return total_cleared


# Enhanced cache warming functions
def warm_feature_cache(posts: List[Dict]):
    """Pre-populate feature cache with common posts"""
    from src.ml.feature_engineering import AdvancedFeatureEngineer
    
    print("🔥 Warming feature cache...")
    feature_engineer = AdvancedFeatureEngineer()
    
    for i, post in enumerate(posts[:50]):  # Warm with first 50 posts
        try:
            features = feature_engineer.extract_all_features(
                post['text'], 
                post.get('tags', []), 
                post.get('language', 'English')
            )
            
            # Cache will be populated by the @cache_features decorator
            if (i + 1) % 10 == 0:
                print(f"  Cached features for {i + 1} posts...")
                
        except Exception as e:
            print(f"  Error caching features for post {i}: {e}")
    
    print("✅ Feature cache warming completed")


if __name__ == "__main__":
    # Test cache functionality
    print("🧪 Testing cache functionality...")
    
    # Test basic operations
    cache_manager.set("test_key", {"message": "Hello, Cache!"}, 60)
    result = cache_manager.get("test_key")
    print(f"Cache test result: {result}")
    
    # Test statistics
    stats = cache_manager.get_stats()
    print(f"Cache stats: {stats}")
    
    print("✅ Cache test completed")
