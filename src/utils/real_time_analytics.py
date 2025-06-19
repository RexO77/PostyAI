"""
Real-time Analytics and Engagement Tracking System

This module provides real-time analytics, trend detection, and performance monitoring
for generated posts and user engagement patterns.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import threading
import queue


class RealTimeAnalytics:
    """
    Real-time analytics engine for post performance and user engagement
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.post_metrics = {}
        self.user_sessions = {}
        self.real_time_events = deque(maxlen=max_history)
        self.trending_topics = defaultdict(int)
        self.performance_cache = {}
        self.cache_expiry = {}
        
        # Event queue for real-time processing
        self.event_queue = queue.Queue()
        self.is_running = False
        self.analytics_thread = None
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'average': 0.4,
            'poor': 0.2
        }
    
    def start_real_time_processing(self):
        """Start real-time analytics processing"""
        if not self.is_running:
            self.is_running = True
            self.analytics_thread = threading.Thread(target=self._process_events, daemon=True)
            self.analytics_thread.start()
            print("🔄 Real-time analytics engine started")
    
    def stop_real_time_processing(self):
        """Stop real-time analytics processing"""
        self.is_running = False
        if self.analytics_thread:
            self.analytics_thread.join()
        print("⏹️ Real-time analytics engine stopped")
    
    def track_post_generation(self, post_id: str, content: str, params: Dict, 
                            predicted_engagement: Optional[float] = None):
        """Track when a post is generated"""
        timestamp = datetime.now()
        
        event = {
            'type': 'post_generated',
            'post_id': post_id,
            'timestamp': timestamp,
            'content_length': len(content),
            'word_count': len(content.split()),
            'params': params,
            'predicted_engagement': predicted_engagement
        }
        
        self.real_time_events.append(event)
        self.event_queue.put(event)
        
        # Initialize post metrics
        self.post_metrics[post_id] = {
            'generated_at': timestamp,
            'content': content,
            'params': params,
            'predicted_engagement': predicted_engagement,
            'views': 0,
            'copies': 0,
            'regenerations': 0,
            'user_rating': None,
            'performance_score': 0.0
        }
    
    def track_post_interaction(self, post_id: str, interaction_type: str, 
                             user_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Track user interactions with generated posts"""
        timestamp = datetime.now()
        
        event = {
            'type': 'post_interaction',
            'post_id': post_id,
            'interaction_type': interaction_type,  # 'view', 'copy', 'regenerate', 'rate'
            'user_id': user_id,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        self.real_time_events.append(event)
        self.event_queue.put(event)
        
        # Update post metrics
        if post_id in self.post_metrics:
            if interaction_type == 'view':
                self.post_metrics[post_id]['views'] += 1
            elif interaction_type == 'copy':
                self.post_metrics[post_id]['copies'] += 1
            elif interaction_type == 'regenerate':
                self.post_metrics[post_id]['regenerations'] += 1
            elif interaction_type == 'rate' and metadata:
                self.post_metrics[post_id]['user_rating'] = metadata.get('rating')
            
            # Update performance score
            self._update_performance_score(post_id)
    
    def track_user_session(self, user_id: str, session_data: Dict):
        """Track user session information"""
        timestamp = datetime.now()
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'first_visit': timestamp,
                'sessions': []
            }
        
        session_data['timestamp'] = timestamp
        self.user_sessions[user_id]['sessions'].append(session_data)
        
        # Keep only recent sessions
        cutoff = timestamp - timedelta(days=30)
        self.user_sessions[user_id]['sessions'] = [
            s for s in self.user_sessions[user_id]['sessions'] 
            if s['timestamp'] > cutoff
        ]
    
    def _update_performance_score(self, post_id: str):
        """Calculate and update performance score for a post"""
        if post_id not in self.post_metrics:
            return
        
        metrics = self.post_metrics[post_id]
        
        # Calculate performance based on multiple factors
        view_score = min(metrics['views'] / 10, 1.0)  # Normalize to 0-1
        copy_score = metrics['copies'] * 0.3  # Copying is a strong engagement signal
        rating_score = (metrics['user_rating'] or 0) / 5.0  # Normalize rating to 0-1
        
        # Time decay factor (newer posts get slight boost)
        time_diff = datetime.now() - metrics['generated_at']
        time_factor = max(0.5, 1.0 - (time_diff.days / 30))  # Decay over 30 days
        
        # Combined performance score
        performance_score = (view_score + copy_score + rating_score) * time_factor
        metrics['performance_score'] = min(performance_score, 1.0)
    
    def get_trending_topics(self, time_window: timedelta = timedelta(hours=24)) -> List[Tuple[str, int]]:
        """Get trending topics based on recent post generation"""
        cutoff_time = datetime.now() - time_window
        topic_counts = defaultdict(int)
        
        for event in self.real_time_events:
            if (event['type'] == 'post_generated' and 
                event['timestamp'] > cutoff_time and 
                'params' in event):
                
                tag = event['params'].get('tag')
                if tag:
                    topic_counts[tag] += 1
        
        # Sort by frequency
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    
    def get_live_metrics(self) -> Dict:
        """Get real-time system metrics"""
        now = datetime.now()
        
        # Check cache first
        cache_key = 'live_metrics'
        if (cache_key in self.performance_cache and 
            cache_key in self.cache_expiry and
            now < self.cache_expiry[cache_key]):
            return self.performance_cache[cache_key]
        
        # Calculate fresh metrics
        recent_events = [
            e for e in self.real_time_events 
            if now - e['timestamp'] < timedelta(hours=1)
        ]
        
        hourly_stats = {
            'posts_generated': len([e for e in recent_events if e['type'] == 'post_generated']),
            'total_interactions': len([e for e in recent_events if e['type'] == 'post_interaction']),
            'active_users': len(set(e.get('user_id') for e in recent_events if e.get('user_id'))),
            'average_engagement_prediction': 0
        }
        
        # Calculate average predicted engagement
        predictions = [
            e.get('predicted_engagement', 0) 
            for e in recent_events 
            if e['type'] == 'post_generated' and e.get('predicted_engagement')
        ]
        
        if predictions:
            hourly_stats['average_engagement_prediction'] = np.mean(predictions)
        
        # Overall system health
        total_posts = len(self.post_metrics)
        avg_performance = np.mean([m['performance_score'] for m in self.post_metrics.values()]) if total_posts > 0 else 0
        
        metrics = {
            'timestamp': now.isoformat(),
            'system_health': {
                'status': 'healthy' if avg_performance > 0.3 else 'warning',
                'total_posts_generated': total_posts,
                'average_performance_score': avg_performance,
                'active_posts': len([p for p in self.post_metrics.values() if p['views'] > 0])
            },
            'hourly_stats': hourly_stats,
            'trending_topics': self.get_trending_topics()[:5],
            'performance_distribution': self._get_performance_distribution()
        }
        
        # Cache for 30 seconds
        self.performance_cache[cache_key] = metrics
        self.cache_expiry[cache_key] = now + timedelta(seconds=30)
        
        return metrics
    
    def _get_performance_distribution(self) -> Dict[str, int]:
        """Get distribution of posts by performance level"""
        distribution = {'excellent': 0, 'good': 0, 'average': 0, 'poor': 0}
        
        for metrics in self.post_metrics.values():
            score = metrics['performance_score']
            
            if score >= self.performance_thresholds['excellent']:
                distribution['excellent'] += 1
            elif score >= self.performance_thresholds['good']:
                distribution['good'] += 1
            elif score >= self.performance_thresholds['average']:
                distribution['average'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def get_post_analytics(self, post_id: str) -> Optional[Dict]:
        """Get detailed analytics for a specific post"""
        if post_id not in self.post_metrics:
            return None
        
        metrics = self.post_metrics[post_id]
        
        # Get related events
        post_events = [
            e for e in self.real_time_events 
            if e.get('post_id') == post_id
        ]
        
        # Calculate engagement timeline
        timeline = []
        for event in post_events:
            timeline.append({
                'timestamp': event['timestamp'].isoformat(),
                'type': event['type'],
                'interaction_type': event.get('interaction_type')
            })
        
        return {
            'post_id': post_id,
            'generated_at': metrics['generated_at'].isoformat(),
            'content_preview': metrics['content'][:100] + '...' if len(metrics['content']) > 100 else metrics['content'],
            'params': metrics['params'],
            'metrics': {
                'views': metrics['views'],
                'copies': metrics['copies'],
                'regenerations': metrics['regenerations'],
                'user_rating': metrics['user_rating'],
                'performance_score': metrics['performance_score']
            },
            'predicted_engagement': metrics['predicted_engagement'],
            'timeline': sorted(timeline, key=lambda x: x['timestamp'])
        }
    
    def get_user_analytics(self, user_id: str) -> Optional[Dict]:
        """Get analytics for a specific user"""
        if user_id not in self.user_sessions:
            return None
        
        user_data = self.user_sessions[user_id]
        sessions = user_data['sessions']
        
        # Calculate user metrics
        total_sessions = len(sessions)
        total_posts_generated = len([
            e for e in self.real_time_events 
            if e.get('user_id') == user_id and e['type'] == 'post_generated'
        ])
        
        # User preferences analysis
        preferences = defaultdict(int)
        for event in self.real_time_events:
            if (event.get('user_id') == user_id and 
                event['type'] == 'post_generated' and 
                'params' in event):
                
                for key, value in event['params'].items():
                    preferences[f"{key}_{value}"] += 1
        
        return {
            'user_id': user_id,
            'first_visit': user_data['first_visit'].isoformat(),
            'total_sessions': total_sessions,
            'total_posts_generated': total_posts_generated,
            'average_session_length': np.mean([
                (s.get('duration', 0)) for s in sessions
            ]) if sessions else 0,
            'preferred_settings': dict(sorted(preferences.items(), key=lambda x: x[1], reverse=True)[:5]),
            'engagement_score': self._calculate_user_engagement(user_id)
        }
    
    def _calculate_user_engagement(self, user_id: str) -> float:
        """Calculate user engagement score"""
        user_events = [
            e for e in self.real_time_events 
            if e.get('user_id') == user_id
        ]
        
        if not user_events:
            return 0.0
        
        # Score based on activity type
        score = 0
        for event in user_events:
            if event['type'] == 'post_generated':
                score += 1
            elif event['type'] == 'post_interaction':
                interaction_type = event.get('interaction_type')
                if interaction_type == 'copy':
                    score += 2  # High engagement
                elif interaction_type == 'rate':
                    score += 1.5
                elif interaction_type == 'view':
                    score += 0.5
        
        # Normalize by time since first visit
        if user_id in self.user_sessions:
            days_active = max(1, (datetime.now() - self.user_sessions[user_id]['first_visit']).days)
            return min(score / days_active, 10.0)  # Cap at 10
        
        return score
    
    def _process_events(self):
        """Background thread to process real-time events"""
        while self.is_running:
            try:
                # Process events from queue
                event = self.event_queue.get(timeout=1)
                
                # Update trending topics
                if (event['type'] == 'post_generated' and 
                    'params' in event and 
                    'tag' in event['params']):
                    self.trending_topics[event['params']['tag']] += 1
                
                # Additional real-time processing can be added here
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing real-time event: {e}")
    
    def export_analytics(self, output_file: str = "analytics_export.json"):
        """Export analytics data"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_live_metrics(),
            'post_metrics': {
                pid: {
                    **metrics,
                    'generated_at': metrics['generated_at'].isoformat()
                } 
                for pid, metrics in self.post_metrics.items()
            },
            'user_sessions': {
                uid: {
                    'first_visit': data['first_visit'].isoformat(),
                    'session_count': len(data['sessions'])
                }
                for uid, data in self.user_sessions.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📊 Analytics exported to {output_file}")


# Global analytics instance
analytics_engine = RealTimeAnalytics()


def initialize_analytics():
    """Initialize the global analytics engine"""
    analytics_engine.start_real_time_processing()
    return analytics_engine


def get_analytics_instance():
    """Get the global analytics instance"""
    return analytics_engine
