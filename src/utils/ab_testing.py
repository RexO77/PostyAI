"""
A/B Testing Framework for PostyAI

This module provides comprehensive A/B testing capabilities for different
post generation strategies, prompts, and model configurations.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
from enum import Enum
import hashlib


class TestStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class ABTestFramework:
    """
    Comprehensive A/B testing framework for post generation optimization
    """
    
    def __init__(self):
        self.tests = {}
        self.user_assignments = {}
        self.test_results = defaultdict(list)
        self.statistical_significance = 0.95
        self.minimum_sample_size = 30
    
    def create_test(self, 
                   test_name: str,
                   description: str,
                   variants: List[Dict],
                   traffic_allocation: Optional[List[float]] = None,
                   duration_days: int = 7,
                   success_metrics: List[str] = None) -> str:
        """
        Create a new A/B test
        
        Args:
            test_name: Unique name for the test
            description: Description of what is being tested
            variants: List of variant configurations
            traffic_allocation: Traffic split between variants (must sum to 1.0)
            duration_days: Test duration in days
            success_metrics: List of metrics to track for success
            
        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        
        # Validate inputs
        if not variants or len(variants) < 2:
            raise ValueError("Must have at least 2 variants")
        
        if traffic_allocation:
            if len(traffic_allocation) != len(variants):
                raise ValueError("Traffic allocation must match number of variants")
            if abs(sum(traffic_allocation) - 1.0) > 0.001:
                raise ValueError("Traffic allocation must sum to 1.0")
        else:
            # Equal split
            traffic_allocation = [1.0 / len(variants)] * len(variants)
        
        # Default success metrics
        if success_metrics is None:
            success_metrics = ['user_rating', 'copy_rate', 'engagement_score']
        
        test_config = {
            'test_id': test_id,
            'name': test_name,
            'description': description,
            'variants': variants,
            'traffic_allocation': traffic_allocation,
            'success_metrics': success_metrics,
            'status': TestStatus.DRAFT,
            'created_at': datetime.now(),
            'start_date': None,
            'end_date': None,
            'duration_days': duration_days,
            'results': {variant['name']: [] for variant in variants},
            'statistical_results': None
        }
        
        self.tests[test_id] = test_config
        print(f"✅ Created A/B test '{test_name}' with ID: {test_id}")
        
        return test_id
    
    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        if test_id not in self.tests:
            return False
        
        test = self.tests[test_id]
        
        if test['status'] != TestStatus.DRAFT:
            print(f"❌ Test {test_id} is not in draft status")
            return False
        
        test['status'] = TestStatus.ACTIVE
        test['start_date'] = datetime.now()
        test['end_date'] = test['start_date'] + timedelta(days=test['duration_days'])
        
        print(f"🚀 Started A/B test '{test['name']}'")
        return True
    
    def stop_test(self, test_id: str) -> bool:
        """Stop an A/B test"""
        if test_id not in self.tests:
            return False
        
        test = self.tests[test_id]
        test['status'] = TestStatus.COMPLETED
        test['end_date'] = datetime.now()
        
        # Calculate final results
        self._calculate_statistical_significance(test_id)
        
        print(f"⏹️ Stopped A/B test '{test['name']}'")
        return True
    
    def assign_user_to_variant(self, test_id: str, user_id: str) -> Optional[Dict]:
        """
        Assign a user to a test variant using consistent hashing
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            
        Returns:
            Variant configuration or None if test not active
        """
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        
        # Check if test is active
        if test['status'] != TestStatus.ACTIVE:
            return None
        
        # Check if test has expired
        if test['end_date'] and datetime.now() > test['end_date']:
            self.stop_test(test_id)
            return None
        
        # Check if user is already assigned
        assignment_key = f"{test_id}:{user_id}"
        if assignment_key in self.user_assignments:
            variant_name = self.user_assignments[assignment_key]
            return next((v for v in test['variants'] if v['name'] == variant_name), None)
        
        # Assign user to variant using consistent hashing
        hash_input = f"{test_id}:{user_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Determine variant based on traffic allocation
        cumulative_allocation = 0
        for i, allocation in enumerate(test['traffic_allocation']):
            cumulative_allocation += allocation
            if normalized_hash <= cumulative_allocation:
                variant = test['variants'][i]
                self.user_assignments[assignment_key] = variant['name']
                return variant
        
        # Fallback to last variant
        variant = test['variants'][-1]
        self.user_assignments[assignment_key] = variant['name']
        return variant
    
    def record_result(self, test_id: str, user_id: str, metrics: Dict[str, Any]):
        """
        Record test results for a user
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            metrics: Dictionary of metric values
        """
        if test_id not in self.tests:
            return
        
        test = self.tests[test_id]
        assignment_key = f"{test_id}:{user_id}"
        
        if assignment_key not in self.user_assignments:
            return
        
        variant_name = self.user_assignments[assignment_key]
        
        result_record = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'variant': variant_name
        }
        
        test['results'][variant_name].append(result_record)
    
    def get_test_results(self, test_id: str) -> Optional[Dict]:
        """Get current results for a test"""
        if test_id not in self.tests:
            return None
        
        test = self.tests[test_id]
        
        # Calculate statistics for each variant
        variant_stats = {}
        
        for variant in test['variants']:
            variant_name = variant['name']
            results = test['results'][variant_name]
            
            if not results:
                variant_stats[variant_name] = {
                    'sample_size': 0,
                    'metrics': {}
                }
                continue
            
            stats = {
                'sample_size': len(results),
                'metrics': {}
            }
            
            # Calculate metrics statistics
            for metric in test['success_metrics']:
                values = [r['metrics'].get(metric) for r in results if metric in r['metrics']]
                values = [v for v in values if v is not None]
                
                if values:
                    stats['metrics'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            variant_stats[variant_name] = stats
        
        return {
            'test_id': test_id,
            'name': test['name'],
            'status': test['status'].value,
            'start_date': test['start_date'].isoformat() if test['start_date'] else None,
            'end_date': test['end_date'].isoformat() if test['end_date'] else None,
            'variant_stats': variant_stats,
            'statistical_results': test.get('statistical_results'),
            'total_participants': sum(len(test['results'][v['name']]) for v in test['variants'])
        }
    
    def _calculate_statistical_significance(self, test_id: str):
        """Calculate statistical significance of test results"""
        test = self.tests[test_id]
        
        # Simple two-sample t-test for the primary metric
        if len(test['variants']) != 2:
            return  # Only support 2-variant tests for now
        
        variant_a, variant_b = test['variants']
        results_a = test['results'][variant_a['name']]
        results_b = test['results'][variant_b['name']]
        
        if len(results_a) < self.minimum_sample_size or len(results_b) < self.minimum_sample_size:
            test['statistical_results'] = {
                'significance': 'insufficient_data',
                'message': f'Need at least {self.minimum_sample_size} samples per variant'
            }
            return
        
        # Calculate for each success metric
        primary_metric = test['success_metrics'][0]
        
        values_a = [r['metrics'].get(primary_metric) for r in results_a if primary_metric in r['metrics']]
        values_b = [r['metrics'].get(primary_metric) for r in results_b if primary_metric in r['metrics']]
        
        values_a = [v for v in values_a if v is not None]
        values_b = [v for v in values_b if v is not None]
        
        if not values_a or not values_b:
            test['statistical_results'] = {
                'significance': 'no_data',
                'message': 'No data available for primary metric'
            }
            return
        
        # Simple statistical test (in production, use scipy.stats)
        mean_a, mean_b = np.mean(values_a), np.mean(values_b)
        std_a, std_b = np.std(values_a), np.std(values_b)
        n_a, n_b = len(values_a), len(values_b)
        
        # Calculate effect size
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        # Simple significance test (replace with proper t-test in production)
        improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0
        
        test['statistical_results'] = {
            'primary_metric': primary_metric,
            'variant_a_mean': mean_a,
            'variant_b_mean': mean_b,
            'improvement_percent': improvement,
            'effect_size': effect_size,
            'sample_sizes': {'variant_a': n_a, 'variant_b': n_b},
            'significance': 'significant' if abs(effect_size) > 0.2 else 'not_significant',
            'recommendation': self._get_recommendation(effect_size, improvement)
        }
    
    def _get_recommendation(self, effect_size: float, improvement: float) -> str:
        """Generate recommendation based on test results"""
        if abs(effect_size) < 0.1:
            return "No meaningful difference detected. Continue with current approach."
        elif improvement > 5 and effect_size > 0.2:
            return "Variant B shows significant improvement. Consider implementing."
        elif improvement < -5 and effect_size < -0.2:
            return "Variant A performs better. Stick with control."
        else:
            return "Results are inconclusive. Consider extending test duration."
    
    def create_prompt_test(self, base_prompt: str, variations: List[str], 
                          test_name: str = None) -> str:
        """
        Create an A/B test for different prompt variations
        
        Args:
            base_prompt: The control prompt
            variations: List of prompt variations to test
            test_name: Optional test name
            
        Returns:
            Test ID
        """
        if not test_name:
            test_name = f"Prompt_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        variants = [
            {
                'name': 'control',
                'prompt': base_prompt,
                'description': 'Original prompt'
            }
        ]
        
        for i, variation in enumerate(variations):
            variants.append({
                'name': f'variation_{i+1}',
                'prompt': variation,
                'description': f'Prompt variation {i+1}'
            })
        
        return self.create_test(
            test_name=test_name,
            description=f"Testing prompt variations for better engagement",
            variants=variants,
            success_metrics=['user_rating', 'copy_rate', 'predicted_engagement']
        )
    
    def create_model_parameter_test(self, parameter_sets: List[Dict], 
                                   test_name: str = None) -> str:
        """
        Create an A/B test for different model parameters
        
        Args:
            parameter_sets: List of parameter configurations
            test_name: Optional test name
            
        Returns:
            Test ID
        """
        if not test_name:
            test_name = f"Parameter_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        variants = []
        for i, params in enumerate(parameter_sets):
            variants.append({
                'name': f'config_{i+1}',
                'parameters': params,
                'description': f'Parameter configuration {i+1}'
            })
        
        return self.create_test(
            test_name=test_name,
            description="Testing different model parameter configurations",
            variants=variants,
            success_metrics=['predicted_engagement', 'generation_time', 'user_rating']
        )
    
    def get_active_tests(self) -> List[Dict]:
        """Get all active tests"""
        active_tests = []
        
        for test_id, test in self.tests.items():
            if test['status'] == TestStatus.ACTIVE:
                # Check if test has expired
                if test['end_date'] and datetime.now() > test['end_date']:
                    self.stop_test(test_id)
                else:
                    active_tests.append({
                        'test_id': test_id,
                        'name': test['name'],
                        'description': test['description'],
                        'start_date': test['start_date'].isoformat(),
                        'end_date': test['end_date'].isoformat(),
                        'variants': len(test['variants']),
                        'participants': sum(len(test['results'][v['name']]) for v in test['variants'])
                    })
        
        return active_tests
    
    def export_test_data(self, test_id: str, output_file: str = None):
        """Export test data for external analysis"""
        if test_id not in self.tests:
            return
        
        test = self.tests[test_id]
        
        if not output_file:
            output_file = f"ab_test_{test['name']}_{test_id[:8]}.json"
        
        export_data = {
            'test_config': {
                'test_id': test_id,
                'name': test['name'],
                'description': test['description'],
                'variants': test['variants'],
                'traffic_allocation': test['traffic_allocation'],
                'success_metrics': test['success_metrics'],
                'duration_days': test['duration_days']
            },
            'results': test['results'],
            'statistical_analysis': test.get('statistical_results'),
            'user_assignments': {
                k: v for k, v in self.user_assignments.items() 
                if k.startswith(f"{test_id}:")
            }
        }
        
        # Convert datetime objects to strings
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=datetime_converter)
        
        print(f"📊 Test data exported to {output_file}")


# Global A/B testing instance
ab_testing_framework = ABTestFramework()


def get_ab_testing_framework():
    """Get the global A/B testing framework instance"""
    return ab_testing_framework
