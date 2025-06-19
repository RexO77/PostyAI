"""
Advanced Engagement Prediction System

This module implements multiple machine learning algorithms to predict social media engagement,
including comprehensive evaluation, feature importance analysis, and model comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import joblib
from datetime import datetime

# ML Models
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    VotingRegressor,
    BaggingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV, 
    RandomizedSearchCV,
    KFold,
    StratifiedKFold
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Advanced preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .feature_engineering import AdvancedFeatureEngineer


class EngagementPredictor:
    """
    Comprehensive engagement prediction system with multiple algorithms and advanced optimization
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engineer = AdvancedFeatureEngineer()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        self.feature_selector = None
        self.preprocessing_pipeline = None
        
        # Initialize models with enhanced configurations
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all machine learning models with optimized hyperparameters"""
        self.models = {
            'random_forest_optimized': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                bootstrap=True
            ),
            'gradient_boosting_optimized': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            ),
            'extra_trees_optimized': ExtraTreesRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                bootstrap=True
            ),
            'xgboost_optimized': xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm_optimized': lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                num_leaves=31,
                min_data_in_leaf=20,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                lambda_l1=0.1,
                lambda_l2=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'bayesian_ridge': BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
                compute_score=True
            ),
            'elastic_net_optimized': ElasticNet(
                alpha=0.5,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            ),
            'svr_optimized': SVR(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                epsilon=0.1
            ),
            # Ensemble methods
            'voting_regressor': None,  # Will be initialized after base models
            'bagging_rf': BaggingRegressor(
                RandomForestRegressor(n_estimators=50, random_state=42),
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Initialize scalers for each model
        self.scalers = {
            name: StandardScaler() if 'svr' in name or 'ridge' in name or 'lasso' in name or 'elastic' in name 
            else None for name in self.models.keys()
        }
    
    def _create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create advanced preprocessing pipeline"""
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing for numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', QuantileTransformer(output_distribution='normal'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )
        
        return preprocessor
    
    def _perform_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, object]:
        """Perform advanced feature selection"""
        print("🔍 Performing feature selection...")
        
        # Method 1: Statistical feature selection
        k_best = SelectKBest(score_func=f_regression, k='all')
        k_best.fit(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': k_best.scores_
        }).sort_values('score', ascending=False)
        
        # Select top features (top 80% of features)
        n_features = max(10, int(len(X.columns) * 0.8))
        top_features = feature_scores.head(n_features)['feature'].tolist()
        
        print(f"✅ Selected {len(top_features)} features out of {len(X.columns)}")
        
        return X[top_features], top_features
    
    def _create_ensemble_model(self):
        """Create voting ensemble from best performing base models"""
        base_models = [
            ('rf', self.models['random_forest_optimized']),
            ('gb', self.models['gradient_boosting_optimized']),
            ('xgb', self.models['xgboost_optimized']),
            ('lgb', self.models['lightgbm_optimized'])
        ]
        
        voting_reg = VotingRegressor(
            estimators=base_models,
            n_jobs=-1
        )
        
        self.models['voting_regressor'] = voting_reg

    def prepare_data(self, posts_data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training
        
        Args:
            posts_data: List of post dictionaries
            
        Returns:
            Features DataFrame and target Series
        """
        # Create feature dataframe
        df = self.feature_engineer.create_feature_dataframe(posts_data)
        
        # Separate features and target
        target_col = 'engagement'
        feature_cols = [col for col in df.columns if col not in [target_col, 'language']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train all models and evaluate performance
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary of model results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Scale data for linear models and SVR
            if name in ['ridge', 'lasso', 'elastic_net', 'svr']:
                scaler = self.scalers[name]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'predictions': {
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred_test.tolist()
                }
            }
            
            print(f"  R² Score: {test_metrics['r2']:.4f}")
            print(f"  MAE: {test_metrics['mae']:.2f}")
            print(f"  RMSE: {test_metrics['rmse']:.2f}")
        
        # Find best model
        self._find_best_model()
        
        return self.results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        }
    
    def _find_best_model(self):
        """Find the best performing model based on test R² score"""
        best_r2 = -float('inf')
        
        for name, result in self.results.items():
            r2_score = result['test_metrics']['r2']
            if r2_score > best_r2:
                best_r2 = r2_score
                self.best_model_name = name
                self.best_model = result['model']
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, models_to_tune: List[str] = None):
        """
        Perform hyperparameter tuning for specified models
        
        Args:
            X: Features DataFrame
            y: Target Series
            models_to_tune: List of model names to tune (default: best 3 models)
        """
        if models_to_tune is None:
            # Select top 3 models based on current performance
            sorted_models = sorted(
                self.results.items(),
                key=lambda x: x[1]['test_metrics']['r2'],
                reverse=True
            )
            models_to_tune = [name for name, _ in sorted_models[:3]]
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        tuned_results = {}
        
        for model_name in models_to_tune:
            if model_name in param_grids:
                print(f"Tuning {model_name}...")
                
                model = self.models[model_name]
                param_grid = param_grids[model_name]
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, scoring='r2', n_jobs=-1
                )
                
                grid_search.fit(X, y)
                
                tuned_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_model': grid_search.best_estimator_
                }
                
                print(f"  Best R² Score: {grid_search.best_score_:.4f}")
                print(f"  Best Params: {grid_search.best_params_}")
        
        return tuned_results
    
    def train_with_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train models with hyperparameter tuning for better performance
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary of model results with optimized hyperparameters
        """
        print("🚀 Starting enhanced training with hyperparameter tuning...")
        
        # Perform feature selection first
        X_selected, selected_features = self._perform_feature_selection(X, y)
        self.selected_features = selected_features
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42, stratify=None
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Create ensemble model
        self._create_ensemble_model()
        
        self.results = {}
        best_score = -np.inf
        
        # Define hyperparameter grids for optimization
        param_grids = {
            'random_forest_optimized': {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost_optimized': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'gradient_boosting_optimized': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9]
            }
        }
        
        for name, model in self.models.items():
            if model is None:  # Skip uninitialized models
                continue
                
            print(f"🔧 Training and tuning {name}...")
            
            try:
                # Prepare data for model
                if name in ['bayesian_ridge', 'elastic_net_optimized', 'svr_optimized']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Hyperparameter tuning for selected models
                if name in param_grids and X_train.shape[0] > 50:  # Only tune if enough data
                    print(f"  🎯 Performing hyperparameter tuning for {name}...")
                    
                    grid_search = RandomizedSearchCV(
                        model, 
                        param_grids[name],
                        n_iter=20,  # Limit iterations for speed
                        cv=min(5, X_train.shape[0] // 10),  # Adaptive CV folds
                        scoring='r2',
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    
                    print(f"  ✅ Best params for {name}: {grid_search.best_params_}")
                    print(f"  📈 Best CV score: {grid_search.best_score_:.4f}")
                    
                    # Use the best model
                    self.models[name] = best_model
                    model = best_model
                else:
                    # Train with default parameters
                    model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate comprehensive metrics
                train_metrics = self._calculate_enhanced_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_enhanced_metrics(y_test, y_pred_test)
                
                # Enhanced cross-validation with stratification
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=min(5, X_train.shape[0] // 10), 
                    scoring='r2'
                )
                
                # Feature importance analysis
                feature_importance = self._get_feature_importance(model, X_train.columns)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'feature_importance': feature_importance,
                    'predictions': {
                        'y_test': y_test.tolist(),
                        'y_pred': y_pred_test.tolist()
                    }
                }
                
                # Track best model
                current_score = test_metrics['r2']
                if current_score > best_score:
                    best_score = current_score
                    self.best_model_name = name
                    self.best_model = model
                
                print(f"  ✅ {name} - R² Score: {current_score:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {str(e)}")
                continue
        
        print(f"\n🏆 Best model: {self.best_model_name} with R² score: {best_score:.4f}")
        
        return self.results
    
    def _calculate_enhanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import explained_variance_score, max_error
        
        # Avoid division by zero
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        explained_var = explained_variance_score(y_true, y_pred)
        max_err = max_error(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error) - handle zero values
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
        
        # Median Absolute Error
        medae = np.median(np.abs(y_true - y_pred))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_var,
            'max_error': max_err,
            'mape': mape,
            'median_absolute_error': medae
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained models"""
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
                importance_dict = dict(zip(feature_names, importances))
            elif hasattr(model, 'estimators_'):
                # Ensemble models - try to get from base estimators
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    # Average importance across estimators
                    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                    importance_dict = dict(zip(feature_names, importances))
        except:
            # If feature importance extraction fails, return empty dict
            pass
        
        return importance_dict
    
    def predict_engagement(self, text: str, tags: List[str] = None, language: str = 'English') -> Dict[str, Any]:
        """
        Predict engagement for a new post
        
        Args:
            text: Post content
            tags: List of post tags
            language: Post language
            
        Returns:
            Prediction results
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Extract features
        features = self.feature_engineer.extract_all_features(text, tags, language)
        feature_df = pd.DataFrame([features])
        
        # Use the same features as training
        training_features = list(self.results[self.best_model_name]['feature_importance'].keys()) if self.results[self.best_model_name]['feature_importance'] else feature_df.columns
        feature_df = feature_df[training_features]
        
        # Scale if needed
        if self.best_model_name in ['ridge', 'lasso', 'elastic_net', 'svr']:
            feature_df = self.scalers[self.best_model_name].transform(feature_df)
        
        # Make prediction
        prediction = self.best_model.predict(feature_df)[0]
        
        # Calculate confidence interval (simple approach)
        confidence_interval = self._calculate_prediction_confidence(feature_df)
        
        return {
            'predicted_engagement': float(prediction),
            'confidence_interval': confidence_interval,
            'model_used': self.best_model_name,
            'features_used': len(training_features)
        }
    
    def _calculate_prediction_confidence(self, features: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate simple confidence interval for prediction
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Simple approach: use RMSE from test set
        if self.best_model_name in self.results:
            rmse = self.results[self.best_model_name]['test_metrics']['rmse']
            prediction = self.best_model.predict(features)[0]
            
            return (
                max(0, prediction - 1.96 * rmse),  # 95% confidence interval
                prediction + 1.96 * rmse
            )
        
        return (0, 1000)  # Default wide interval
    
    def create_model_comparison_report(self) -> str:
        """Create a comprehensive model comparison report"""
        if not self.results:
            return "No models have been trained yet."
        
        # Sort models by performance
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_metrics']['r2'],
            reverse=True
        )
        
        report = "\n" + "="*80 + "\n"
        report += "🏆 MODEL PERFORMANCE COMPARISON - ENGAGEMENT PREDICTION\n"
        report += "="*80 + "\n\n"
        
        # Header
        report += f"{'Rank':<5} {'Model':<15} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV Score':<12}\n"
        report += "-" * 70 + "\n"
        
        # Model results
        for i, (name, result) in enumerate(sorted_results, 1):
            test_metrics = result['test_metrics']
            cv_score = result['cv_score_mean']
            
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            report += f"{emoji} {i:<3} {name:<15} {test_metrics['r2']:<8.4f} {test_metrics['rmse']:<8.1f} {test_metrics['mae']:<8.1f} {cv_score:<8.4f} ± {result['cv_score_std']:.3f}\n"
        
        # Best model details
        best_name, best_result = sorted_results[0]
        report += f"\n🎯 BEST MODEL: {best_name.upper()}\n"
        report += "-" * 40 + "\n"
        report += f"R² Score: {best_result['test_metrics']['r2']:.4f}\n"
        report += f"RMSE: {best_result['test_metrics']['rmse']:.2f} engagement points\n"
        report += f"MAE: {best_result['test_metrics']['mae']:.2f} engagement points\n"
        report += f"MAPE: {best_result['test_metrics']['mape']:.2f}%\n"
        
        return report
    
    def save_models(self, filepath: str):
        """Save all trained models and results"""
        save_data = {
            'models': {},
            'scalers': self.scalers,
            'results': self.results,
            'best_model_name': self.best_model_name,
            'feature_engineer': self.feature_engineer,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save each model separately due to different types
        for name, model in self.models.items():
            model_filepath = filepath.replace('.pkl', f'_{name}.pkl')
            joblib.dump(model, model_filepath)
        
        # Save metadata
        metadata_filepath = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_filepath, 'w') as f:
            # Convert results to JSON-serializable format
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            json_results = {}
            for name, result in self.results.items():
                json_results[name] = convert_numpy_types({
                    'train_metrics': result['train_metrics'],
                    'test_metrics': result['test_metrics'],
                    'cv_score_mean': result['cv_score_mean'],
                    'cv_score_std': result['cv_score_std'],
                    'feature_importance': result['feature_importance'],
                    'predictions': result['predictions']
                })
            
            save_data['results'] = json_results
            del save_data['models']  # Remove models from JSON
            del save_data['scalers']  # Remove scalers from JSON
            del save_data['feature_engineer']  # Remove feature engineer from JSON
            
            json.dump(save_data, f, indent=2)
        
        print(f"Models saved to {filepath}")
    
    def create_visualizations(self) -> Dict[str, Any]:
        """Create comprehensive visualizations of model performance"""
        if not self.results:
            return {}
        
        visualizations = {}
        
        # 1. Model Comparison Bar Chart
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['test_metrics']['r2'] for name in model_names]
        
        fig_comparison = go.Figure(data=[
            go.Bar(x=model_names, y=r2_scores, text=[f'{score:.3f}' for score in r2_scores])
        ])
        fig_comparison.update_layout(
            title="Model Performance Comparison (R² Score)",
            xaxis_title="Models",
            yaxis_title="R² Score",
            showlegend=False
        )
        visualizations['model_comparison'] = fig_comparison
        
        # 2. Prediction vs Actual Scatter Plot (Best Model)
        if self.best_model_name:
            best_result = self.results[self.best_model_name]
            y_test = best_result['predictions']['y_test']
            y_pred = best_result['predictions']['y_pred']
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode='markers',
                name='Predictions',
                text=[f'Actual: {actual}<br>Predicted: {pred:.1f}' for actual, pred in zip(y_test, y_pred)]
            ))
            
            # Add perfect prediction line
            min_val, max_val = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            fig_scatter.update_layout(
                title=f"Predictions vs Actual Values ({self.best_model_name})",
                xaxis_title="Actual Engagement",
                yaxis_title="Predicted Engagement"
            )
            visualizations['prediction_scatter'] = fig_scatter
        
        # 3. Feature Importance (if available)
        if self.best_model_name and self.results[self.best_model_name]['feature_importance']:
            importance = self.results[self.best_model_name]['feature_importance']
            features = list(importance.keys())
            importances = list(importance.values())
            
            # Sort by importance
            sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_pairs[:20])  # Top 20 features
            
            fig_importance = go.Figure(data=[
                go.Bar(x=list(importances), y=list(features), orientation='h')
            ])
            fig_importance.update_layout(
                title="Top 20 Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=600
            )
            visualizations['feature_importance'] = fig_importance
        
        return visualizations


if __name__ == "__main__":
    # Demo usage
    print("Engagement Predictor Demo")
    print("=" * 50)
    
    # This would normally load real data
    sample_posts = [
        {
            'text': 'Amazing tips for career growth! 🚀',
            'engagement': 150,
            'tags': ['Career', 'Tips'],
            'language': 'English'
        },
        {
            'text': 'Just completed a challenging project. Feeling accomplished!',
            'engagement': 89,
            'tags': ['Achievement'],
            'language': 'English'
        }
    ]
    
    predictor = EngagementPredictor()
    print("Predictor initialized successfully!")