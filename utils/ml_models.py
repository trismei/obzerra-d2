import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import warnings
from utils.feature_explainer import FeatureExplainer
warnings.filterwarnings('ignore')

# Try to import advanced models (LightGBM, XGBoost) - make them optional
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception as e:
    LGBM_AVAILABLE = False
    print(f"Warning: LightGBM not available ({type(e).__name__}), using enhanced RF/LR only")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception as e:
    XGB_AVAILABLE = False
    print(f"Warning: XGBoost not available ({type(e).__name__}), using enhanced RF/LR only")

class MLModelManager:
    """Manages machine learning models for fraud detection."""
    
    def __init__(self):
        # Core models as per capstone methodology
        self.logistic_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced',  # Handle imbalanced data
            C=0.1  # Regularization
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            random_state=42, 
            max_depth=15,  # Increased from 10
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'  # Handle imbalanced data
        )
        self.knn_model = KNeighborsClassifier(
            n_neighbors=7,  # Optimized for fraud detection
            weights='distance',  # Weight by inverse distance
            algorithm='auto',
            leaf_size=30,
            metric='minkowski',
            p=2  # Euclidean distance
        )
        
        # Initialize advanced models if available
        self.lgb_model = None
        self.xgb_model = None
        
        if LGBM_AVAILABLE:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
        
        if XGB_AVAILABLE:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,  # Handle imbalanced data
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
        
        self.scaler = StandardScaler()
        self.is_trained_flag = False
        self.feature_columns = []
        self.model_metrics = {}
        self.feature_importance = pd.DataFrame()
        self.calibrated_models = {}  # Store calibrated versions
        self.feature_explainer = FeatureExplainer()  # SHAP-like explainer
        
        # Adjust ensemble weights based on available models (Capstone: LR + RF + KNN)
        if LGBM_AVAILABLE and XGB_AVAILABLE:
            self.ensemble_weights = {'lr': 0.25, 'rf': 0.25, 'knn': 0.20, 'lgb': 0.15, 'xgb': 0.15}
        else:
            # Capstone baseline: LR + RF + KNN
            self.ensemble_weights = {'lr': 0.35, 'rf': 0.40, 'knn': 0.25, 'lgb': 0.0, 'xgb': 0.0}
        
    def train_models(self, training_data):
        """Train the ensemble of ML models."""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if X is None or len(X) < 50 or (y is not None and y.sum() < 5):  # Need minimum samples and positive cases
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Apply SMOTE to handle imbalanced data
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Train all core models (LR + RF + KNN)
            self.logistic_model.fit(X_train_balanced, y_train_balanced)
            self.rf_model.fit(X_train_balanced, y_train_balanced)
            self.knn_model.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate models
            self._evaluate_models(X_test_scaled, y_test)
            
            # Store feature importance
            self._calculate_feature_importance()
            
            self.is_trained_flag = True
            return True
            
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            return False
    
    def train_from_real_data(self, csv_path, fraud_column='fraud_reported'):
        """Train enhanced ensemble models with cross-validation and calibration."""
        try:
            # Load dataset
            data = pd.read_csv(csv_path)
            
            # Ensure fraud column exists and is binary
            if fraud_column not in data.columns:
                print(f"Warning: {fraud_column} not found in dataset")
                return False
            
            # Convert fraud_reported to binary (Y=1, N=0)
            if data[fraud_column].dtype == 'object':
                data[fraud_column] = (data[fraud_column].str.upper() == 'Y').astype(int)
            
            # Use the SAME _get_feature_columns method as prediction to ensure schema match
            X_features = self._get_feature_columns(data)
            if X_features.empty or len(X_features) < 50:
                print(f"Not enough features or samples: {len(X_features)}")
                return False
            
            y_labels = data[fraud_column]
            
            # Ensure we have some positive cases
            if y_labels.sum() < 5:
                print(f"Not enough positive fraud cases: {y_labels.sum()}")
                return False
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Enhanced imbalanced data handling with SMOTETomek (SMOTE + Tomek links cleaning)
            try:
                smote_tomek = SMOTETomek(random_state=42)
                X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train)
            except:
                # Fallback to regular SMOTE if SMOTETomek fails
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Store feature columns before training
            self.feature_columns = X_features.columns.tolist()
            
            # Train all available models
            print(f"Training ensemble models on {len(data)} samples ({y_labels.sum()} fraud cases)...")
            
            # Core capstone models: LR + RF + KNN
            self.logistic_model.fit(X_train_balanced, y_train_balanced)
            self.rf_model.fit(X_train_balanced, y_train_balanced)
            self.knn_model.fit(X_train_balanced, y_train_balanced)
            
            # Advanced models if available
            if self.lgb_model is not None:
                self.lgb_model.fit(X_train_balanced, y_train_balanced)
            
            if self.xgb_model is not None:
                self.xgb_model.fit(X_train_balanced, y_train_balanced)
            
            # Apply probability calibration for better confidence scores
            print("Applying probability calibration...")
            self.calibrated_models = {
                'lr': CalibratedClassifierCV(self.logistic_model, cv=3, method='isotonic'),
                'rf': CalibratedClassifierCV(self.rf_model, cv=3, method='isotonic'),
                'knn': CalibratedClassifierCV(self.knn_model, cv=3, method='isotonic')
            }
            
            if self.lgb_model is not None:
                self.calibrated_models['lgb'] = CalibratedClassifierCV(self.lgb_model, cv=3, method='isotonic')
            
            if self.xgb_model is not None:
                self.calibrated_models['xgb'] = CalibratedClassifierCV(self.xgb_model, cv=3, method='isotonic')
            
            # Fit calibrated models on non-SMOTE data for better calibration
            for model_name, cal_model in self.calibrated_models.items():
                try:
                    cal_model.fit(X_train_scaled, y_train)
                except:
                    # If calibration fails, use uncalibrated model
                    pass
            
            # Perform cross-validation for robust evaluation
            print("Running cross-validation...")
            cv_scores = self._cross_validate_models(X_train_scaled, y_train)
            
            # Evaluate on test set
            self._evaluate_enhanced_models(X_test_scaled, y_test)
            
            # Store CV scores
            self.model_metrics['cv_scores'] = cv_scores
            
            # Store feature importance from tree-based models
            self._calculate_feature_importance()
            
            # Fit feature explainer for SHAP-like explanations
            print("Fitting feature explainer...")
            self.feature_explainer.fit(self, X_train)
            
            self.is_trained_flag = True
            print(f"✓ Enhanced training complete!")
            print(f"  Ensemble Accuracy: {self.model_metrics.get('accuracy', 0):.1%}")
            print(f"  Precision: {self.model_metrics.get('precision', 0):.1%}")
            print(f"  Recall: {self.model_metrics.get('recall', 0):.1%}")
            print(f"  F1-Score: {self.model_metrics.get('f1', 0):.1%}")
            print(f"  PR-AUC: {self.model_metrics.get('pr_auc', 0):.3f}")
            return True
            
        except Exception as e:
            print(f"Training from real data failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_batch(self, data):
        """Make predictions using enhanced ensemble with calibrated models."""
        if not self.is_trained_flag:
            return np.full(len(data), 0.5)
        
        try:
            X = self._prepare_prediction_data(data)
            if X is None or len(X) == 0:
                return np.full(len(data), 0.5)
            
            X_scaled = self.scaler.transform(X)
            
            # Get calibrated probabilities from available models
            model_probs = {}
            
            # Logistic Regression (always available)
            if 'lr' in self.calibrated_models:
                try:
                    lr_probs = self.calibrated_models['lr'].predict_proba(X_scaled)
                except:
                    lr_probs = self.logistic_model.predict_proba(X_scaled)
            else:
                lr_probs = self.logistic_model.predict_proba(X_scaled)
            model_probs['lr'] = lr_probs[:, 1] if lr_probs.shape[1] > 1 else lr_probs[:, 0]
            
            # Random Forest (always available)
            if 'rf' in self.calibrated_models:
                try:
                    rf_probs = self.calibrated_models['rf'].predict_proba(X_scaled)
                except:
                    rf_probs = self.rf_model.predict_proba(X_scaled)
            else:
                rf_probs = self.rf_model.predict_proba(X_scaled)
            model_probs['rf'] = rf_probs[:, 1] if rf_probs.shape[1] > 1 else rf_probs[:, 0]
            
            # K-Nearest Neighbors (core capstone model)
            if 'knn' in self.calibrated_models:
                try:
                    knn_probs = self.calibrated_models['knn'].predict_proba(X_scaled)
                except:
                    knn_probs = self.knn_model.predict_proba(X_scaled)
            else:
                knn_probs = self.knn_model.predict_proba(X_scaled)
            model_probs['knn'] = knn_probs[:, 1] if knn_probs.shape[1] > 1 else knn_probs[:, 0]
            
            # LightGBM (if available)
            if self.lgb_model is not None:
                if 'lgb' in self.calibrated_models:
                    try:
                        lgb_probs = self.calibrated_models['lgb'].predict_proba(X_scaled)
                    except:
                        lgb_probs = self.lgb_model.predict_proba(X_scaled)
                else:
                    lgb_probs = self.lgb_model.predict_proba(X_scaled)
                model_probs['lgb'] = lgb_probs[:, 1] if lgb_probs.shape[1] > 1 else lgb_probs[:, 0]
            
            # XGBoost (if available)
            if self.xgb_model is not None:
                if 'xgb' in self.calibrated_models:
                    try:
                        xgb_probs = self.calibrated_models['xgb'].predict_proba(X_scaled)
                    except:
                        xgb_probs = self.xgb_model.predict_proba(X_scaled)
                else:
                    xgb_probs = self.xgb_model.predict_proba(X_scaled)
                model_probs['xgb'] = xgb_probs[:, 1] if xgb_probs.shape[1] > 1 else xgb_probs[:, 0]
            
            # Weighted ensemble prediction (including KNN as per capstone)
            ensemble_probs = (
                self.ensemble_weights['lr'] * model_probs.get('lr', 0) +
                self.ensemble_weights['rf'] * model_probs.get('rf', 0) +
                self.ensemble_weights['knn'] * model_probs.get('knn', 0) +
                self.ensemble_weights['lgb'] * model_probs.get('lgb', 0) +
                self.ensemble_weights['xgb'] * model_probs.get('xgb', 0)
            )
            
            return ensemble_probs
            
        except Exception as e:
            print(f"Batch prediction failed: {str(e)}")
            return np.full(len(data), 0.5)
    
    def predict_single(self, data):
        """Make prediction on a single claim using enhanced ensemble."""
        if not self.is_trained_flag:
            return 0.5
        
        try:
            # Convert single claim to DataFrame if needed
            if isinstance(data, pd.Series):
                data = data.to_frame().T
            elif isinstance(data, dict):
                data = pd.DataFrame([data])
            
            X = self._prepare_prediction_data(data)
            if X is None or len(X) == 0:
                return 0.5
            
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models (calibrated if available)
            model_probs = {}
            
            # Logistic Regression (always available)
            if 'lr' in self.calibrated_models:
                try:
                    lr_proba = self.calibrated_models['lr'].predict_proba(X_scaled)
                except:
                    lr_proba = self.logistic_model.predict_proba(X_scaled)
            else:
                lr_proba = self.logistic_model.predict_proba(X_scaled)
            model_probs['lr'] = lr_proba[0, 1] if lr_proba.shape[1] > 1 else lr_proba[0, 0]
            
            # Random Forest (always available)
            if 'rf' in self.calibrated_models:
                try:
                    rf_proba = self.calibrated_models['rf'].predict_proba(X_scaled)
                except:
                    rf_proba = self.rf_model.predict_proba(X_scaled)
            else:
                rf_proba = self.rf_model.predict_proba(X_scaled)
            model_probs['rf'] = rf_proba[0, 1] if rf_proba.shape[1] > 1 else rf_proba[0, 0]
            
            # K-Nearest Neighbors (core capstone model)
            if 'knn' in self.calibrated_models:
                try:
                    knn_proba = self.calibrated_models['knn'].predict_proba(X_scaled)
                except:
                    knn_proba = self.knn_model.predict_proba(X_scaled)
            else:
                knn_proba = self.knn_model.predict_proba(X_scaled)
            model_probs['knn'] = knn_proba[0, 1] if knn_proba.shape[1] > 1 else knn_proba[0, 0]
            
            # LightGBM (if available)
            if self.lgb_model is not None:
                if 'lgb' in self.calibrated_models:
                    try:
                        lgb_proba = self.calibrated_models['lgb'].predict_proba(X_scaled)
                    except:
                        lgb_proba = self.lgb_model.predict_proba(X_scaled)
                else:
                    lgb_proba = self.lgb_model.predict_proba(X_scaled)
                model_probs['lgb'] = lgb_proba[0, 1] if lgb_proba.shape[1] > 1 else lgb_proba[0, 0]
            
            # XGBoost (if available)
            if self.xgb_model is not None:
                if 'xgb' in self.calibrated_models:
                    try:
                        xgb_proba = self.calibrated_models['xgb'].predict_proba(X_scaled)
                    except:
                        xgb_proba = self.xgb_model.predict_proba(X_scaled)
                else:
                    xgb_proba = self.xgb_model.predict_proba(X_scaled)
                model_probs['xgb'] = xgb_proba[0, 1] if xgb_proba.shape[1] > 1 else xgb_proba[0, 0]
            
            # Weighted ensemble prediction (including KNN as per capstone)
            ensemble_prob = (
                self.ensemble_weights['lr'] * model_probs.get('lr', 0) +
                self.ensemble_weights['rf'] * model_probs.get('rf', 0) +
                self.ensemble_weights['knn'] * model_probs.get('knn', 0) +
                self.ensemble_weights['lgb'] * model_probs.get('lgb', 0) +
                self.ensemble_weights['xgb'] * model_probs.get('xgb', 0)
            )
            
            return float(ensemble_prob)
            
        except Exception as e:
            print(f"Single prediction failed: {str(e)}")
            return 0.5
    
    def _prepare_training_data(self, data):
        """Prepare data for training."""
        # Create synthetic labels based on risk scores for training
        # In a real system, you would use actual fraud labels
        X_features = self._get_feature_columns(data)
        
        if X_features.empty:
            return None, None
        
        # Create labels based on risk scores and triggered rules
        # High risk (>70) and multiple rules triggered = likely fraud
        y_labels = (
            (data['risk_score'] > 70) & 
            (data['triggered_rules'].str.count(',') >= 2)
        ).astype(int)
        
        # Ensure we have some positive cases
        if y_labels.sum() == 0:
            # If no high-risk cases, create some based on highest scores
            high_score_threshold = data['risk_score'].quantile(0.9)
            y_labels = (data['risk_score'] > high_score_threshold).astype(int)
        
        self.feature_columns = X_features.columns.tolist()
        return X_features, y_labels
    
    def _prepare_prediction_data(self, data):
        """Prepare data for prediction."""
        X_features = self._get_feature_columns(data)
        
        if X_features.empty:
            return None
        
        # Ensure we have the same features as training
        missing_features = set(self.feature_columns) - set(X_features.columns)
        for feature in missing_features:
            X_features[feature] = 0  # Fill missing features with 0
        
        # Reorder columns to match training
        X_features = X_features[self.feature_columns]
        
        return X_features
    
    def _get_feature_columns(self, data):
        """Extract relevant features for ML models."""
        numeric_features = []
        
        # Core features
        core_features = ['total_claim_amount', 'incident_hour_of_the_day', 'log_claim_amount']
        for feature in core_features:
            if feature in data.columns:
                numeric_features.append(feature)
        
        # Additional features
        additional_features = [
            'age', 'unusual_hour', 'is_round_amount', 'young_driver',
            'severity_numeric', 'high_risk_state', 'benford_anomaly_score',
            'amount_per_age', 'high_amount_no_witnesses', 'witnesses', 'first_digit'
        ]
        
        for feature in additional_features:
            if feature in data.columns:
                numeric_features.append(feature)
        
        # Select only numeric features that exist
        available_features = [f for f in numeric_features if f in data.columns]
        
        if not available_features:
            return pd.DataFrame()
        
        return data[available_features].select_dtypes(include=[np.number]).fillna(0)
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate model performance with KNN included."""
        try:
            # Logistic Regression predictions
            lr_pred = self.logistic_model.predict(X_test)
            lr_proba_matrix = self.logistic_model.predict_proba(X_test)
            lr_probs = lr_proba_matrix[:, 1] if lr_proba_matrix.shape[1] > 1 else lr_proba_matrix[:, 0]
            
            # Random Forest predictions
            rf_pred = self.rf_model.predict(X_test)
            rf_proba_matrix = self.rf_model.predict_proba(X_test)
            rf_probs = rf_proba_matrix[:, 1] if rf_proba_matrix.shape[1] > 1 else rf_proba_matrix[:, 0]
            
            # K-Nearest Neighbors predictions
            knn_pred = self.knn_model.predict(X_test)
            knn_proba_matrix = self.knn_model.predict_proba(X_test)
            knn_probs = knn_proba_matrix[:, 1] if knn_proba_matrix.shape[1] > 1 else knn_proba_matrix[:, 0]
            
            # Ensemble predictions using configured weights (LR=35%, RF=40%, KNN=25%)
            ensemble_probs = (
                self.ensemble_weights['lr'] * lr_probs +
                self.ensemble_weights['rf'] * rf_probs +
                self.ensemble_weights['knn'] * knn_probs
            )
            ensemble_pred = (ensemble_probs > 0.5).astype(int)
            
            # Calculate metrics for ensemble
            self.model_metrics = {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred, zero_division='warn'),
                'recall': recall_score(y_test, ensemble_pred, zero_division='warn'),
                'f1': f1_score(y_test, ensemble_pred, zero_division='warn'),
                'auc': roc_auc_score(y_test, ensemble_probs) if len(np.unique(y_test)) > 1 else 0.5,
                'n_samples': len(y_test)
            }
            
        except Exception as e:
            print(f"Model evaluation failed: {str(e)}")
            self.model_metrics = {
                'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5,
                'f1': 0.5, 'auc': 0.5, 'n_samples': 0
            }
    
    def _cross_validate_models(self, X, y):
        """Perform cross-validation for robust performance estimation."""
        cv_scores = {}
        try:
            # Use stratified k-fold for imbalanced data
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Cross-validate each model (core capstone models + advanced)
            cv_scores['lr_f1'] = cross_val_score(self.logistic_model, X, y, cv=skf, scoring='f1').mean()
            cv_scores['rf_f1'] = cross_val_score(self.rf_model, X, y, cv=skf, scoring='f1').mean()
            cv_scores['knn_f1'] = cross_val_score(self.knn_model, X, y, cv=skf, scoring='f1').mean()
            
            if self.lgb_model is not None:
                cv_scores['lgb_f1'] = cross_val_score(self.lgb_model, X, y, cv=skf, scoring='f1').mean()
            
            if self.xgb_model is not None:
                cv_scores['xgb_f1'] = cross_val_score(self.xgb_model, X, y, cv=skf, scoring='f1').mean()
            
            print(f"  CV F1-Scores: LR={cv_scores.get('lr_f1', 0):.3f}, RF={cv_scores.get('rf_f1', 0):.3f}, "
                  f"KNN={cv_scores.get('knn_f1', 0):.3f}, LGB={cv_scores.get('lgb_f1', 0):.3f}, XGB={cv_scores.get('xgb_f1', 0):.3f}")
        except Exception as e:
            print(f"Cross-validation failed: {str(e)}")
        
        return cv_scores
    
    def _evaluate_enhanced_models(self, X_test, y_test):
        """Enhanced evaluation with comprehensive metrics."""
        try:
            # Get predictions from all models
            model_preds = {}
            model_probs = {}
            
            # Logistic Regression
            lr_pred = self.logistic_model.predict(X_test)
            lr_proba_matrix = self.logistic_model.predict_proba(X_test)
            model_preds['lr'] = lr_pred
            model_probs['lr'] = lr_proba_matrix[:, 1] if lr_proba_matrix.shape[1] > 1 else lr_proba_matrix[:, 0]
            
            # Random Forest
            rf_pred = self.rf_model.predict(X_test)
            rf_proba_matrix = self.rf_model.predict_proba(X_test)
            model_preds['rf'] = rf_pred
            model_probs['rf'] = rf_proba_matrix[:, 1] if rf_proba_matrix.shape[1] > 1 else rf_proba_matrix[:, 0]
            
            # K-Nearest Neighbors
            knn_pred = self.knn_model.predict(X_test)
            knn_proba_matrix = self.knn_model.predict_proba(X_test)
            model_preds['knn'] = knn_pred
            model_probs['knn'] = knn_proba_matrix[:, 1] if knn_proba_matrix.shape[1] > 1 else knn_proba_matrix[:, 0]
            
            # LightGBM (if available)
            if self.lgb_model is not None:
                lgb_pred = self.lgb_model.predict(X_test)
                lgb_proba_matrix = self.lgb_model.predict_proba(X_test)
                model_preds['lgb'] = lgb_pred
                model_probs['lgb'] = lgb_proba_matrix[:, 1] if lgb_proba_matrix.shape[1] > 1 else lgb_proba_matrix[:, 0]
            
            # XGBoost (if available)
            if self.xgb_model is not None:
                xgb_pred = self.xgb_model.predict(X_test)
                xgb_proba_matrix = self.xgb_model.predict_proba(X_test)
                model_preds['xgb'] = xgb_pred
                model_probs['xgb'] = xgb_proba_matrix[:, 1] if xgb_proba_matrix.shape[1] > 1 else xgb_proba_matrix[:, 0]
            
            # Ensemble predictions with optimized weights (including KNN)
            ensemble_probs = (
                self.ensemble_weights['lr'] * model_probs.get('lr', 0) +
                self.ensemble_weights['rf'] * model_probs.get('rf', 0) +
                self.ensemble_weights['knn'] * model_probs.get('knn', 0) +
                self.ensemble_weights['lgb'] * model_probs.get('lgb', 0) +
                self.ensemble_weights['xgb'] * model_probs.get('xgb', 0)
            )
            ensemble_pred = (ensemble_probs > 0.5).astype(int)
            
            # Calculate comprehensive metrics
            self.model_metrics = {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'precision': precision_score(y_test, ensemble_pred, zero_division='warn'),
                'recall': recall_score(y_test, ensemble_pred, zero_division='warn'),
                'f1': f1_score(y_test, ensemble_pred, zero_division='warn'),
                'roc_auc': roc_auc_score(y_test, ensemble_probs) if len(np.unique(y_test)) > 1 else 0.5,
                'pr_auc': average_precision_score(y_test, ensemble_probs) if len(np.unique(y_test)) > 1 else 0.5,
                'n_samples': len(y_test),
                'n_fraud': int(y_test.sum()),
                'n_legitimate': int((1 - y_test).sum())
            }
            
            # Store individual model performance for comparison
            self.model_metrics['individual'] = {
                'lr_f1': f1_score(y_test, model_preds.get('lr'), zero_division='warn') if 'lr' in model_preds else 0,
                'rf_f1': f1_score(y_test, model_preds.get('rf'), zero_division='warn') if 'rf' in model_preds else 0,
                'knn_f1': f1_score(y_test, model_preds.get('knn'), zero_division='warn') if 'knn' in model_preds else 0,
                'lgb_f1': f1_score(y_test, model_preds.get('lgb'), zero_division='warn') if 'lgb' in model_preds else 0,
                'xgb_f1': f1_score(y_test, model_preds.get('xgb'), zero_division='warn') if 'xgb' in model_preds else 0
            }
            
        except Exception as e:
            print(f"Enhanced evaluation failed: {str(e)}")
            self.model_metrics = {
                'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5,
                'f1': 0.5, 'roc_auc': 0.5, 'pr_auc': 0.5, 'n_samples': 0
            }
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance from ensemble models."""
        try:
            if len(self.feature_columns) > 0:
                # Combine feature importance from tree-based models (RF, LGB, XGB)
                importances = []
                
                if hasattr(self.rf_model, 'feature_importances_'):
                    importances.append(self.rf_model.feature_importances_)
                
                if self.lgb_model is not None and hasattr(self.lgb_model, 'feature_importances_'):
                    importances.append(self.lgb_model.feature_importances_)
                
                if self.xgb_model is not None and hasattr(self.xgb_model, 'feature_importances_'):
                    importances.append(self.xgb_model.feature_importances_)
                
                if importances:
                    # Average importance across models
                    avg_importance = np.mean(importances, axis=0)
                    
                    importance_data = {
                        'feature': self.feature_columns,
                        'importance': avg_importance
                    }
                    self.feature_importance = pd.DataFrame(importance_data)
                    self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
            
        except Exception as e:
            print(f"Feature importance calculation failed: {str(e)}")
            self.feature_importance = pd.DataFrame()
    
    def is_trained(self):
        """Check if models are trained."""
        return self.is_trained_flag
    
    def get_model_accuracy(self):
        """Get model accuracy."""
        return self.model_metrics.get('accuracy', 0.5)
    
    def get_model_metrics(self):
        """Get all model metrics."""
        return self.model_metrics
    
    def get_feature_importance(self):
        """Get feature importance DataFrame."""
        return self.feature_importance
    
    def save_models(self, filepath):
        """Save trained models to file."""
        if self.is_trained_flag:
            model_data = {
                'logistic_model': self.logistic_model,
                'rf_model': self.rf_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_metrics': self.model_metrics,
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_data, filepath)
    
    def load_models(self, filepath):
        """Load trained models from file."""
        try:
            model_data = joblib.load(filepath)
            self.logistic_model = model_data['logistic_model']
            self.rf_model = model_data['rf_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data['model_metrics']
            self.feature_importance = model_data['feature_importance']
            self.is_trained_flag = True
            return True
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            return False
    
    def explain_prediction(self, claim_data: dict, fraud_probability: float) -> dict:
        """
        Generate SHAP-like explanation for a single prediction.
        
        Args:
            claim_data: Dictionary with claim features
            fraud_probability: Predicted fraud probability (0-1)
        
        Returns:
            Dictionary with feature contributions and explanations
        """
        if not self.is_trained_flag:
            return {
                'prediction': fraud_probability,
                'baseline': 0.5,
                'contributions': {},
                'top_positive': [],
                'top_negative': [],
                'explanation_text': 'Model not trained yet - unable to generate explanation.'
            }
        
        try:
            explanation = self.feature_explainer.explain_prediction(
                claim_data, fraud_probability
            )
            
            # Add human-readable text explanation
            claim_id = claim_data.get('claim_id', 'Unknown')
            explanation['explanation_text'] = self.feature_explainer.generate_explanation_text(
                explanation, claim_id
            )
            
            return explanation
        except Exception as e:
            print(f"Explanation generation failed: {str(e)}")
            return {
                'prediction': fraud_probability,
                'baseline': 0.5,
                'contributions': {},
                'top_positive': [],
                'top_negative': [],
                'explanation_text': f'Unable to generate explanation: {str(e)}'
            }
    
    def get_global_feature_importance(self, top_n: int = 10) -> list:
        """
        Get global feature importance rankings from the explainer.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of dicts with feature names and importance scores
        """
        if not self.is_trained_flag:
            return []
        
        try:
            return self.feature_explainer.get_global_importance(top_n)
        except Exception as e:
            print(f"Global importance retrieval failed: {str(e)}")
            return []
