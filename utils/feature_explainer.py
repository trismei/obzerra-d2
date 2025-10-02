"""
Custom SHAP-like feature importance explainer for fraud detection.
Provides local and global feature importance explanations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class FeatureExplainer:
    """
    Custom feature importance explainer that mimics SHAP functionality.
    Computes feature contributions to individual predictions.
    """
    
    def __init__(self):
        """Initialize the feature explainer."""
        self.feature_names = []
        self.global_importance = {}
        self.baseline_value = 0.0
        
    def fit(self, ml_manager, X_train):
        """
        Extract feature importance from trained models.
        
        Args:
            ml_manager: Trained MLModelManager instance
            X_train: Training features DataFrame
        """
        self.feature_names = X_train.columns.tolist()
        self.baseline_value = 0.5  # Neutral fraud probability
        
        # Extract feature importance from tree-based models
        importances = []
        
        # Random Forest importance
        if hasattr(ml_manager.rf_model, 'feature_importances_'):
            importances.append(ml_manager.rf_model.feature_importances_)
        
        # LightGBM importance (if available)
        if ml_manager.lgb_model is not None and hasattr(ml_manager.lgb_model, 'feature_importances_'):
            importances.append(ml_manager.lgb_model.feature_importances_)
        
        # XGBoost importance (if available)
        if ml_manager.xgb_model is not None and hasattr(ml_manager.xgb_model, 'feature_importances_'):
            importances.append(ml_manager.xgb_model.feature_importances_)
        
        # Average feature importance across models
        if importances:
            avg_importance = np.mean(importances, axis=0)
            self.global_importance = dict(zip(self.feature_names, avg_importance))
        else:
            # Fallback to uniform importance
            self.global_importance = {name: 1.0/len(self.feature_names) 
                                     for name in self.feature_names}
    
    def explain_prediction(self, claim_data: Dict, prediction_prob: float) -> Dict:
        """
        Explain a single prediction by computing feature contributions.
        
        Args:
            claim_data: Dict with claim features
            prediction_prob: Model's fraud probability prediction
        
        Returns:
            Dict with feature contributions and explanations
        """
        # Calculate deviation from baseline
        prediction_deviation = prediction_prob - self.baseline_value
        
        # Compute feature contributions based on values and global importance
        contributions = {}
        
        for feature in self.feature_names:
            if feature not in claim_data:
                continue
            
            value = claim_data[feature]
            importance = self.global_importance.get(feature, 0.0)
            
            # Simple contribution: importance * normalized_value * deviation
            # This approximates how much each feature contributed to moving
            # the prediction away from baseline
            if isinstance(value, (int, float)):
                # Normalize numeric values to [-1, 1] range
                normalized_value = self._normalize_value(feature, value)
                contribution = importance * normalized_value * prediction_deviation
            else:
                # For categorical features, use importance directly
                contribution = importance * prediction_deviation
            
            contributions[feature] = contribution
        
        # Sort by absolute contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'prediction': prediction_prob,
            'baseline': self.baseline_value,
            'contributions': dict(sorted_contributions),
            'top_positive': self._get_top_contributors(sorted_contributions, positive=True),
            'top_negative': self._get_top_contributors(sorted_contributions, positive=False)
        }
    
    def _normalize_value(self, feature: str, value: float) -> float:
        """
        Normalize a feature value to [-1, 1] range.
        Uses simple heuristics based on feature name.
        """
        # Age: typical range 18-80
        if 'age' in feature.lower():
            return (value - 49) / 31  # Center at 49, range ±31
        
        # Amount: typical range 0-500000
        elif 'amount' in feature.lower():
            return (value - 250000) / 250000
        
        # Hour: 0-23
        elif 'hour' in feature.lower():
            return (value - 11.5) / 11.5
        
        # Frequency: 0-10
        elif 'frequency' in feature.lower() or 'count' in feature.lower():
            return (value - 2) / 3
        
        # Default: assume centered at 0 with std of 1
        else:
            return np.clip(value / 100, -1, 1)
    
    def _get_top_contributors(self, sorted_contributions: List[Tuple], 
                              positive: bool, n: int = 3) -> List[Dict]:
        """
        Get top contributing features (positive or negative).
        
        Args:
            sorted_contributions: List of (feature, contribution) tuples
            positive: If True, get positive contributors; else negative
            n: Number of top contributors to return
        """
        if positive:
            contributors = [(f, c) for f, c in sorted_contributions if c > 0][:n]
        else:
            contributors = [(f, c) for f, c in sorted_contributions if c < 0][:n]
        
        return [
            {
                'feature': self._format_feature_name(feature),
                'contribution': contribution,
                'direction': 'increases' if contribution > 0 else 'decreases'
            }
            for feature, contribution in contributors
        ]
    
    def _format_feature_name(self, feature: str) -> str:
        """Convert feature name to human-readable format."""
        # Remove underscores and capitalize words
        words = feature.replace('_', ' ').split()
        return ' '.join(word.capitalize() for word in words)
    
    def get_global_importance(self, top_n: int = 10) -> List[Dict]:
        """
        Get global feature importance rankings.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of dicts with feature names and importance scores
        """
        sorted_importance = sorted(
            self.global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {
                'feature': self._format_feature_name(feature),
                'importance': importance
            }
            for feature, importance in sorted_importance
        ]
    
    def generate_explanation_text(self, explanation: Dict, claim_id: str) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            explanation: Explanation dict from explain_prediction
            claim_id: Claim identifier
        
        Returns:
            Formatted explanation string
        """
        pred = explanation['prediction']
        baseline = explanation['baseline']
        
        # Determine prediction direction
        if pred > baseline + 0.2:
            direction = "significantly higher than"
            risk = "elevated fraud risk"
        elif pred > baseline:
            direction = "slightly higher than"
            risk = "moderate fraud risk"
        elif pred < baseline - 0.2:
            direction = "significantly lower than"
            risk = "low fraud risk"
        else:
            direction = "near"
            risk = "neutral risk"
        
        text = f"Claim {claim_id} has a fraud probability of {pred:.1%}, "
        text += f"which is {direction} the baseline ({baseline:.1%}), "
        text += f"indicating {risk}.\n\n"
        
        # Top positive contributors
        if explanation['top_positive']:
            text += "Key factors increasing fraud likelihood:\n"
            for contrib in explanation['top_positive']:
                text += f"  • {contrib['feature']}: "
                text += f"+{abs(contrib['contribution']):.3f} impact\n"
            text += "\n"
        
        # Top negative contributors
        if explanation['top_negative']:
            text += "Key factors decreasing fraud likelihood:\n"
            for contrib in explanation['top_negative']:
                text += f"  • {contrib['feature']}: "
                text += f"-{abs(contrib['contribution']):.3f} impact\n"
        
        return text.strip()
