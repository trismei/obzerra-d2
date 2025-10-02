import pandas as pd
import numpy as np
from scipy import stats
import warnings
from utils.statistical_tests import NormalityTester
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

class FraudEngine:
    def __init__(self):
        # Thresholds
        self.amount_threshold_high = 150000  # High claim threshold
        self.amount_threshold_extreme = 250000  # Extreme claim threshold
        self.unusual_hours = [0, 1, 2, 3, 4, 5, 22, 23]  # Late night/early morning
        self.high_risk_hours = [1, 2, 3, 4]  # Most suspicious hours
        self.young_age_threshold = 25
        self.high_claim_young_threshold = 100000
        
        # Aggressive weights for each indicator
        self.indicator_weights = {
            'z_score_outlier': 18,
            'benford_violation': 12,
            'unusual_hour': 15,
            'very_suspicious_hour': 25,  # 1-4 AM
            'round_amount': 12,
            'high_amount': 20,
            'extreme_amount': 30,
            'young_high_claim': 28,
            'no_witnesses_high_claim': 22,
            'no_police_report': 25,
            'severity_mismatch': 20,
            'theft_no_police': 35,  # Vehicle theft without police report
            'total_loss_suspicious': 25
        }
        
        # Multipliers for dangerous combinations
        self.combo_multipliers = {
            ('young_high_claim', 'no_witnesses_high_claim'): 1.4,
            ('young_high_claim', 'unusual_hour'): 1.3,
            ('extreme_amount', 'no_police_report'): 1.35,
            ('theft_no_police', 'no_witnesses_high_claim'): 1.5,
            ('very_suspicious_hour', 'extreme_amount'): 1.3,
            ('no_police_report', 'total_loss_suspicious'): 1.4
        }
    
    def analyze_single_claim(self, claim):
        """Analyze a single claim and return fraud indicators with score."""
        indicators = []
        score = 0
        
        claim_amount = claim.get('total_claim_amount', 0)
        incident_hour = claim.get('incident_hour_of_the_day', 12)
        age = claim.get('age', 40)
        witnesses = claim.get('witnesses', 1)
        police_report = claim.get('police_report_available', 'YES')
        incident_type = claim.get('incident_type', '')
        severity = claim.get('incident_severity', '')
        
        # 1. Z-Score Outlier (using simple heuristic if no batch data)
        if claim_amount > 300000:
            indicators.append('z_score_outlier')
            score += self.indicator_weights['z_score_outlier']
        
        # 2. Benford's Law violation
        if self._check_benford_violation(claim_amount):
            indicators.append('benford_violation')
            score += self.indicator_weights['benford_violation']
        
        # 3. Unusual/Suspicious Hour
        if incident_hour in self.high_risk_hours:
            indicators.append('very_suspicious_hour')
            score += self.indicator_weights['very_suspicious_hour']
        elif incident_hour in self.unusual_hours:
            indicators.append('unusual_hour')
            score += self.indicator_weights['unusual_hour']
        
        # 4. Round Amount
        if self._is_round_amount(claim_amount):
            indicators.append('round_amount')
            score += self.indicator_weights['round_amount']
        
        # 5. High/Extreme Amount
        if claim_amount >= self.amount_threshold_extreme:
            indicators.append('extreme_amount')
            score += self.indicator_weights['extreme_amount']
        elif claim_amount >= self.amount_threshold_high:
            indicators.append('high_amount')
            score += self.indicator_weights['high_amount']
        
        # 6. Young driver with high claim
        if age < self.young_age_threshold and claim_amount >= self.high_claim_young_threshold:
            indicators.append('young_high_claim')
            score += self.indicator_weights['young_high_claim']
        
        # 7. No witnesses on high claim
        if witnesses == 0 and claim_amount >= self.amount_threshold_high:
            indicators.append('no_witnesses_high_claim')
            score += self.indicator_weights['no_witnesses_high_claim']
        
        # 8. No police report
        if police_report in ['NO', 'UNKNOWN', 'N']:
            indicators.append('no_police_report')
            score += self.indicator_weights['no_police_report']
            
            # Extra penalty for vehicle theft without police report
            if 'theft' in str(incident_type).lower():
                indicators.append('theft_no_police')
                score += self.indicator_weights['theft_no_police']
        
        # 9. Severity mismatches
        if severity in ['Total Loss', 'Major Damage']:
            indicators.append('total_loss_suspicious')
            score += self.indicator_weights['total_loss_suspicious']
            
            if witnesses == 0:
                indicators.append('severity_mismatch')
                score += self.indicator_weights['severity_mismatch']
        
        # Apply combo multipliers
        score = self._apply_combo_multipliers(indicators, score)
        
        # Cap at 100
        score = min(score, 100)
        
        return {
            'claim_id': claim.get('claim_id', 'N/A'),
            'risk_score': score,
            'top_indicators': indicators,
            'total_claim_amount': claim_amount
        }
    
    def _check_benford_violation(self, amount):
        """Check if amount violates Benford's Law."""
        if amount == 0:
            return False
        first_digit = int(str(int(amount))[0])
        # Benford's law: first digit 1 appears ~30% of time, 9 appears ~4.6%
        # Flag if first digit is 8 or 9 (less common in natural data)
        return first_digit in [8, 9]
    
    def _is_round_amount(self, amount):
        """Check if amount is suspiciously round."""
        # Check for amounts divisible by 10000, 50000, or 100000
        return (amount % 100000 == 0 or 
                amount % 50000 == 0 or 
                (amount % 10000 == 0 and amount >= 100000))
    
    def _apply_combo_multipliers(self, indicators, base_score):
        """Apply multipliers for dangerous indicator combinations."""
        multiplier = 1.0
        indicators_set = set(indicators)
        
        for (ind1, ind2), mult in self.combo_multipliers.items():
            if ind1 in indicators_set and ind2 in indicators_set:
                multiplier = max(multiplier, mult)
        
        return base_score * multiplier
    
    def combine_single_prediction(self, rule_result, ml_result):
        """Combine rule-based and ML predictions for single claim."""
        rule_score = rule_result.get('risk_score', 0)
        ml_prob = ml_result.get('fraud_probability', 0)
        
        # Weighted combination: 60% rules, 40% ML
        if ml_prob is not None and not np.isnan(ml_prob):
            combined_score = (0.6 * rule_score) + (0.4 * ml_prob * 100)
        else:
            combined_score = rule_score
        
        return {
            **rule_result,
            'ml_fraud_prob': ml_prob,
            'risk_score': combined_score
        }
    
    def combine_predictions(self, rule_results_df, ml_predictions):
        """Combine rule-based and ML predictions for batch."""
        combined_df = rule_results_df.copy()
        
        if ml_predictions and len(ml_predictions) == len(rule_results_df):
            ml_probs = [pred.get('fraud_probability', 0) for pred in ml_predictions]
            combined_df['ml_fraud_prob'] = ml_probs
            
            # Weighted combination: 60% rules, 40% ML
            combined_df['risk_score'] = (
                0.6 * combined_df['risk_score'] + 
                0.4 * combined_df['ml_fraud_prob'] * 100
            )
        
        return combined_df