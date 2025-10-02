import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class NormalityTester:
    """
    Implements normality testing as specified in capstone methodology.
    Tests: Shapiro-Wilk, Kolmogorov-Smirnov (K-S), and Q-Q plots
    before applying Z-score analysis.
    """
    
    def __init__(self, alpha=0.05):
        """
        Initialize normality tester.
        
        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
        self.test_results = {}
    
    def test_normality(self, data, column_name='data'):
        """
        Perform comprehensive normality testing.
        
        Args:
            data: Array-like data to test
            column_name: Name of the column being tested
            
        Returns:
            dict: Test results including:
                - is_normal: Boolean indicating if data passes normality tests
                - shapiro_stat: Shapiro-Wilk test statistic
                - shapiro_pvalue: Shapiro-Wilk p-value
                - ks_stat: Kolmogorov-Smirnov test statistic
                - ks_pvalue: K-S test p-value
                - recommendation: Whether to use Z-score or alternative
        """
        # Clean data: remove NaN and infinite values
        clean_data = np.array(data)
        clean_data = clean_data[~np.isnan(clean_data)]
        clean_data = clean_data[~np.isinf(clean_data)]
        
        if len(clean_data) < 3:
            return {
                'is_normal': False,
                'shapiro_stat': None,
                'shapiro_pvalue': None,
                'ks_stat': None,
                'ks_pvalue': None,
                'recommendation': 'insufficient_data',
                'sample_size': len(clean_data)
            }
        
        # Check for zero or near-zero variance
        std = np.std(clean_data, ddof=1)
        if std < 1e-10:  # Essentially zero variance
            return {
                'is_normal': False,
                'shapiro_stat': None,
                'shapiro_pvalue': None,
                'ks_stat': None,
                'ks_pvalue': None,
                'recommendation': 'use_alternative_method',
                'sample_size': len(clean_data),
                'mean': float(np.mean(clean_data)),
                'std': float(std)
            }
        
        # Shapiro-Wilk Test
        # H0: Data is normally distributed
        # Reject H0 if p-value < alpha
        shapiro_stat, shapiro_pvalue = stats.shapiro(clean_data)
        shapiro_normal = shapiro_pvalue >= self.alpha
        
        # Kolmogorov-Smirnov Test
        # Compare against normal distribution with sample mean and std
        mean = np.mean(clean_data)
        ks_stat, ks_pvalue = stats.kstest(clean_data, lambda x: stats.norm.cdf(x, mean, std))
        ks_normal = ks_pvalue >= self.alpha
        
        # Overall decision: Data is normal if both tests pass
        is_normal = shapiro_normal and ks_normal
        
        # Recommendation based on results
        if is_normal:
            recommendation = 'use_zscore'
        elif shapiro_normal or ks_normal:
            recommendation = 'use_zscore_with_caution'
        else:
            recommendation = 'use_alternative_method'
        
        results = {
            'is_normal': is_normal,
            'shapiro_stat': float(shapiro_stat),
            'shapiro_pvalue': float(shapiro_pvalue),
            'shapiro_normal': shapiro_normal,
            'ks_stat': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_normal': ks_normal,
            'recommendation': recommendation,
            'sample_size': len(clean_data),
            'mean': float(mean),
            'std': float(std)
        }
        
        self.test_results[column_name] = results
        return results
    
    def generate_qq_plot(self, data, column_name='data'):
        """
        Generate Q-Q plot for visual normality assessment.
        
        Args:
            data: Array-like data to plot
            column_name: Name of the column for plot title
            
        Returns:
            plotly.graph_objects.Figure: Q-Q plot figure
        """
        # Clean data
        clean_data = np.array(data)
        clean_data = clean_data[~np.isnan(clean_data)]
        clean_data = clean_data[~np.isinf(clean_data)]
        
        if len(clean_data) < 3:
            # Return empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for Q-Q plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=f"Q-Q Plot: {column_name}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
            return fig
        
        # Calculate theoretical and sample quantiles
        sorted_data = np.sort(clean_data)
        n = len(sorted_data)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        
        # Standardize sample data
        mean = np.mean(clean_data)
        std = np.std(clean_data, ddof=1)
        standardized_data = (sorted_data - mean) / std
        
        # Create Q-Q plot
        fig = go.Figure()
        
        # Add scatter plot of quantiles
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=standardized_data,
            mode='markers',
            name='Data Points',
            marker=dict(color='#3b82f6', size=6, opacity=0.6)
        ))
        
        # Add reference line (perfect normal distribution)
        min_val = min(theoretical_quantiles.min(), standardized_data.min())
        max_val = max(theoretical_quantiles.max(), standardized_data.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#ef4444', dash='dash', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Q-Q Plot: {column_name}",
            xaxis_title="Theoretical Quantiles (Normal)",
            yaxis_title="Sample Quantiles (Standardized)",
            hovermode='closest',
            template='plotly_white',
            showlegend=True,
            width=600,
            height=500
        )
        
        return fig
    
    def get_normality_summary(self):
        """
        Get summary of all normality tests performed.
        
        Returns:
            pd.DataFrame: Summary of test results
        """
        if not self.test_results:
            return pd.DataFrame()
        
        summary_data = []
        for column, results in self.test_results.items():
            summary_data.append({
                'Column': column,
                'Sample Size': results['sample_size'],
                'Shapiro-Wilk p-value': results.get('shapiro_pvalue', None),
                'K-S p-value': results.get('ks_pvalue', None),
                'Is Normal': results['is_normal'],
                'Recommendation': results['recommendation']
            })
        
        return pd.DataFrame(summary_data)
    
    def should_use_zscore(self, data=None, column_name=None):
        """
        Determine if Z-score should be used based on normality tests.
        
        Args:
            data: Optional data to test if not already tested
            column_name: Column name for the data
            
        Returns:
            bool: True if Z-score is appropriate, False otherwise
        """
        if data is not None and column_name is not None:
            results = self.test_normality(data, column_name)
        elif column_name in self.test_results:
            results = self.test_results[column_name]
        else:
            return False
        
        return results['recommendation'] in ['use_zscore', 'use_zscore_with_caution']
    
    def get_alternative_outlier_method(self, data):
        """
        Suggest alternative outlier detection method if data is not normal.
        Uses IQR (Interquartile Range) method as alternative.
        
        Args:
            data: Pandas Series or array-like data
            
        Returns:
            dict: Contains bounds and method details (caller applies to original data)
        """
        # Convert to pandas Series if needed to preserve index
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        # Work with clean data for calculation
        clean_data = data.dropna()
        clean_data = clean_data[~np.isinf(clean_data)]
        
        if len(clean_data) < 4:
            return {
                'method': 'iqr',
                'lower_bound': None,
                'upper_bound': None,
                'q1': None,
                'q3': None,
                'iqr': None
            }
        
        # IQR method (robust to non-normal distributions)
        q1 = clean_data.quantile(0.25)
        q3 = clean_data.quantile(0.75)
        iqr = q3 - q1
        
        # Standard multiplier of 1.5 for outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return {
            'method': 'iqr',
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr)
        }
