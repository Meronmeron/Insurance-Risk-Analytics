"""
Utility functions for Exploratory Data Analysis (EDA)
AlphaCare Insurance Solutions - Risk Analytics Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, List, Tuple, Optional


def calculate_insurance_kpis(df: pd.DataFrame) -> Dict:
    """
    Calculate key insurance performance indicators
    
    Args:
        df: DataFrame with insurance data
        
    Returns:
        Dictionary of KPIs
    """
    kpis = {}
    
    # Loss Ratio
    kpis['overall_loss_ratio'] = df['TotalClaims'].sum() / df['TotalPremium'].sum()
    
    # Claim Frequency
    kpis['claim_frequency'] = (df['TotalClaims'] > 0).sum() / len(df)
    
    # Average Claim Severity
    claims_data = df[df['TotalClaims'] > 0]['TotalClaims']
    kpis['avg_claim_severity'] = claims_data.mean() if len(claims_data) > 0 else 0
    
    # Average Premium
    kpis['avg_premium'] = df['TotalPremium'].mean()
    
    # Total metrics
    kpis['total_policies'] = len(df)
    kpis['total_claims'] = (df['TotalClaims'] > 0).sum()
    kpis['total_premium'] = df['TotalPremium'].sum()
    kpis['total_claims_amount'] = df['TotalClaims'].sum()
    
    return kpis


def analyze_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    Analyze insurance metrics by a specific dimension
    
    Args:
        df: DataFrame with insurance data
        dimension: Column name to group by
        
    Returns:
        DataFrame with analysis results
    """
    if dimension not in df.columns:
        raise ValueError(f"Dimension '{dimension}' not found in DataFrame")
    
    analysis = df.groupby(dimension).agg({
        'TotalPremium': ['sum', 'mean', 'count'],
        'TotalClaims': ['sum', 'mean']
    }).round(2)
    
    # Flatten column names
    analysis.columns = ['_'.join(col).strip() for col in analysis.columns]
    analysis = analysis.reset_index()
    
    # Calculate Loss Ratio
    analysis['LossRatio'] = (analysis['TotalClaims_sum'] / 
                           analysis['TotalPremium_sum']).round(3)
    
    # Calculate Claim Frequency
    claim_freq = df.groupby(dimension).apply(
        lambda x: (x['TotalClaims'] > 0).sum() / len(x)
    ).reset_index()
    claim_freq.columns = [dimension, 'ClaimFrequency']
    
    analysis = analysis.merge(claim_freq, on=dimension)
    
    return analysis.sort_values('LossRatio', ascending=False)


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> Tuple[List, List]:
    """
    Detect outliers using IQR method
    
    Args:
        data: Pandas Series
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        Tuple of (outlier_indices, outlier_values)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = data[outlier_mask].index.tolist()
    outlier_values = data[outlier_mask].tolist()
    
    return outlier_indices, outlier_values


def create_risk_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create risk segments based on loss ratio
    
    Args:
        df: DataFrame with insurance data
        
    Returns:
        DataFrame with risk segments added
    """
    df_copy = df.copy()
    
    # Calculate loss ratio for each policy
    df_copy['LossRatio'] = df_copy['TotalClaims'] / df_copy['TotalPremium']
    df_copy['LossRatio'] = df_copy['LossRatio'].replace([np.inf, -np.inf], np.nan)
    
    # Create risk segments
    def categorize_risk(loss_ratio):
        if pd.isna(loss_ratio):
            return 'Unknown'
        elif loss_ratio == 0:
            return 'No Claims'
        elif loss_ratio < 0.5:
            return 'Low Risk'
        elif loss_ratio < 1.0:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    df_copy['RiskSegment'] = df_copy['LossRatio'].apply(categorize_risk)
    
    return df_copy


def generate_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data quality report
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with data quality metrics
    """
    report = {}
    
    # Basic info
    report['total_rows'] = len(df)
    report['total_columns'] = len(df.columns)
    report['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    # Missing values
    missing_counts = df.isnull().sum()
    report['missing_values'] = {
        'total_missing': missing_counts.sum(),
        'columns_with_missing': (missing_counts > 0).sum(),
        'missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100
    }
    
    # Data types
    report['data_types'] = df.dtypes.value_counts().to_dict()
    
    # Duplicates
    report['duplicate_rows'] = df.duplicated().sum()
    
    # Numerical columns statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    report['numerical_columns'] = len(numerical_cols)
    
    if len(numerical_cols) > 0:
        report['numerical_stats'] = {
            'zero_values': (df[numerical_cols] == 0).sum().sum(),
            'negative_values': (df[numerical_cols] < 0).sum().sum(),
            'infinite_values': np.isinf(df[numerical_cols]).sum().sum()
        }
    
    return report


def plot_distribution_analysis(df: pd.DataFrame, columns: List[str], 
                             fig_size: Tuple[int, int] = (15, 10)) -> None:
    """
    Create distribution plots for specified columns
    
    Args:
        df: DataFrame with data
        columns: List of column names to plot
        fig_size: Figure size tuple
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=fig_size)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
    
    for i, col in enumerate(columns):
        if i < len(axes):
            # Histogram
            axes[i].hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[i].legend()
    
    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def statistical_test_summary(df: pd.DataFrame, 
                           categorical_col: str, 
                           numerical_col: str) -> Dict:
    """
    Perform statistical tests between categorical and numerical variables
    
    Args:
        df: DataFrame with data
        categorical_col: Categorical variable column name
        numerical_col: Numerical variable column name
        
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Get groups
    groups = [group[numerical_col].dropna() for name, group in df.groupby(categorical_col)]
    
    if len(groups) < 2:
        return {"error": "Need at least 2 groups for comparison"}
    
    # Normality tests for each group
    results['normality_tests'] = {}
    for i, group in enumerate(groups):
        if len(group) >= 8:  # Minimum sample size for Shapiro-Wilk
            _, p_value = stats.shapiro(group[:5000])  # Limit sample size
            results['normality_tests'][f'group_{i}'] = {
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
    
    # Choose appropriate test
    if len(groups) == 2:
        # Two groups: t-test or Mann-Whitney U
        stat, p_value = stats.ttest_ind(groups[0], groups[1])
        results['two_sample_test'] = {
            'test_type': 't-test',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Non-parametric alternative
        stat_np, p_value_np = stats.mannwhitneyu(groups[0], groups[1])
        results['non_parametric_test'] = {
            'test_type': 'Mann-Whitney U',
            'statistic': stat_np,
            'p_value': p_value_np,
            'significant': p_value_np < 0.05
        }
    
    else:
        # Multiple groups: ANOVA or Kruskal-Wallis
        stat, p_value = stats.f_oneway(*groups)
        results['multiple_group_test'] = {
            'test_type': 'ANOVA',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Non-parametric alternative
        stat_np, p_value_np = stats.kruskal(*groups)
        results['non_parametric_test'] = {
            'test_type': 'Kruskal-Wallis',
            'statistic': stat_np,
            'p_value': p_value_np,
            'significant': p_value_np < 0.05
        }
    
    return results 