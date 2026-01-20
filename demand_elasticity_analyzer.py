"""
Demand Elasticity Analyzer for Aadhaar Services
===============================================

A governance-focused operational intelligence system that evaluates whether
increasing operational capacity in districts reduces anomaly frequency.

This module implements elasticity analysis at district and state levels,
providing policy recommendations based on historical capacity-response patterns.

Author: Data Analytics Team
Date: 2026
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


def identify_capacity_increase_periods(
    district_df: pd.DataFrame,
    min_sustained_months: int = 2,
    percentile_threshold: float = 75.0
) -> List[Tuple[int, int]]:
    """
    Identify periods of sustained capacity increase within a district's time series.
    
    A capacity increase period is defined as a sustained rise in total_updates
    or update_rate over at least min_sustained_months consecutive months.
    
    Args:
        district_df: DataFrame for a single district, sorted by month
        min_sustained_months: Minimum consecutive months for a trend to qualify
        percentile_threshold: Percentile threshold for significant changes (0-100)
    
    Returns:
        List of tuples (start_idx, end_idx) indicating capacity increase periods
    """
    if len(district_df) < min_sustained_months + 1:
        return []
    
    # Use total_updates as primary indicator
    updates = district_df['total_updates'].values
    
    # Calculate month-to-month changes
    changes = np.diff(updates)
    
    # Identify threshold for significant increase
    positive_changes = changes[changes > 0]
    if len(positive_changes) == 0:
        return []
    
    # Use percentile of positive changes OR minimum positive change, whichever is smaller
    percentile_val = np.percentile(positive_changes, percentile_threshold)
    min_positive = np.min(positive_changes)
    threshold = min(percentile_val, min_positive * 1.5)  # Allow for moderate variance
    
    # Find sustained increases (all positive changes count, but we track consistency)
    capacity_periods = []
    current_start = None
    sustained_count = 0
    
    for i in range(len(changes)):
        # Consider a change sustained if it's positive
        if changes[i] > 0:
            if current_start is None:
                current_start = i
            sustained_count += 1
        else:
            # Break in positive trend
            if sustained_count >= min_sustained_months:
                capacity_periods.append((current_start, i))
            current_start = None
            sustained_count = 0
    
    # Handle end of series
    if sustained_count >= min_sustained_months and current_start is not None:
        capacity_periods.append((current_start, len(changes)))
    
    return capacity_periods


def compute_anomaly_rates(
    period_df: pd.DataFrame
) -> float:
    """
    Compute anomaly frequency in a period.
    
    Args:
        period_df: DataFrame subset for a specific period
    
    Returns:
        Proportion of months with anomalies (0.0 to 1.0)
    """
    if len(period_df) == 0:
        return np.nan
    
    anomaly_count = (period_df['anomaly_flag'] == True).sum()
    return anomaly_count / len(period_df)


def compute_elasticity_metrics(
    district_df: pd.DataFrame,
    min_sustained_months: int = 2
) -> Dict:
    """
    Compute elasticity metrics for a single district.
    
    Identifies capacity increase periods, compares anomaly rates before and after,
    and calculates elasticity score and classification.
    
    Args:
        district_df: DataFrame for a single district, sorted by month
        min_sustained_months: Minimum months defining capacity increase
    
    Returns:
        Dictionary containing elasticity metrics
    """
    if len(district_df) == 0:
        return {
            'elasticity_score': np.nan,
            'elasticity_classification': 'Insufficient Data',
            'expected_intervention_effectiveness': 'Unknown',
            'policy_recommendation': 'Insufficient historical data for analysis',
            'analysis_basis': 'No data available',
            'pre_capacity_anomaly_rate': np.nan,
            'post_capacity_anomaly_rate': np.nan,
            'anomaly_rate_change': np.nan,
            'capacity_increase_periods_count': 0
        }
    
    # Identify capacity increase periods
    capacity_periods = identify_capacity_increase_periods(
        district_df,
        min_sustained_months=min_sustained_months
    )
    
    if len(capacity_periods) == 0:
        # No clear capacity increases detected
        return {
            'elasticity_score': 0.5,
            'elasticity_classification': 'Moderate',
            'expected_intervention_effectiveness': 'Limited',
            'policy_recommendation': 'Capacity patterns unclear; further monitoring needed',
            'analysis_basis': 'No sustained capacity increases detected',
            'pre_capacity_anomaly_rate': np.nan,
            'post_capacity_anomaly_rate': np.nan,
            'anomaly_rate_change': np.nan,
            'capacity_increase_periods_count': 0
        }
    
    # Analyze anomaly rates before and after capacity increases
    pre_anomaly_rates = []
    post_anomaly_rates = []
    
    for start_idx, end_idx in capacity_periods:
        # Pre-capacity period: months before the increase
        if start_idx > 0:
            pre_period = district_df.iloc[:start_idx]
            pre_rate = compute_anomaly_rates(pre_period)
            if not np.isnan(pre_rate):
                pre_anomaly_rates.append(pre_rate)
        
        # Post-capacity period: months during and after the increase
        if end_idx < len(district_df):
            # Include the capacity increase period + following months
            post_period = district_df.iloc[start_idx:min(end_idx + 3, len(district_df))]
            post_rate = compute_anomaly_rates(post_period)
            if not np.isnan(post_rate):
                post_anomaly_rates.append(post_rate)
    
    # Aggregate results across all capacity increase periods
    if len(pre_anomaly_rates) == 0 or len(post_anomaly_rates) == 0:
        return {
            'elasticity_score': 0.5,
            'elasticity_classification': 'Moderate',
            'expected_intervention_effectiveness': 'Limited',
            'policy_recommendation': 'Insufficient pre/post capacity data for comparison',
            'analysis_basis': f'{len(capacity_periods)} capacity increase periods detected',
            'pre_capacity_anomaly_rate': np.mean(pre_anomaly_rates) if pre_anomaly_rates else np.nan,
            'post_capacity_anomaly_rate': np.mean(post_anomaly_rates) if post_anomaly_rates else np.nan,
            'anomaly_rate_change': np.nan,
            'capacity_increase_periods_count': len(capacity_periods)
        }
    
    avg_pre_rate = np.mean(pre_anomaly_rates)
    avg_post_rate = np.mean(post_anomaly_rates)
    
    # Calculate percentage change (negative = improvement)
    if avg_pre_rate > 0:
        anomaly_rate_change = (avg_post_rate - avg_pre_rate) / avg_pre_rate
    else:
        anomaly_rate_change = 0.0 if avg_post_rate == 0 else 1.0
    
    # Compute elasticity score (0 = high elasticity/improvement, 1 = low elasticity/no improvement)
    # Normalized to [0, 1] range
    elasticity_score = max(0.0, min(1.0, 0.5 + anomaly_rate_change / 2.0))
    
    # Classify elasticity
    if elasticity_score <= 0.33:
        classification = 'High'
        effectiveness = 'Effective'
        recommendation = 'Capacity increases have demonstrated effectiveness; continue expansion strategy'
    elif elasticity_score < 0.67:
        classification = 'Moderate'
        effectiveness = 'Moderately Effective'
        recommendation = 'Capacity increases show mixed results; investigate root causes of persistent anomalies'
    else:
        classification = 'Low'
        effectiveness = 'Limited'
        recommendation = 'Anomalies persist despite capacity increases; investigate structural/systemic issues'
    
    return {
        'elasticity_score': elasticity_score,
        'elasticity_classification': classification,
        'expected_intervention_effectiveness': effectiveness,
        'policy_recommendation': recommendation,
        'analysis_basis': f'{len(capacity_periods)} capacity increase periods analyzed',
        'pre_capacity_anomaly_rate': avg_pre_rate,
        'post_capacity_anomaly_rate': avg_post_rate,
        'anomaly_rate_change': anomaly_rate_change,
        'capacity_increase_periods_count': len(capacity_periods)
    }


def analyze_demand_elasticity(
    df: pd.DataFrame,
    min_sustained_months: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, callable]:
    """
    Analyze demand elasticity across districts and states.
    
    Main entry point for elasticity analysis. Processes raw monthly district-level
    data and returns governance-focused outputs suitable for policy decisions.
    
    Args:
        df: Raw DataFrame with columns:
            - month (datetime or YYYY-MM)
            - state (string)
            - district (string)
            - demographic_updates (int)
            - biometric_updates (int)
            - enrolments (int)
            - anomaly_flag (boolean)
            - anomaly_persistent (boolean)
            - total_updates (int) [derived: demographic_updates + biometric_updates]
            - update_rate (float) [derived: total_updates / enrolments]
        
        min_sustained_months: Minimum consecutive months for capacity increase detection
    
    Returns:
        Tuple of:
            1. district_summary (DataFrame): One row per district with elasticity metrics
            2. state_summary (DataFrame): State-level aggregates and effectiveness metrics
            3. generate_insight (callable): Function to generate plain-language insights
    """
    
    # Validate required columns
    required_cols = ['month', 'state', 'district', 'anomaly_flag', 'total_updates']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure data is sorted
    df = df.sort_values(['state', 'district', 'month']).reset_index(drop=True)
    
    # Compute elasticity for each district
    results = []
    
    for (state, district), group in df.groupby(['state', 'district']):
        group = group.sort_values('month').reset_index(drop=True)
        metrics = compute_elasticity_metrics(group, min_sustained_months=min_sustained_months)
        
        result_row = {
            'state': state,
            'district': district,
            'elasticity_score': metrics['elasticity_score'],
            'elasticity_classification': metrics['elasticity_classification'],
            'expected_intervention_effectiveness': metrics['expected_intervention_effectiveness'],
            'policy_recommendation': metrics['policy_recommendation'],
            'pre_capacity_anomaly_rate': metrics['pre_capacity_anomaly_rate'],
            'post_capacity_anomaly_rate': metrics['post_capacity_anomaly_rate'],
            'anomaly_rate_change': metrics['anomaly_rate_change'],
            'capacity_increase_periods_count': metrics['capacity_increase_periods_count']
        }
        results.append(result_row)
    
    # Create district-level summary
    district_summary = pd.DataFrame(results)
    
    # Validate output columns
    output_cols = [
        'state', 'district', 'elasticity_score', 'elasticity_classification',
        'expected_intervention_effectiveness', 'policy_recommendation'
    ]
    district_summary = district_summary[output_cols]
    
    # Create state-level summary
    state_summary = []
    
    for state, state_group in district_summary.groupby('state'):
        total_districts = len(state_group)
        high_elasticity = (state_group['elasticity_classification'] == 'High').sum()
        moderate_elasticity = (state_group['elasticity_classification'] == 'Moderate').sum()
        low_elasticity = (state_group['elasticity_classification'] == 'Low').sum()
        
        effective_count = (state_group['expected_intervention_effectiveness'] == 'Effective').sum()
        capacity_responsive_pct = (effective_count / total_districts * 100) if total_districts > 0 else 0
        low_elasticity_pct = (low_elasticity / total_districts * 100) if total_districts > 0 else 0
        
        avg_elasticity_score = state_group['elasticity_score'].mean()
        
        state_summary.append({
            'state': state,
            'total_districts_analyzed': total_districts,
            'high_elasticity_districts': high_elasticity,
            'moderate_elasticity_districts': moderate_elasticity,
            'low_elasticity_districts': low_elasticity,
            'capacity_responsive_pct': capacity_responsive_pct,
            'low_elasticity_pct': low_elasticity_pct,
            'avg_elasticity_score': avg_elasticity_score
        })
    
    state_summary = pd.DataFrame(state_summary)
    
    # Define insight generation function
    def generate_insight(state: str, district: str) -> str:
        """
        Generate a plain-language insight for a specific district.
        
        Args:
            state: State name
            district: District name
        
        Returns:
            Plain-language insight string
        """
        row = district_summary[
            (district_summary['state'] == state) & 
            (district_summary['district'] == district)
        ]
        
        if row.empty:
            return f"No data available for {district}, {state}."
        
        row = row.iloc[0]
        classification = row['elasticity_classification']
        effectiveness = row['expected_intervention_effectiveness']
        recommendation = row['policy_recommendation']
        
        # Generate context-specific insight
        if classification == 'High':
            insight = (
                f"Historical analysis of {district} district shows strong capacity elasticity: "
                f"periods of increased operational capacity have consistently been followed by "
                f"reduced anomaly frequency. This demonstrates clear operational responsiveness. "
                f"Recommendation: {recommendation}"
            )
        elif classification == 'Moderate':
            insight = (
                f"{district} district shows mixed results when capacity is increased. "
                f"While some improvement is observed, anomalies have not been eliminated proportionally. "
                f"This suggests underlying systemic factors beyond capacity constraints. "
                f"Recommendation: {recommendation}"
            )
        else:  # Low elasticity
            insight = (
                f"Historical data shows that increasing capacity in {district} district "
                f"has not significantly reduced anomaly frequency, suggesting structural issues "
                f"beyond operational capacity. Anomalies may stem from data quality, process design, "
                f"or external factors. Recommendation: {recommendation}"
            )
        
        return insight
    
    return district_summary, state_summary, generate_insight


# ============================================================================
# Example Usage and Testing
# ============================================================================

def main():
    """
    Example usage demonstrating the elasticity analyzer.
    """
    print("Demand Elasticity Analyzer initialized.")
    print("\nUsage:")
    print("  district_summary, state_summary, generate_insight = analyze_demand_elasticity(df)")
    print("\nwhere df contains the required columns.")


if __name__ == '__main__':
    main()
