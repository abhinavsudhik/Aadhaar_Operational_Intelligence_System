"""
Aadhaar Update Capacity Stress Prediction System
================================================

This module estimates the likelihood that districts will exceed their normal
Aadhaar update capacity in the near term (next 1 month).

Author: Senior Data Analyst
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# SECTION 1: HELPER FUNCTIONS
# ============================================================================

def compute_rolling_normal_capacity(
    district_data: pd.DataFrame,
    window_months: int = 12,
    percentile: float = 75.0
) -> pd.Series:
    """
    Compute the rolling normal capacity for a single district.
    
    Normal capacity is defined as the 75th percentile of total_updates
    over a rolling 12-month window.
    
    Args:
        district_data: DataFrame for a single district, sorted by month
        window_months: Size of rolling window (default: 12 months)
        percentile: Percentile threshold (default: 75)
    
    Returns:
        Series indexed by month with rolling normal capacity values
    """
    # Use rolling window to compute percentile; min_periods ensures we only
    # compute when we have at least 3 months of data to avoid instability
    rolling_capacity = district_data['total_updates'].rolling(
        window=window_months,
        min_periods=3
    ).quantile(percentile / 100.0)
    
    return rolling_capacity


def identify_stress_events(
    district_data: pd.DataFrame,
    normal_capacity: pd.Series
) -> pd.Series:
    """
    Identify historical stress events for a district.
    
    A stress event occurs when total_updates exceeds normal_capacity.
    
    Args:
        district_data: DataFrame for a single district
        normal_capacity: Series of normal capacity values
    
    Returns:
        Boolean Series indicating stress events (True = stress event)
    """
    # Stress event is True when actual updates exceed capacity
    # Handle NaN capacity values by treating them as False (insufficient history)
    stress_events = (
        (district_data['total_updates'].values > normal_capacity.values) &
        (~normal_capacity.isna().values)
    )
    
    return pd.Series(stress_events, index=district_data.index)


def compute_month_over_month_growth(
    district_data: pd.DataFrame
) -> pd.Series:
    """
    Compute month-over-month growth rate for total_updates.
    
    Growth rate = (current_month - previous_month) / previous_month
    
    Args:
        district_data: DataFrame for a single district, sorted by month
    
    Returns:
        Series of growth rates (NaN for first month)
    """
    growth_rate = district_data['total_updates'].pct_change()
    
    return growth_rate


def identify_recent_spikes(
    district_data: pd.DataFrame,
    growth_threshold: float = 0.20,
    use_anomaly_flag: bool = True
) -> pd.Series:
    """
    Identify recent spikes in total_updates.
    
    A recent spike is defined as:
    - Month-over-month growth > 20%, OR
    - Anomaly flag is True (if column exists)
    
    Args:
        district_data: DataFrame for a single district
        growth_threshold: MoM growth threshold for spike (default: 20%)
        use_anomaly_flag: Whether to consider anomaly_flag column (default: True)
    
    Returns:
        Boolean Series indicating recent spikes
    """
    # Compute month-over-month growth
    growth_rate = compute_month_over_month_growth(district_data)
    
    # Initialize spikes as high growth
    spikes = growth_rate > growth_threshold
    
    # If anomaly_flag column exists, include those as spikes too
    if use_anomaly_flag and 'anomaly_flag' in district_data.columns:
        anomaly_flag = district_data['anomaly_flag'].astype(bool)
        spikes = spikes | anomaly_flag
    
    return pd.Series(spikes, index=district_data.index)


def compute_empirical_stress_probability(
    district_data: pd.DataFrame,
    recent_spikes: pd.Series,
    stress_events: pd.Series,
    lookahead_months: int = 1
) -> float:
    """
    Estimate stress likelihood using empirical probability.
    
    This computes: (# times a recent spike was followed by a stress event 
    within lookahead_months) / (# times a recent spike occurred)
    
    Args:
        district_data: DataFrame for a single district
        recent_spikes: Boolean Series of spike indicators
        stress_events: Boolean Series of stress event indicators
        lookahead_months: Months ahead to check for stress (default: 1)
    
    Returns:
        Empirical probability (float, 0-1), or None if insufficient history
    """
    # Find indices where spikes occurred
    spike_indices = np.where(recent_spikes.values)[0]
    
    # Count spike events (minimum 2 for meaningful probability)
    spike_count = len(spike_indices)
    if spike_count < 2:
        return None
    
    # Check if each spike was followed by a stress event within lookahead_months
    stress_following_spike_count = 0
    
    for spike_idx in spike_indices:
        # Define lookahead range
        lookahead_start = spike_idx + 1
        lookahead_end = min(spike_idx + lookahead_months + 1, len(stress_events))
        
        # Check if any stress event in lookahead window
        if lookahead_end > lookahead_start:
            if stress_events.iloc[lookahead_start:lookahead_end].any():
                stress_following_spike_count += 1
    
    # Empirical probability
    empirical_prob = stress_following_spike_count / spike_count
    
    return empirical_prob


def compute_heuristic_risk_score(
    district_data: pd.DataFrame,
    recent_months: int = 6
) -> float:
    """
    Compute a heuristic risk score for fallback when insufficient history exists.
    
    Risk score combines:
    - Recent growth rate (weight: 0.4)
    - Update volatility over last 6 months (weight: 0.4)
    - Anomaly flag presence (weight: 0.2)
    
    Score is normalized to 0-1 range.
    
    Args:
        district_data: DataFrame for a single district
        recent_months: Number of recent months for volatility (default: 6)
    
    Returns:
        Risk score (float, 0-1)
    """
    scores = []
    weights = []
    
    # Component 1: Recent growth rate (last month)
    # Normalized: growth > 20% -> high risk
    if len(district_data) > 1:
        last_growth = district_data['total_updates'].pct_change().iloc[-1]
        if pd.notna(last_growth):
            # Map growth rate to 0-1 score: 50% growth = 1.0, 0% = 0.0, negative = 0.0
            growth_score = min(last_growth / 0.50, 1.0) if last_growth > 0 else 0.0
            scores.append(growth_score)
            weights.append(0.4)
    
    # Component 2: Update volatility (std of last 6 months)
    # Higher volatility = higher risk
    recent_updates = district_data['total_updates'].tail(recent_months)
    if len(recent_updates) > 1:
        volatility = recent_updates.std()
        mean_updates = recent_updates.mean()
        # Coefficient of variation (normalized std)
        if mean_updates > 0:
            cv = volatility / mean_updates
            # Map CV to 0-1 score: CV > 0.5 -> 1.0
            volatility_score = min(cv / 0.5, 1.0)
        else:
            volatility_score = 0.0
        scores.append(volatility_score)
        weights.append(0.4)
    
    # Component 3: Anomaly flag presence
    # If any recent records flagged as anomalous, increase risk
    if 'anomaly_flag' in district_data.columns:
        recent_anomalies = district_data['anomaly_flag'].tail(recent_months).sum()
        anomaly_score = min(recent_anomalies / recent_months, 1.0)
        scores.append(anomaly_score)
        weights.append(0.2)
    
    # Weighted average of scores
    if scores:
        heuristic_score = np.average(scores, weights=weights[:len(scores)])
    else:
        heuristic_score = 0.0
    
    return np.clip(heuristic_score, 0.0, 1.0)


def predict_next_month_updates(
    district_data: pd.DataFrame,
    method: str = 'mean'
) -> float:
    """
    Predict total_updates for the next month.
    
    Methods:
    - 'mean': Simple average of last 3 months
    - 'trend': Linear extrapolation of last 6 months
    
    Args:
        district_data: DataFrame for a single district
        method: Prediction method (default: 'mean')
    
    Returns:
        Predicted total_updates for next month
    """
    if method == 'mean':
        # Simple 3-month average
        recent = district_data['total_updates'].tail(3)
        prediction = recent.mean() if len(recent) > 0 else 0.0
    
    elif method == 'trend':
        # Linear trend extrapolation from last 6 months
        recent = district_data['total_updates'].tail(6)
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = recent.values
            # Fit linear trend
            coeffs = np.polyfit(x, y, 1)
            # Predict next point
            prediction = coeffs[0] * len(recent) + coeffs[1]
            prediction = max(prediction, 0.0)  # Ensure non-negative
        else:
            prediction = recent.iloc[-1] if len(recent) > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return prediction


def map_likelihood_to_risk_level(stress_likelihood: float) -> str:
    """
    Map stress likelihood probability to interpretable risk level.
    
    Args:
        stress_likelihood: Probability between 0 and 1
    
    Returns:
        Risk level: 'Low', 'Medium', or 'High'
    """
    if stress_likelihood < 0.33:
        return 'Low'
    elif stress_likelihood < 0.67:
        return 'Medium'
    else:
        return 'High'


# ============================================================================
# SECTION 2: MAIN ESTIMATION FUNCTION
# ============================================================================

def estimate_district_stress_likelihood(
    df: pd.DataFrame,
    reference_month: Optional[pd.Timestamp] = None,
    growth_threshold: float = 0.20,
    window_months: int = 12,
    percentile: float = 75.0,
    min_spike_history: int = 2,
    lookahead_months: int = 1
) -> pd.DataFrame:
    """
    Estimate stress likelihood for all districts using empirical probability
    and heuristic fallback.
    
    This function processes each district independently to compute:
    - Normal capacity (75th percentile of 12-month rolling window)
    - Historical stress events (updates > normal capacity)
    - Recent spikes (MoM growth > 20% or anomaly flag)
    - Stress likelihood (empirical probability or heuristic fallback)
    - Risk level (Low / Medium / High)
    
    Args:
        df: Aggregated Aadhaar data DataFrame with columns:
            - month (datetime)
            - state (string)
            - district (string)
            - total_updates (int)
            - anomaly_flag (bool, optional)
            And other columns as needed
        
        reference_month: Reference month for analysis. If None, uses last month
                        in data. Should be a pd.Timestamp or datetime-like.
        
        growth_threshold: MoM growth threshold for spike detection (default: 0.20)
        
        window_months: Rolling window size for capacity computation (default: 12)
        
        percentile: Percentile for normal capacity (default: 75.0)
        
        min_spike_history: Minimum spike occurrences for empirical probability
                          (default: 2)
        
        lookahead_months: Months ahead to check for stress events (default: 1)
    
    Returns:
        DataFrame with columns:
        - state: State name
        - district: District name
        - reference_month: Reference month for predictions
        - normal_capacity: 75th percentile of 12-month rolling updates
        - predicted_updates_next_month: Predicted updates for next month
        - stress_likelihood: Probability of exceeding capacity (0-1)
        - risk_level: Categorical risk assessment (Low/Medium/High)
    """
    
    # Validate input
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_cols = ['month', 'state', 'district', 'total_updates']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure month is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['month']):
        df = df.copy()
        df['month'] = pd.to_datetime(df['month'])
    
    # Set reference month if not provided
    if reference_month is None:
        reference_month = df['month'].max()
    else:
        reference_month = pd.Timestamp(reference_month)
    
    # Initialize results container
    results = []
    
    # Process each district independently
    for (state, district), district_data in df.groupby(['state', 'district']):
        # Sort by month to ensure proper rolling window calculation
        district_data = district_data.sort_values('month').reset_index(drop=True)
        
        # Skip districts with insufficient data
        if len(district_data) < 2:
            continue
        
        # ====================================================================
        # Step 1: Compute rolling normal capacity
        # ====================================================================
        normal_capacity = compute_rolling_normal_capacity(
            district_data,
            window_months=window_months,
            percentile=percentile
        )
        
        # ====================================================================
        # Step 2: Identify historical stress events
        # ====================================================================
        stress_events = identify_stress_events(
            district_data,
            normal_capacity
        )
        
        # ====================================================================
        # Step 3: Identify recent spikes
        # ====================================================================
        recent_spikes = identify_recent_spikes(
            district_data,
            growth_threshold=growth_threshold,
            use_anomaly_flag=True
        )
        
        # ====================================================================
        # Step 4: Compute stress likelihood (primary or fallback method)
        # ====================================================================
        empirical_prob = compute_empirical_stress_probability(
            district_data,
            recent_spikes,
            stress_events,
            lookahead_months=lookahead_months
        )
        
        # Determine stress likelihood
        if empirical_prob is not None:
            # Sufficient spike history: use empirical probability
            stress_likelihood = empirical_prob
        else:
            # Insufficient spike history: use heuristic fallback
            heuristic_score = compute_heuristic_risk_score(district_data)
            stress_likelihood = heuristic_score
        
        # ====================================================================
        # Step 5: Get capacity and predictions for reference month
        # ====================================================================
        
        # Get the latest normal capacity (or most recent valid value)
        latest_capacity = normal_capacity.dropna().iloc[-1] if len(normal_capacity.dropna()) > 0 else district_data['total_updates'].mean()
        
        # Predict next month's updates
        predicted_next_month = predict_next_month_updates(
            district_data,
            method='mean'
        )
        
        # Map likelihood to risk level
        risk_level = map_likelihood_to_risk_level(stress_likelihood)
        
        # ====================================================================
        # Step 6: Record results
        # ====================================================================
        results.append({
            'state': state,
            'district': district,
            'reference_month': reference_month,
            'normal_capacity': round(latest_capacity, 2),
            'predicted_updates_next_month': round(predicted_next_month, 2),
            'stress_likelihood': round(stress_likelihood, 4),
            'risk_level': risk_level
        })
    
    # Construct results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by stress_likelihood descending for policy priority
    if len(results_df) > 0:
        results_df = results_df.sort_values(
            'stress_likelihood',
            ascending=False
        ).reset_index(drop=True)
    
    return results_df


# ============================================================================
# SECTION 3: UTILITY FUNCTIONS FOR INTERPRETATION
# ============================================================================

def summarize_stress_results(results_df: pd.DataFrame) -> Dict:
    """
    Generate a summary of stress predictions for policy makers.
    
    Args:
        results_df: Output DataFrame from estimate_district_stress_likelihood()
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_districts': len(results_df),
        'high_risk_count': (results_df['risk_level'] == 'High').sum(),
        'medium_risk_count': (results_df['risk_level'] == 'Medium').sum(),
        'low_risk_count': (results_df['risk_level'] == 'Low').sum(),
        'mean_stress_likelihood': results_df['stress_likelihood'].mean(),
        'median_stress_likelihood': results_df['stress_likelihood'].median(),
        'max_stress_likelihood': results_df['stress_likelihood'].max(),
        'high_risk_districts': results_df[results_df['risk_level'] == 'High'][
            ['state', 'district', 'stress_likelihood']
        ].to_dict('records')
    }
    
    return summary


def get_district_details(
    results_df: pd.DataFrame,
    state: str,
    district: str
) -> Dict:
    """
    Retrieve detailed stress prediction for a specific district.
    
    Args:
        results_df: Output DataFrame from estimate_district_stress_likelihood()
        state: State name
        district: District name
    
    Returns:
        Dictionary with district details, or None if not found
    """
    match = results_df[
        (results_df['state'] == state) & (results_df['district'] == district)
    ]
    
    if len(match) > 0:
        return match.iloc[0].to_dict()
    else:
        return None


# ============================================================================
# SECTION 4: EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the stress prediction system.
    
    Assumes df is already loaded with the required structure:
    - month (datetime)
    - state (string)
    - district (string)
    - enrolments (int)
    - demographic_updates (int)
    - biometric_updates (int)
    - address_updates (int)
    - total_updates (int, derived)
    - anomaly_flag (bool, optional)
    """
    
    # NOTE: Uncomment and modify the following lines to run with actual data
    
    # df = pd.read_csv('aadhaar_data.csv')
    # df['month'] = pd.to_datetime(df['month'])
    # df['total_updates'] = df['demographic_updates'] + df['biometric_updates']
    # 
    # # Run stress prediction
    # results = estimate_district_stress_likelihood(
    #     df,
    #     reference_month=pd.Timestamp('2024-12-31')
    # )
    # 
    # # Display top high-risk districts
    # print("Top High-Risk Districts:")
    # print(results[results['risk_level'] == 'High'].head(10))
    # 
    # # Get summary statistics
    # summary = summarize_stress_results(results)
    # print("\nSummary Statistics:")
    # for key, value in summary.items():
    #     if key != 'high_risk_districts':
    #         print(f"  {key}: {value}")
    # 
    # # Save results
    # results.to_csv('stress_predictions.csv', index=False)
    
    print("Aadhaar Stress Prediction System loaded successfully.")
    print("Ready to process aggregated district-level data.")
