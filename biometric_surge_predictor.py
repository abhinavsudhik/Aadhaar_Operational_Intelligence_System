"""
Biometric Update Surge Predictor for Aadhaar District Operations
================================================================

Predicts district-level biometric update surges driven by natural lifecycle
transitions and periodic refresh cycles (1-3 month horizon).

Key Methodology:
- Analyzes 24-month rolling windows of historical biometric patterns
- Identifies seasonal peaks and recurring cycles via rolling averages
- Examines lagged relationships (5-17 age group → 18+ biometric updates)
- Detects entry into known biometric refresh phases
- Estimates surge likelihood and expected impact

Output: Compact, actionable district-level and state-level predictions
suitable for operational dashboards and capacity planning.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


def prepare_district_timeseries(
    df: pd.DataFrame,
    state: str,
    district: str,
    lookback_months: int = 24
) -> pd.DataFrame:
    """
    Extract and prepare time series for a single district.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset with columns: month, state, district, age_group, 
        biometric_updates, enrolments, biometric_update_rate, anomaly_flag
    state : str
        State filter
    district : str
        District filter
    lookback_months : int
        Historical window size (default: 24 months)
    
    Returns:
    --------
    pd.DataFrame
        Time series for district, sorted by month, with derived features
    """
    # Filter by state and district
    mask = (df['state'] == state) & (df['district'] == district)
    district_df = df[mask].copy()
    
    if len(district_df) == 0:
        return pd.DataFrame()
    
    # Convert month to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(district_df['month']):
        district_df['month'] = pd.to_datetime(district_df['month'])
    
    # Sort by month
    district_df = district_df.sort_values('month').reset_index(drop=True)
    
    # Limit to lookback window
    if len(district_df) > lookback_months:
        district_df = district_df.tail(lookback_months).reset_index(drop=True)
    
    # Ensure no gaps in time series (forward fill for missing months)
    month_range = pd.date_range(
        start=district_df['month'].min(),
        end=district_df['month'].max(),
        freq='MS'
    )
    
    # Create complete month index
    complete_df = pd.DataFrame({'month': month_range})
    district_df = complete_df.merge(
        district_df,
        on='month',
        how='left'
    )
    
    # Forward fill biometric updates and enrolments
    for col in ['biometric_updates', 'enrolments', 'biometric_update_rate']:
        if col in district_df.columns:
            district_df[col] = district_df[col].fillna(method='ffill')
    
    # Fill remaining NaNs with 0
    numeric_cols = district_df.select_dtypes(include=[np.number]).columns
    district_df[numeric_cols] = district_df[numeric_cols].fillna(0)
    
    # Back-fill state, district (from original or assumed)
    district_df['state'] = state
    district_df['district'] = district
    
    return district_df[['month', 'state', 'district', 'biometric_updates', 
                        'enrolments', 'biometric_update_rate']].reset_index(drop=True)


def extract_age_group_series(
    df: pd.DataFrame,
    state: str,
    district: str,
    age_group: str,
    metric: str = 'biometric_updates'
) -> pd.Series:
    """
    Extract time series for specific age group and metric.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    state : str
        State filter
    district : str
        District filter
    age_group : str
        Age group filter ('0-5', '5-17', '18+')
    metric : str
        Metric to extract ('biometric_updates' or 'enrolments')
    
    Returns:
    --------
    pd.Series
        Time series indexed by month
    """
    mask = ((df['state'] == state) & 
            (df['district'] == district) & 
            (df['age_group'] == age_group))
    
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return pd.Series(dtype=float)
    
    if not pd.api.types.is_datetime64_any_dtype(subset['month']):
        subset['month'] = pd.to_datetime(subset['month'])
    
    subset = subset.sort_values('month').set_index('month')
    return subset[metric].asfreq('MS', fill_value=0)


def compute_seasonal_baseline(
    ts: pd.Series,
    window: int = 3
) -> Tuple[float, float]:
    """
    Compute seasonal baseline using rolling average and volatility.
    
    Parameters:
    -----------
    ts : pd.Series
        Time series of biometric updates
    window : int
        Rolling window size (months)
    
    Returns:
    --------
    Tuple[float, float]
        (baseline_level, baseline_std_dev)
    """
    if len(ts) < window:
        return ts.mean(), ts.std()
    
    # Remove zero/anomaly observations for baseline
    clean_ts = ts[ts > 0]
    if len(clean_ts) == 0:
        return ts.mean(), ts.std()
    
    # Rolling mean and std
    rolling_mean = clean_ts.rolling(window=window, center=True).mean()
    baseline = rolling_mean.median() if len(rolling_mean) > 0 else clean_ts.mean()
    baseline_std = clean_ts.std()
    
    return float(baseline), float(baseline_std)


def detect_seasonal_peaks(
    ts: pd.Series,
    baseline: float,
    baseline_std: float,
    threshold_multiplier: float = 1.5
) -> List[int]:
    """
    Identify months with significant biometric update peaks.
    
    A peak is defined as: biometric_updates > baseline + threshold_multiplier * std_dev
    
    Parameters:
    -----------
    ts : pd.Series
        Time series of biometric updates
    baseline : float
        Historical average
    baseline_std : float
        Historical std dev
    threshold_multiplier : float
        Sensitivity parameter (1.5 = 1.5 std dev above baseline)
    
    Returns:
    --------
    List[int]
        List of month indices where peaks occur
    """
    if baseline_std == 0 or baseline == 0:
        return []
    
    threshold = baseline + threshold_multiplier * baseline_std
    peaks = np.where(ts > threshold)[0].tolist()
    return peaks


def compute_age_transition_lag(
    df: pd.DataFrame,
    state: str,
    district: str,
    lag_months: int = 6
) -> float:
    """
    Measure correlation between 5-17 enrolments and subsequent 18+ biometric updates.
    
    Logic: Natural lifecycle transitions cause a lag between when youth transition
    into 18+ age group and when biometric updates occur. This lag typically spans
    2-6 months.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    state : str
        State filter
    district : str
        District filter
    lag_months : int
        Max lag to test (default: 6)
    
    Returns:
    --------
    float
        Correlation strength (0-1) with best lag
    """
    # Get age group time series
    enrol_5_17 = extract_age_group_series(df, state, district, '5-17', 'enrolments')
    biom_18_plus = extract_age_group_series(df, state, district, '18+', 'biometric_updates')
    
    if len(enrol_5_17) < lag_months + 2 or len(biom_18_plus) < lag_months + 2:
        return 0.0
    
    # Normalize to handle scale differences
    enrol_5_17_norm = (enrol_5_17 - enrol_5_17.mean()) / (enrol_5_17.std() + 1e-6)
    biom_18_norm = (biom_18_plus - biom_18_plus.mean()) / (biom_18_plus.std() + 1e-6)
    
    max_corr = 0.0
    
    # Test lags from 1 to lag_months
    for lag in range(1, lag_months + 1):
        # Shift biometric series forward (check if past enrolments predict future updates)
        lagged_enrol = enrol_5_17_norm.shift(lag).dropna()
        corresponding_biom = biom_18_norm[lagged_enrol.index]
        
        if len(lagged_enrol) > 2:
            corr = float(np.corrcoef(lagged_enrol, corresponding_biom)[0, 1])
            if not np.isnan(corr) and corr > max_corr:
                max_corr = corr
    
    return max(0.0, float(max_corr))


def predict_district_surge(
    df: pd.DataFrame,
    state: str,
    district: str,
    reference_month: str = None,
    surge_window_months: int = 3,
    lookback_months: int = 24
) -> Dict[str, Any]:
    """
    Predict biometric update surge for a district in the next 1-3 months.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    state : str
        State name
    district : str
        District name
    reference_month : str, optional
        Reference month for prediction (default: latest month in data)
    surge_window_months : int
        Forward-looking prediction window (default: 3 months)
    lookback_months : int
        Historical window for baseline (default: 24 months)
    
    Returns:
    --------
    Dict[str, Any]
        Prediction results containing:
        - biometric_surge_expected (bool)
        - surge_probability (float, 0-1)
        - expected_surge_window (str)
        - expected_impact_level (str)
        - operational_note (str)
        - supporting_signals (dict)
    """
    
    # Prepare district time series
    ts_df = prepare_district_timeseries(df, state, district, lookback_months)
    
    if len(ts_df) == 0:
        return {
            'state': state,
            'district': district,
            'reference_month': reference_month or 'unknown',
            'biometric_surge_expected': 'No',
            'surge_probability': 0.0,
            'expected_surge_window': 'N/A',
            'expected_impact_level': 'Low',
            'operational_note': 'Insufficient data for prediction.',
            'supporting_signals': {}
        }
    
    # Set reference month to latest in dataset if not provided
    if reference_month is None:
        reference_month = ts_df['month'].max().strftime('%Y-%m')
    
    # Extract biometric updates series (aggregated across age groups)
    biom_series = ts_df['biometric_updates'].fillna(0)
    
    if len(biom_series[biom_series > 0]) < 3:
        return {
            'state': state,
            'district': district,
            'reference_month': reference_month,
            'biometric_surge_expected': 'No',
            'surge_probability': 0.0,
            'expected_surge_window': 'N/A',
            'expected_impact_level': 'Low',
            'operational_note': 'Insufficient biometric activity for surge prediction.',
            'supporting_signals': {}
        }
    
    # SIGNAL 1: Seasonal peak detection
    baseline, baseline_std = compute_seasonal_baseline(biom_series, window=3)
    peaks = detect_seasonal_peaks(biom_series, baseline, baseline_std, threshold_multiplier=1.5)
    
    # Check if recent months show increasing trend toward peak season
    recent_biom = biom_series.tail(6)
    if len(recent_biom) >= 3:
        recent_trend = recent_biom.rolling(2).mean().diff().mean()
    else:
        recent_trend = 0.0
    
    seasonal_signal = 0.0
    if len(peaks) > 0:
        # If we've had peaks in past 24 months, likely to recur (seasonal)
        seasonal_signal = min(1.0, len(peaks) / 8.0) * 0.6  # Weight: 60%
    
    if recent_trend > 0:
        seasonal_signal += min(recent_trend / (baseline_std + 1e-6), 0.4)  # Trend boost
    
    # SIGNAL 2: Age transition lag correlation
    transition_corr = compute_age_transition_lag(df, state, district, lag_months=6)
    transition_signal = transition_corr * 0.4  # Weight: 40%
    
    # SIGNAL 3: Recent volatility (increased variance = preparation phase)
    if len(recent_biom) >= 6:
        recent_vol = recent_biom.std()
        historical_vol = biom_series.std()
        volatility_signal = 0.3 if (recent_vol > historical_vol * 1.2) else 0.0
    else:
        volatility_signal = 0.0
    
    # COMBINED SURGE PROBABILITY
    surge_probability = min(1.0, seasonal_signal + transition_signal + volatility_signal * 0.3)
    
    # DETERMINE SURGE EXPECTED (threshold: 0.35)
    surge_threshold = 0.35
    biometric_surge_expected = 'Yes' if surge_probability >= surge_threshold else 'No'
    
    # ESTIMATE IMPACT LEVEL based on expected magnitude
    if surge_probability >= 0.7:
        expected_impact_level = 'High'
        expected_surge_window = 'Next 1–2 months'
    elif surge_probability >= 0.5:
        expected_impact_level = 'Moderate'
        expected_surge_window = 'Next 2–3 months'
    else:
        expected_impact_level = 'Low'
        expected_surge_window = 'Next 3 months (likely)'
    
    # BUILD OPERATIONAL NOTE
    signals_desc = []
    if seasonal_signal > 0.2:
        signals_desc.append('seasonal refresh cycle detected')
    if transition_corr > 0.3:
        signals_desc.append('age-group transition pattern observed')
    if volatility_signal > 0.1:
        signals_desc.append('increased activity in recent months')
    
    if signals_desc:
        operational_note = (
            f"District entering biometric refresh phase ({', '.join(signals_desc)}). "
            f"Recommend capacity review for biometric capture infrastructure."
        )
    else:
        operational_note = "Current activity levels stable. Continued monitoring recommended."
    
    return {
        'state': state,
        'district': district,
        'reference_month': reference_month,
        'biometric_surge_expected': biometric_surge_expected,
        'surge_probability': round(surge_probability, 3),
        'expected_surge_window': expected_surge_window,
        'expected_impact_level': expected_impact_level,
        'operational_note': operational_note,
        'supporting_signals': {
            'seasonal_signal': round(seasonal_signal, 3),
            'transition_correlation': round(transition_corr, 3),
            'recent_volatility_signal': round(volatility_signal, 3),
            'baseline_monthly_updates': round(baseline, 0),
        }
    }


def predict_all_districts(
    df: pd.DataFrame,
    reference_month: str = None,
    surge_window_months: int = 3,
    lookback_months: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict biometric surges for ALL districts in dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    reference_month : str, optional
        Reference month for all predictions (default: latest)
    surge_window_months : int
        Forward-looking window (default: 3 months)
    lookback_months : int
        Historical window (default: 24 months)
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        - district_predictions: One row per district with key metrics
        - state_summary: State-level aggregation
    """
    
    # Get unique state-district combinations
    unique_districts = df.groupby(['state', 'district']).size().reset_index(name='count')
    
    predictions = []
    
    for _, row in unique_districts.iterrows():
        state = row['state']
        district = row['district']
        
        pred = predict_district_surge(
            df, state, district,
            reference_month=reference_month,
            surge_window_months=surge_window_months,
            lookback_months=lookback_months
        )
        predictions.append(pred)
    
    # Convert to DataFrame
    district_predictions = pd.DataFrame(predictions)
    
    # Ensure consistent column order
    column_order = [
        'state', 'district', 'reference_month',
        'biometric_surge_expected', 'surge_probability',
        'expected_surge_window', 'expected_impact_level',
        'operational_note'
    ]
    district_predictions = district_predictions[column_order]
    
    # STATE-LEVEL SUMMARY
    state_summary = district_predictions.groupby('state').agg({
        'district': 'count',
        'biometric_surge_expected': lambda x: (x == 'Yes').sum()
    }).reset_index()
    
    state_summary.columns = ['state', 'total_districts', 'districts_with_surge']
    state_summary['percent_affected'] = (
        100 * state_summary['districts_with_surge'] / state_summary['total_districts']
    ).round(1)
    
    state_summary = state_summary[[
        'state', 'total_districts', 'districts_with_surge', 'percent_affected'
    ]]
    
    return district_predictions, state_summary


def format_operational_insight(prediction_row: pd.Series) -> str:
    """
    Convert a district prediction row into a plain-language operational insight.
    
    Parameters:
    -----------
    prediction_row : pd.Series
        Single row from district_predictions DataFrame
    
    Returns:
    --------
    str
        Human-readable insight suitable for dashboards/reports
    """
    state = prediction_row['state']
    district = prediction_row['district']
    surge_expected = prediction_row['biometric_surge_expected']
    probability = prediction_row['surge_probability']
    impact = prediction_row['expected_impact_level']
    window = prediction_row['expected_surge_window']
    
    if surge_expected == 'Yes':
        insight = (
            f"{district} ({state}) is entering a natural biometric refresh phase. "
            f"Surge probability: {probability*100:.0f}%. Expected impact: {impact}. "
            f"Recommended action: Review biometric capture capacity and staff scheduling "
            f"for {window}. Enhanced monitoring advised."
        )
    else:
        insight = (
            f"{district} ({state}) shows stable biometric activity patterns. "
            f"Current demand levels sustainable with existing infrastructure. "
            f"Routine monitoring sufficient."
        )
    
    return insight


def generate_prediction_report(
    df: pd.DataFrame,
    reference_month: str = None,
    include_top_surge_districts: int = 10
) -> Dict[str, Any]:
    """
    Generate comprehensive prediction report with key insights.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    reference_month : str, optional
        Reference month (default: latest)
    include_top_surge_districts : int
        Number of high-risk districts to highlight (default: 10)
    
    Returns:
    --------
    Dict[str, Any]
        Report containing predictions, summaries, and insights
    """
    
    # Run full predictions
    district_preds, state_summary = predict_all_districts(df, reference_month)
    
    # Top surge districts (sorted by probability)
    top_surge_districts = (
        district_preds[district_preds['biometric_surge_expected'] == 'Yes']
        .nlargest(include_top_surge_districts, 'surge_probability')
    )
    
    # Generate insights for top districts
    top_insights = top_surge_districts.apply(
        format_operational_insight, axis=1
    ).tolist()
    
    report = {
        'reference_month': reference_month or df['month'].max().strftime('%Y-%m'),
        'prediction_timestamp': datetime.now().isoformat(),
        'total_districts_analyzed': len(district_preds),
        'districts_with_surge': (district_preds['biometric_surge_expected'] == 'Yes').sum(),
        'overall_surge_percentage': round(
            100 * (district_preds['biometric_surge_expected'] == 'Yes').sum() / len(district_preds), 1
        ),
        'district_predictions': district_preds,
        'state_summary': state_summary,
        'top_surge_districts': top_surge_districts,
        'top_surge_insights': top_insights,
    }
    
    return report


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == '__main__':
    """
    Example usage:
    
    # Load your data
    df = pd.read_csv('your_aadhaar_data.csv')
    
    # Generate predictions for all districts
    report = generate_prediction_report(df)
    
    # Access key outputs
    print(report['state_summary'])
    print(report['district_predictions'])
    
    # Print top surge districts
    for insight in report['top_surge_insights']:
        print(f"- {insight}\n")
    """
    
    print("Biometric Surge Predictor loaded successfully.")
    print("Use generate_prediction_report(df) to run predictions.")
