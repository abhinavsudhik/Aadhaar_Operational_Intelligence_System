"""
Aadhaar Demand Forecasting Module

Provides district-level forecasts of Aadhaar updates for operational planning.
Returns processed summaries suitable for dashboards and executive reports.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta


def forecast_district_demand(
    df: pd.DataFrame,
    lookback_months: int = 12,
    forecast_horizon: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast Aadhaar update demand at district level with operational classification.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns:
        - month (datetime or YYYY-MM)
        - state (str)
        - district (str)
        - total_updates (int) = demographic_updates + biometric_updates
    lookback_months : int, default=12
        Number of historical months to use for trend estimation (6-12 recommended)
    forecast_horizon : int, default=3
        Number of months ahead to forecast (1-3)
    
    Returns
    -------
    district_forecast : pd.DataFrame
        ONE ROW PER DISTRICT with columns:
        - state
        - district
        - reference_month (latest observed month)
        - forecast_1m (expected updates next month)
        - forecast_3m_avg (average monthly forecast over next 3 months)
        - demand_level (Normal / Elevated / High)
        - operational_note (brief explanation)
    
    state_summary : pd.DataFrame
        State-level aggregation:
        - state
        - high_demand_districts (count)
        - elevated_demand_districts (count)
        - total_districts (count)
    
    Notes
    -----
    - Uses simple linear regression on recent months for trend estimation
    - Uncertainty bands based on ±1 std dev of historical volatility
    - Demand classification compares forecast to historical mean + 0.5*std
    """
    
    # Ensure month is datetime
    df = df.copy()
    if df['month'].dtype == 'object':
        df['month'] = pd.to_datetime(df['month'])
    
    # Sort by district and month
    df = df.sort_values(['state', 'district', 'month']).reset_index(drop=True)
    
    # Get unique districts
    districts = df.groupby(['state', 'district']).size().reset_index()[['state', 'district']]
    
    # Initialize results list
    results = []
    
    for _, row in districts.iterrows():
        state = row['state']
        district = row['district']
        
        # Filter data for this district
        district_data = df[
            (df['state'] == state) & 
            (df['district'] == district)
        ].copy()
        
        # Get last N months
        district_data = district_data.tail(lookback_months)
        
        # Skip if insufficient data
        if len(district_data) < 3:
            continue
        
        # Compute forecast
        forecast_summary = _compute_district_forecast(
            district_data, 
            state, 
            district, 
            forecast_horizon
        )
        
        if forecast_summary is not None:
            results.append(forecast_summary)
    
    # Create district-level output
    district_forecast = pd.DataFrame(results)
    
    # Create state-level summary
    state_summary = _create_state_summary(district_forecast)
    
    return district_forecast, state_summary


def _compute_district_forecast(
    district_data: pd.DataFrame,
    state: str,
    district: str,
    forecast_horizon: int
) -> dict:
    """
    Compute forecast metrics for a single district.
    
    Returns a dictionary with forecast values and operational classification.
    """
    
    # Extract time series
    months = district_data['month'].values
    updates = district_data['total_updates'].values
    
    reference_month = months[-1]
    
    # Convert months to numeric (days since first month) for linear regression
    months_numeric = (months - months[0]).astype('timedelta64[D]').astype(int)
    
    # Simple linear regression: y = a + b*x
    n = len(months_numeric)
    x_mean = np.mean(months_numeric)
    y_mean = np.mean(updates)
    
    # Compute slope
    numerator = np.sum((months_numeric - x_mean) * (updates - y_mean))
    denominator = np.sum((months_numeric - x_mean) ** 2)
    
    if denominator == 0:
        # No trend, use mean
        slope = 0
        intercept = y_mean
    else:
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
    
    # Historical statistics
    historical_mean = np.mean(updates)
    historical_std = np.std(updates)
    
    # Forecast future months
    # Assume months are roughly 30 days apart
    last_month_numeric = months_numeric[-1]
    days_per_month = 30
    
    forecasts = []
    for m in range(1, forecast_horizon + 1):
        future_x = last_month_numeric + (m * days_per_month)
        forecast_value = intercept + slope * future_x
        # Ensure non-negative
        forecast_value = max(0, forecast_value)
        forecasts.append(forecast_value)
    
    forecast_1m = forecasts[0]
    forecast_3m_avg = np.mean(forecasts)
    
    # Compute uncertainty band (±1 std dev)
    lower_bound = forecast_1m - historical_std
    upper_bound = forecast_1m + historical_std
    
    # Classify demand level
    # Thresholds:
    # - High: forecast > historical_mean + 0.5 * std
    # - Elevated: forecast > historical_mean
    # - Normal: otherwise
    
    high_threshold = historical_mean + 0.5 * historical_std
    elevated_threshold = historical_mean
    
    if forecast_1m > high_threshold:
        demand_level = "High"
        pct_above = ((forecast_1m - historical_mean) / historical_mean) * 100
        operational_note = f"Forecast {pct_above:.0f}% above historical avg. Prepare additional capacity."
    elif forecast_1m > elevated_threshold:
        demand_level = "Elevated"
        pct_above = ((forecast_1m - historical_mean) / historical_mean) * 100
        operational_note = f"Forecast {pct_above:.0f}% above avg. Monitor closely."
    else:
        demand_level = "Normal"
        operational_note = "Forecast within normal range. Standard operations."
    
    # Format reference month
    if isinstance(reference_month, (pd.Timestamp, np.datetime64)):
        ref_month_str = pd.Timestamp(reference_month).strftime('%Y-%m')
    else:
        ref_month_str = str(reference_month)
    
    return {
        'state': state,
        'district': district,
        'reference_month': ref_month_str,
        'forecast_1m': int(round(forecast_1m)),
        'forecast_3m_avg': int(round(forecast_3m_avg)),
        'uncertainty_lower': int(round(lower_bound)),
        'uncertainty_upper': int(round(upper_bound)),
        'demand_level': demand_level,
        'operational_note': operational_note
    }


def _create_state_summary(district_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate district forecasts to state-level summary.
    
    Returns DataFrame with high/elevated demand counts per state.
    """
    
    if district_forecast.empty:
        return pd.DataFrame(columns=[
            'state', 
            'high_demand_districts', 
            'elevated_demand_districts',
            'total_districts'
        ])
    
    # Count districts by demand level per state
    summary = district_forecast.groupby('state').agg(
        high_demand_districts=('demand_level', lambda x: (x == 'High').sum()),
        elevated_demand_districts=('demand_level', lambda x: (x == 'Elevated').sum()),
        total_districts=('district', 'count')
    ).reset_index()
    
    # Sort by states with most high-demand districts
    summary = summary.sort_values('high_demand_districts', ascending=False)
    
    return summary


def generate_forecast_report(
    district_forecast: pd.DataFrame,
    state_summary: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    Generate a text report summarizing forecast results.
    
    Parameters
    ----------
    district_forecast : pd.DataFrame
        District-level forecast output
    state_summary : pd.DataFrame
        State-level summary
    output_path : str, optional
        Path to save report. If None, returns as string only.
    
    Returns
    -------
    report : str
        Formatted text report
    """
    
    lines = []
    lines.append("=" * 70)
    lines.append("AADHAAR DEMAND FORECAST - OPERATIONAL SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall statistics
    total_districts = len(district_forecast)
    high_count = (district_forecast['demand_level'] == 'High').sum()
    elevated_count = (district_forecast['demand_level'] == 'Elevated').sum()
    normal_count = (district_forecast['demand_level'] == 'Normal').sum()
    
    lines.append(f"Total Districts Analyzed: {total_districts}")
    lines.append(f"  - High Demand:     {high_count:4d} ({high_count/total_districts*100:5.1f}%)")
    lines.append(f"  - Elevated Demand: {elevated_count:4d} ({elevated_count/total_districts*100:5.1f}%)")
    lines.append(f"  - Normal Demand:   {normal_count:4d} ({normal_count/total_districts*100:5.1f}%)")
    lines.append("")
    
    # State-level summary
    lines.append("STATE-LEVEL BREAKDOWN")
    lines.append("-" * 70)
    for _, row in state_summary.head(10).iterrows():
        lines.append(
            f"{row['state']:20s} | "
            f"High: {row['high_demand_districts']:3d} | "
            f"Elevated: {row['elevated_demand_districts']:3d} | "
            f"Total: {row['total_districts']:3d}"
        )
    lines.append("")
    
    # Top 10 high-demand districts
    high_demand = district_forecast[
        district_forecast['demand_level'] == 'High'
    ].sort_values('forecast_1m', ascending=False).head(10)
    
    if len(high_demand) > 0:
        lines.append("TOP 10 HIGH-DEMAND DISTRICTS (Next Month Forecast)")
        lines.append("-" * 70)
        for _, row in high_demand.iterrows():
            lines.append(
                f"{row['district']:25s} ({row['state']:15s}) | "
                f"Forecast: {row['forecast_1m']:8,d} | "
                f"{row['operational_note']}"
            )
        lines.append("")
    
    report = "\n".join(lines)
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


# Example usage
if __name__ == "__main__":
    # This demonstrates how to use the module
    # Assumes you have a DataFrame 'df' loaded with the required columns
    
    print("Aadhaar Demand Forecaster Module")
    print("=" * 50)
    print("\nUsage Example:")
    print("""
    import pandas as pd
    from aadhaar_demand_forecaster import forecast_district_demand, generate_forecast_report
    
    # Load your data
    df = pd.read_csv('your_aadhaar_data.csv')
    
    # Ensure total_updates column exists
    df['total_updates'] = df['demographic_updates'] + df['biometric_updates']
    
    # Generate forecasts
    district_forecast, state_summary = forecast_district_demand(
        df, 
        lookback_months=12,
        forecast_horizon=3
    )
    
    # View results
    print(district_forecast.head())
    print(state_summary)
    
    # Generate report
    report = generate_forecast_report(district_forecast, state_summary)
    print(report)
    
    # Save outputs
    district_forecast.to_csv('district_demand_forecast.csv', index=False)
    state_summary.to_csv('state_demand_summary.csv', index=False)
    """)
