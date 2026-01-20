"""
Anomaly Persistence Analyzer for Aadhaar Operations

This module determines whether detected anomalies in district-level Aadhaar indicators
are likely to persist (structural change) or self-correct (temporary spike).

Output is designed for decision-makers, not analysts.

Author: Senior Data Analyst Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime, timedelta


class AnomalyPersistenceAnalyzer:
    """
    Analyzes anomaly patterns to predict persistence vs self-correction.
    
    Definitions:
    - Persistent: Elevated for ≥3 months OR post-mean significantly higher than baseline
    - Self-correcting: Returns to within ±10% of baseline within 2 months
    """
    
    def __init__(self, baseline_months: int = 6, followup_months: int = 3, 
                 correction_threshold: float = 0.10, min_persistence_months: int = 3):
        """
        Initialize analyzer with configurable parameters.
        
        Args:
            baseline_months: Months to use for pre-anomaly baseline (default: 6)
            followup_months: Months to track post-anomaly behavior (default: 3)
            correction_threshold: Acceptable deviation from baseline (default: 0.10 = ±10%)
            min_persistence_months: Consecutive months for persistence (default: 3)
        """
        self.baseline_months = baseline_months
        self.followup_months = followup_months
        self.correction_threshold = correction_threshold
        self.min_persistence_months = min_persistence_months
    
    
    def analyze_anomalies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main analysis function. Classifies anomalies and generates decision summaries.
        
        Args:
            df: DataFrame with columns: month, state, district, indicator, 
                indicator_value, anomaly_flag, anomaly_magnitude
        
        Returns:
            Tuple of (anomaly_classifications, district_summary)
            - anomaly_classifications: One row per active anomaly with classification
            - district_summary: Aggregated statistics by district
        """
        # Ensure month is datetime
        df = df.copy()
        df['month'] = pd.to_datetime(df['month'])
        df = df.sort_values(['state', 'district', 'indicator', 'month']).reset_index(drop=True)
        
        # Identify all detected anomalies
        anomalies = df[df['anomaly_flag'] == True].copy()
        
        if len(anomalies) == 0:
            return self._empty_results()
        
        # Calculate historical persistence rates
        historical_rates = self._calculate_historical_persistence(df)
        
        # Classify each anomaly
        classifications = []
        for idx, anomaly in anomalies.iterrows():
            classification = self._classify_single_anomaly(
                df, anomaly, historical_rates
            )
            if classification is not None:
                classifications.append(classification)
        
        if len(classifications) == 0:
            return self._empty_results()
        
        # Create anomaly classification DataFrame
        anomaly_df = pd.DataFrame(classifications)
        anomaly_df = self._assign_categories(anomaly_df)
        
        # Generate district summary
        district_summary = self._generate_district_summary(anomaly_df)
        
        return anomaly_df, district_summary
    
    
    def _classify_single_anomaly(self, df: pd.DataFrame, anomaly: pd.Series, 
                                  historical_rates: Dict) -> Dict:
        """
        Classify a single anomaly as persistent or self-correcting.
        
        Returns:
            Dictionary with classification details, or None if insufficient data
        """
        state = anomaly['state']
        district = anomaly['district']
        indicator = anomaly['indicator']
        anomaly_month = anomaly['month']
        
        # Filter to this specific indicator/district combination
        entity_df = df[
            (df['state'] == state) & 
            (df['district'] == district) & 
            (df['indicator'] == indicator)
        ].sort_values('month')
        
        # Calculate pre-anomaly baseline (6 months before)
        baseline_start = anomaly_month - pd.DateOffset(months=self.baseline_months)
        baseline_data = entity_df[
            (entity_df['month'] >= baseline_start) & 
            (entity_df['month'] < anomaly_month)
        ]
        
        if len(baseline_data) < 3:  # Need at least 3 months for baseline
            return None
        
        baseline_mean = baseline_data['indicator_value'].mean()
        baseline_std = baseline_data['indicator_value'].std()
        
        # Track post-anomaly behavior (next 3 months)
        followup_end = anomaly_month + pd.DateOffset(months=self.followup_months)
        followup_data = entity_df[
            (entity_df['month'] > anomaly_month) & 
            (entity_df['month'] <= followup_end)
        ]
        
        # Calculate persistence metrics
        persistence_signals = self._evaluate_persistence(
            anomaly, baseline_mean, baseline_std, followup_data
        )
        
        # Get historical persistence rate for this context
        hist_key = f"{indicator}_{state}"
        historical_persistence_rate = historical_rates.get(hist_key, 0.5)  # Default to 50%
        
        # Combine signals to compute persistence probability
        persistence_probability = self._compute_persistence_probability(
            persistence_signals, historical_persistence_rate
        )
        
        return {
            'state': state,
            'district': district,
            'indicator': indicator,
            'anomaly_month': anomaly_month,
            'anomaly_magnitude': anomaly['anomaly_magnitude'],
            'baseline_mean': baseline_mean,
            'persistence_probability': persistence_probability,
            'months_elevated': persistence_signals['months_elevated'],
            'returned_to_baseline': persistence_signals['returned_to_baseline']
        }
    
    
    def _evaluate_persistence(self, anomaly: pd.Series, baseline_mean: float, 
                              baseline_std: float, followup_data: pd.DataFrame) -> Dict:
        """
        Evaluate persistence signals from post-anomaly behavior.
        
        Returns:
            Dictionary with persistence metrics
        """
        signals = {
            'months_elevated': 0,
            'returned_to_baseline': False,
            'post_mean_elevated': False,
            'max_consecutive_elevated': 0
        }
        
        if len(followup_data) == 0:
            return signals
        
        # Check each month in followup period
        upper_bound = baseline_mean * (1 + self.correction_threshold)
        lower_bound = baseline_mean * (1 - self.correction_threshold)
        
        consecutive_elevated = 0
        max_consecutive = 0
        
        for _, row in followup_data.iterrows():
            value = row['indicator_value']
            
            # Count months outside acceptable range
            if value > upper_bound or value < lower_bound:
                signals['months_elevated'] += 1
                consecutive_elevated += 1
                max_consecutive = max(max_consecutive, consecutive_elevated)
            else:
                # Returned to baseline range
                signals['returned_to_baseline'] = True
                consecutive_elevated = 0
        
        signals['max_consecutive_elevated'] = max_consecutive
        
        # Check if post-anomaly mean is significantly elevated
        if len(followup_data) >= 2:
            post_mean = followup_data['indicator_value'].mean()
            # Significant if > baseline + 1 std dev
            if post_mean > baseline_mean + baseline_std:
                signals['post_mean_elevated'] = True
        
        return signals
    
    
    def _compute_persistence_probability(self, signals: Dict, 
                                         historical_rate: float) -> float:
        """
        Combine multiple signals to compute persistence probability (0-1).
        
        Uses weighted scoring approach for interpretability.
        """
        score = 0.0
        
        # Signal 1: Consecutive months elevated (0-0.4 weight)
        if signals['max_consecutive_elevated'] >= self.min_persistence_months:
            score += 0.4
        elif signals['max_consecutive_elevated'] == 2:
            score += 0.2
        elif signals['max_consecutive_elevated'] == 1:
            score += 0.1
        
        # Signal 2: Post-mean elevated (0-0.3 weight)
        if signals['post_mean_elevated']:
            score += 0.3
        
        # Signal 3: Did NOT return to baseline (0-0.2 weight)
        if not signals['returned_to_baseline'] and signals['months_elevated'] > 0:
            score += 0.2
        
        # Signal 4: Historical context (0-0.1 weight)
        score += historical_rate * 0.1
        
        # Normalize to 0-1 range
        probability = min(1.0, max(0.0, score))
        
        return round(probability, 3)
    
    
    def _calculate_historical_persistence(self, df: pd.DataFrame) -> Dict:
        """
        Calculate historical persistence rates by indicator and state.
        
        Returns:
            Dictionary mapping "{indicator}_{state}" to persistence rate
        """
        rates = {}
        
        # Group by indicator and state
        for (indicator, state), group in df.groupby(['indicator', 'state']):
            anomalies = group[group['anomaly_flag'] == True]
            
            if len(anomalies) < 3:  # Need sufficient history
                continue
            
            persistent_count = 0
            total_count = 0
            
            for idx, anomaly in anomalies.iterrows():
                # Look at behavior after each historical anomaly
                anomaly_month = anomaly['month']
                district = anomaly['district']
                
                followup_end = anomaly_month + pd.DateOffset(months=self.followup_months)
                followup = group[
                    (group['month'] > anomaly_month) & 
                    (group['month'] <= followup_end) &
                    (group['district'] == district)
                ]
                
                if len(followup) >= 2:
                    # Simple heuristic: persistent if anomaly_flag remains True
                    if followup['anomaly_flag'].sum() >= 2:
                        persistent_count += 1
                    total_count += 1
            
            if total_count > 0:
                key = f"{indicator}_{state}"
                rates[key] = persistent_count / total_count
        
        return rates
    
    
    def _assign_categories(self, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign classification and recommended response based on persistence probability.
        
        Categories:
        - Likely Self-Correcting (p < 0.33)
        - Uncertain – Monitor (0.33 ≤ p < 0.67)
        - Likely Persistent (p ≥ 0.67)
        """
        def classify(prob):
            if prob < 0.33:
                return 'Likely Self-Correcting'
            elif prob < 0.67:
                return 'Uncertain – Monitor'
            else:
                return 'Likely Persistent'
        
        def recommend(prob):
            if prob < 0.33:
                return 'Monitor – No immediate action'
            elif prob < 0.67:
                return 'Prepare – Assess response capacity'
            else:
                return 'Act – Immediate intervention needed'
        
        anomaly_df['classification'] = anomaly_df['persistence_probability'].apply(classify)
        anomaly_df['recommended_response'] = anomaly_df['persistence_probability'].apply(recommend)
        
        # Select and order columns for final output
        output_cols = [
            'state', 'district', 'indicator', 'anomaly_month',
            'persistence_probability', 'classification', 'recommended_response',
            'anomaly_magnitude', 'baseline_mean', 'months_elevated'
        ]
        
        return anomaly_df[output_cols]
    
    
    def _generate_district_summary(self, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate district-level summary statistics.
        
        Returns:
            DataFrame with total anomalies, % persistent, % self-correcting per district
        """
        summary = []
        
        for (state, district), group in anomaly_df.groupby(['state', 'district']):
            total = len(group)
            persistent = len(group[group['classification'] == 'Likely Persistent'])
            self_correcting = len(group[group['classification'] == 'Likely Self-Correcting'])
            uncertain = len(group[group['classification'] == 'Uncertain – Monitor'])
            
            summary.append({
                'state': state,
                'district': district,
                'total_anomalies': total,
                'persistent_count': persistent,
                'self_correcting_count': self_correcting,
                'uncertain_count': uncertain,
                'pct_persistent': round(100 * persistent / total, 1),
                'pct_self_correcting': round(100 * self_correcting / total, 1),
                'avg_persistence_probability': round(group['persistence_probability'].mean(), 3)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('pct_persistent', ascending=False)
        
        return summary_df
    
    
    def _empty_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return empty DataFrames with correct structure when no anomalies found."""
        empty_classifications = pd.DataFrame(columns=[
            'state', 'district', 'indicator', 'anomaly_month',
            'persistence_probability', 'classification', 'recommended_response',
            'anomaly_magnitude', 'baseline_mean', 'months_elevated'
        ])
        
        empty_summary = pd.DataFrame(columns=[
            'state', 'district', 'total_anomalies', 'persistent_count',
            'self_correcting_count', 'uncertain_count', 'pct_persistent',
            'pct_self_correcting', 'avg_persistence_probability'
        ])
        
        return empty_classifications, empty_summary


def generate_insight_text(row: pd.Series) -> str:
    """
    Convert a classification row into a human-readable insight string.
    
    Args:
        row: Single row from the anomaly classification DataFrame
    
    Returns:
        Human-readable string suitable for executive dashboard
    
    Example:
        >>> insight = generate_insight_text(anomaly_row)
        >>> print(insight)
        'High Risk: Karnataka-Bangalore Urban shows persistent MPI anomaly (85% likelihood).
         Immediate intervention needed.'
    """
    state = row['state']
    district = row['district']
    indicator = row['indicator'].upper()
    prob = row['persistence_probability']
    classification = row['classification']
    response = row['recommended_response']
    anomaly_month = pd.to_datetime(row['anomaly_month']).strftime('%B %Y')
    
    # Determine risk level
    if prob >= 0.67:
        risk_level = "High Risk"
    elif prob >= 0.33:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    # Format magnitude if available
    magnitude_text = ""
    if 'anomaly_magnitude' in row.index and pd.notna(row['anomaly_magnitude']):
        mag = abs(row['anomaly_magnitude'])
        magnitude_text = f" (deviation: {mag:.1f}%)"
    
    # Construct insight
    insight = (
        f"{risk_level}: {state} - {district} shows {classification.lower()} "
        f"{indicator} anomaly detected in {anomaly_month}"
        f"{magnitude_text}. "
        f"Persistence likelihood: {prob*100:.0f}%. "
        f"Recommendation: {response}."
    )
    
    return insight


def generate_executive_summary(anomaly_df: pd.DataFrame, 
                               district_summary: pd.DataFrame) -> str:
    """
    Generate a concise executive summary of the anomaly analysis.
    
    Args:
        anomaly_df: Classification results
        district_summary: District-level aggregations
    
    Returns:
        Multi-line string summarizing key findings
    """
    if len(anomaly_df) == 0:
        return "No anomalies detected in the current period."
    
    total_anomalies = len(anomaly_df)
    persistent_pct = (anomaly_df['classification'] == 'Likely Persistent').sum() / total_anomalies * 100
    self_correcting_pct = (anomaly_df['classification'] == 'Likely Self-Correcting').sum() / total_anomalies * 100
    
    # Top 3 highest-risk districts
    high_risk = anomaly_df[anomaly_df['persistence_probability'] >= 0.67].nlargest(
        3, 'persistence_probability'
    )
    
    summary = f"""
AADHAAR ANOMALY PERSISTENCE ANALYSIS - EXECUTIVE SUMMARY
{'='*70}

Overall Status:
- Total Active Anomalies: {total_anomalies}
- Likely Persistent: {persistent_pct:.1f}%
- Likely Self-Correcting: {self_correcting_pct:.1f}%
- Requiring Immediate Action: {len(anomaly_df[anomaly_df['persistence_probability'] >= 0.67])}

Top Priority Districts (Highest Persistence Risk):
"""
    
    if len(high_risk) > 0:
        for idx, row in high_risk.iterrows():
            summary += f"\n  • {row['state']} - {row['district']}: {row['indicator'].upper()} "
            summary += f"({row['persistence_probability']*100:.0f}% persistence likelihood)"
    else:
        summary += "\n  • No high-risk anomalies identified"
    
    summary += f"\n\nDistricts Affected: {len(district_summary)}"
    summary += f"\nAverage Persistence Probability: {anomaly_df['persistence_probability'].mean():.2f}"
    
    summary += "\n\nRecommended Actions:"
    urgent = len(anomaly_df[anomaly_df['recommended_response'].str.contains('Act')])
    prepare = len(anomaly_df[anomaly_df['recommended_response'].str.contains('Prepare')])
    monitor = len(anomaly_df[anomaly_df['recommended_response'].str.contains('Monitor')])
    
    summary += f"\n  • Immediate Intervention: {urgent} cases"
    summary += f"\n  • Prepare Response Capacity: {prepare} cases"
    summary += f"\n  • Continue Monitoring: {monitor} cases"
    
    summary += f"\n{'='*70}"
    
    return summary


# Convenience function for quick analysis
def quick_analysis(df: pd.DataFrame, 
                   baseline_months: int = 6,
                   followup_months: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    One-line function to run complete anomaly persistence analysis.
    
    Args:
        df: Input DataFrame with required columns
        baseline_months: Pre-anomaly baseline window (default: 6)
        followup_months: Post-anomaly tracking window (default: 3)
    
    Returns:
        Tuple of (classifications, district_summary, executive_summary_text)
    
    Example:
        >>> classifications, summary, text = quick_analysis(df)
        >>> print(text)
        >>> classifications.to_csv('anomaly_classifications.csv', index=False)
    """
    analyzer = AnomalyPersistenceAnalyzer(
        baseline_months=baseline_months,
        followup_months=followup_months
    )
    
    classifications, district_summary = analyzer.analyze_anomalies(df)
    executive_text = generate_executive_summary(classifications, district_summary)
    
    return classifications, district_summary, executive_text


if __name__ == "__main__":
    """
    Example usage and testing
    """
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2025-12-01', freq='MS')
    
    sample_data = []
    states = ['Karnataka', 'Maharashtra', 'Tamil Nadu']
    districts = {
        'Karnataka': ['Bangalore Urban', 'Mysore'],
        'Maharashtra': ['Mumbai', 'Pune'],
        'Tamil Nadu': ['Chennai', 'Coimbatore']
    }
    indicators = ['mpi', 'ssi']
    
    for state in states:
        for district in districts[state]:
            for indicator in indicators:
                for date in dates:
                    # Generate indicator value with occasional anomalies
                    base_value = 100 + np.random.normal(0, 5)
                    
                    # Inject some anomalies
                    is_anomaly = np.random.random() < 0.05  # 5% anomaly rate
                    if is_anomaly:
                        anomaly_mag = np.random.uniform(15, 40)
                        base_value += anomaly_mag
                    else:
                        anomaly_mag = 0
                    
                    sample_data.append({
                        'month': date,
                        'state': state,
                        'district': district,
                        'indicator': indicator,
                        'indicator_value': base_value,
                        'anomaly_flag': is_anomaly,
                        'anomaly_magnitude': anomaly_mag if is_anomaly else 0
                    })
    
    df_sample = pd.DataFrame(sample_data)
    
    # Run analysis
    print("Running Anomaly Persistence Analysis...\n")
    classifications, district_summary, executive_summary = quick_analysis(df_sample)
    
    print(executive_summary)
    print("\n\nDetailed Classifications:")
    print(classifications.head(10))
    
    print("\n\nDistrict Summary:")
    print(district_summary.head())
    
    # Test insight generation
    if len(classifications) > 0:
        print("\n\nSample Insight:")
        sample_insight = generate_insight_text(classifications.iloc[0])
        print(sample_insight)
    
    # Export results to CSV
    print("\n\nExporting results to CSV files...")
    classifications.to_csv('anomaly_classifications.csv', index=False)
    district_summary.to_csv('district_summary_report.csv', index=False)
    
    # Create detailed insights CSV
    if len(classifications) > 0:
        insights_data = []
        for idx, row in classifications.iterrows():
            insights_data.append({
                'state': row['state'],
                'district': row['district'],
                'indicator': row['indicator'],
                'anomaly_month': row['anomaly_month'],
                'insight': generate_insight_text(row)
            })
        insights_df = pd.DataFrame(insights_data)
        insights_df.to_csv('anomaly_insights.csv', index=False)
    
    # Save executive summary to text file
    with open('executive_summary.txt', 'w') as f:
        f.write(executive_summary)
    
    print("✅ Files exported successfully:")
    print("   - anomaly_classifications.csv (detailed anomaly data)")
    print("   - district_summary_report.csv (aggregated by district)")
    print("   - anomaly_insights.csv (human-readable insights)")
    print("   - executive_summary.txt (executive summary)")
