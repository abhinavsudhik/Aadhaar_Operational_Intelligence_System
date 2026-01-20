"""
Resource Adequacy Assessment Module

Evaluates whether districts are under-resourced, adequately resourced, or 
well-resourced based on Aadhaar update activity patterns and stability.

Designed for policymakers and system planners to identify capacity constraints
and operational readiness for demand absorption.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime


def assess_resource_adequacy(
    df: pd.DataFrame,
    lookback_months: int = 6,
    demand_absorption_threshold: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assess resource adequacy for each district based on update activity metrics.
    
    This function evaluates operational capacity by analyzing:
    - Update volume and consistency (stability of operations)
    - Anomaly resolution effectiveness (operational maturity)
    - Capacity to absorb demand increase (forward-looking resilience)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns:
        - month (datetime or YYYY-MM string, must be sortable)
        - state (str)
        - district (str)
        - demographic_updates (int)
        - biometric_updates (int)
        - enrolments (int)
        - anomaly_flag (bool): True if month has anomalies
        - anomaly_persistent (bool): True if anomalies continued from prior month
        
        Derived columns (if missing, will be computed):
        - total_updates = demographic_updates + biometric_updates
        - update_rate = total_updates / enrolments
    
    lookback_months : int, default=6
        Number of recent months to use for metrics (typically 6 for stability)
    
    demand_absorption_threshold : float, default=0.15
        Percentage increase in demand to assess capacity (15% = 0.15)
    
    Returns
    -------
    district_results : pd.DataFrame
        ONE ROW PER DISTRICT with columns:
        - state
        - district
        - avg_monthly_updates (float): Average total_updates over lookback period
        - update_stability_score (0-100): Lower volatility = higher score
        - anomaly_resolution_rate (0-1): Fraction of anomalies resolved next month
        - resource_adequacy_score (0-100): Composite score for resource capacity
        - capacity_classification (str): "Under-Resourced", "Adequately Resourced", 
                                         or "Well Resourced"
        - absorption_capability (str): "Yes" if can handle demand increase, "At Risk"
        - explanatory_note (str): 1-2 line human-readable explanation
    
    state_summary : pd.DataFrame
        State-level rollup with columns:
        - state
        - total_districts
        - pct_under_resourced
        - pct_adequately_resourced
        - pct_well_resourced
        - avg_adequacy_score
    
    Notes
    -----
    - Returns one row per district (no temporal dimension in output)
    - Handles missing months and districts with sparse data gracefully
    - All metrics normalized to 0-100 for interpretability
    - Suitable for dashboards and policy briefings
    """
    
    # =========================================================================
    # 1. DATA PREPARATION AND VALIDATION
    # =========================================================================
    
    # Create working copy
    df = df.copy()
    
    # Convert month to datetime if string
    if df['month'].dtype == 'object':
        df['month'] = pd.to_datetime(df['month'])
    
    # Compute derived columns if missing
    if 'total_updates' not in df.columns:
        df['total_updates'] = df['demographic_updates'] + df['biometric_updates']
    
    if 'update_rate' not in df.columns:
        df['update_rate'] = df['total_updates'] / df['enrolments'].replace(0, np.nan)
    
    # Ensure boolean columns exist (fill with False if missing for safety)
    if 'anomaly_flag' not in df.columns:
        df['anomaly_flag'] = False
    if 'anomaly_persistent' not in df.columns:
        df['anomaly_persistent'] = False
    
    # Sort by district and month for stable operations
    df = df.sort_values(['state', 'district', 'month']).reset_index(drop=True)
    
    # =========================================================================
    # 2. DISTRICT-LEVEL METRIC COMPUTATION
    # =========================================================================
    
    def compute_district_metrics(group: pd.DataFrame) -> Dict:
        """
        Compute resource adequacy metrics for a single district.
        
        Parameters
        ----------
        group : pd.DataFrame
            All records for one district (sorted by month)
        
        Returns
        -------
        dict : Metrics for resource assessment
        """
        
        # Get recent lookback_months of data
        group = group.sort_values('month').tail(lookback_months)
        
        if len(group) < 2:
            # Insufficient data: return neutral metrics
            return {
                'avg_monthly_updates': group['total_updates'].mean() if len(group) > 0 else 0,
                'update_stability_score': 50,  # Neutral
                'anomaly_resolution_rate': 0.5,  # Unknown
                'data_completeness': len(group) / lookback_months,
            }
        
        # --- Metric 1: Update Volume ---
        # Higher average activity suggests better operational capacity
        avg_updates = group['total_updates'].mean()
        
        # --- Metric 2: Update Stability (Coefficient of Variation) ---
        # Lower volatility indicates mature, reliable operations
        std_updates = group['total_updates'].std()
        if std_updates == 0 or avg_updates == 0:
            # No variation or no activity
            cv = 0
        else:
            cv = std_updates / avg_updates
        
        # Convert CV to stability score (0-100): lower CV = higher score
        # CV > 1 = unstable (score ~0), CV ~0.1-0.3 = stable (score ~80-90)
        stability_score = max(0, min(100, 100 * np.exp(-2 * cv)))
        
        # --- Metric 3: Anomaly Resolution Rate ---
        # Higher resolution rate = more operational responsiveness
        anomalies_detected = group['anomaly_flag'].sum()
        if anomalies_detected > 0:
            # Check if anomalies were resolved in following months
            # (anomaly_persistent = False indicates resolution)
            persistent_count = group['anomaly_persistent'].sum()
            resolution_rate = 1.0 - (persistent_count / anomalies_detected)
        else:
            # No anomalies detected = best case
            resolution_rate = 1.0
        
        # --- Metric 4: Data Completeness ---
        # Penalize sparse data (incomplete records indicate operational challenges)
        data_completeness = len(group) / lookback_months
        
        return {
            'avg_monthly_updates': avg_updates,
            'update_stability_score': stability_score,
            'anomaly_resolution_rate': resolution_rate,
            'data_completeness': data_completeness,
        }
    
    # Compute metrics for each district
    # Compute metrics for each district and expand dict into columns
    metrics_series = (
        df.groupby(['state', 'district'])
        .apply(compute_district_metrics, include_groups=False)
    )
    metrics_expanded = metrics_series.apply(pd.Series).reset_index()
    district_metrics = metrics_expanded
    
    # =========================================================================
    # 3. NORMALIZATION OF METRICS (0-100 SCALE)
    # =========================================================================
    
    def normalize_metric(series: pd.Series, percentile_range: Tuple[float, float] = (5, 95)) -> pd.Series:
        """
        Normalize a metric to 0-100 scale using percentile-based scaling.
        
        This prevents outliers from compressing the scale for typical districts.
        """
        p_low = series.quantile(percentile_range[0] / 100)
        p_high = series.quantile(percentile_range[1] / 100)
        
        if p_high == p_low:
            # No variation in percentile range
            return pd.Series(50, index=series.index)
        
        normalized = ((series - p_low) / (p_high - p_low)) * 100
        return normalized.clip(0, 100)
    
    # Normalize update volume
    if district_metrics['avg_monthly_updates'].max() > 0:
        district_metrics['update_volume_score'] = normalize_metric(
            district_metrics['avg_monthly_updates']
        )
    else:
        district_metrics['update_volume_score'] = 0
    
    # Stability score is already normalized
    district_metrics['stability_score'] = district_metrics['update_stability_score']
    
    # Normalize anomaly resolution rate (0-1 -> 0-100)
    district_metrics['anomaly_resolution_score'] = (
        district_metrics['anomaly_resolution_rate'] * 100
    )
    
    # Normalize data completeness (0-1 -> 0-100)
    district_metrics['completeness_score'] = (
        district_metrics['data_completeness'] * 100
    )
    
    # =========================================================================
    # 4. COMPOSITE RESOURCE ADEQUACY SCORE
    # =========================================================================
    
    # Weight components: volume (40%), stability (35%), resolution (15%), completeness (10%)
    # Weights reflect: volume & stability are operational foundations,
    # anomaly resolution is important but less critical, completeness is hygiene factor
    
    district_metrics['resource_adequacy_score'] = (
        0.40 * district_metrics['update_volume_score'] +
        0.35 * district_metrics['stability_score'] +
        0.15 * district_metrics['anomaly_resolution_score'] +
        0.10 * district_metrics['completeness_score']
    ).round(1)
    
    # =========================================================================
    # 5. CAPACITY CLASSIFICATION AND DEMAND ABSORPTION
    # =========================================================================
    
    def classify_capacity(score: float) -> str:
        """Classify district based on adequacy score."""
        if score < 35:
            return "Under-Resourced"
        elif score < 65:
            return "Adequately Resourced"
        else:
            return "Well Resourced"
    
    def assess_absorption_capability(row: pd.Series, threshold: float = 0.15) -> str:
        """
        Assess if district can absorb 15-20% demand increase.
        
        Districts with:
        - High volume + high stability + good resolution = "Yes"
        - Otherwise = "At Risk" (needs monitoring or reinforcement)
        """
        # Can absorb if score is "good" and stability is not concerning
        can_absorb = (
            row['resource_adequacy_score'] >= 65 and
            row['stability_score'] >= 60
        )
        return "Yes" if can_absorb else "At Risk"
    
    district_metrics['capacity_classification'] = (
        district_metrics['resource_adequacy_score'].apply(classify_capacity)
    )
    
    district_metrics['absorption_capability'] = (
        district_metrics.apply(
            lambda row: assess_absorption_capability(row, threshold=demand_absorption_threshold),
            axis=1
        )
    )
    
    # =========================================================================
    # 6. GENERATE EXPLANATORY NOTES
    # =========================================================================
    
    def generate_explanation(row: pd.Series) -> str:
        """Create 1-2 line human-readable explanation."""
        
        score = row['resource_adequacy_score']
        stability = row['stability_score']
        volume = row['avg_monthly_updates']
        
        # Build explanation based on profile
        if row['capacity_classification'] == "Well Resourced":
            explanation = f"Strong operational capacity with {volume:.0f} avg monthly updates and {stability:.0f}/100 stability. "
            if row['absorption_capability'] == "Yes":
                explanation += "Ready for demand growth."
            else:
                explanation += "Monitor for demand surge."
        
        elif row['capacity_classification'] == "Adequately Resourced":
            explanation = f"Stable operations ({volume:.0f} avg updates) but moderate growth headroom. "
            if row['anomaly_resolution_score'] < 70:
                explanation += "Improve anomaly response."
            else:
                explanation += "Monitor capacity trends."
        
        else:  # Under-Resourced
            explanation = f"Limited update activity ({volume:.0f}/month) limits current capacity. "
            if row['stability_score'] < 40:
                explanation += "High volatility suggests process issues."
            else:
                explanation += "Requires staffing or process review."
        
        return explanation.strip()
    
    district_metrics['explanatory_note'] = district_metrics.apply(generate_explanation, axis=1)
    
    # =========================================================================
    # 7. FORMAT FINAL DISTRICT OUTPUT
    # =========================================================================
    
    district_results = district_metrics[[
        'state',
        'district',
        'avg_monthly_updates',
        'update_stability_score',
        'anomaly_resolution_rate',
        'resource_adequacy_score',
        'capacity_classification',
        'absorption_capability',
        'explanatory_note'
    ]].copy()
    
    # Rename for clarity in output
    district_results.columns = [
        'state',
        'district',
        'avg_monthly_updates',
        'update_stability_score',
        'anomaly_resolution_rate',
        'resource_adequacy_score',
        'capacity_classification',
        'absorption_capability',
        'explanatory_note'
    ]
    
    # =========================================================================
    # 8. STATE-LEVEL SUMMARY
    # =========================================================================
    
    state_summary = (
        district_results
        .groupby('state')
        .agg(
            total_districts=('district', 'count'),
            pct_under_resourced=(
                'capacity_classification',
                lambda x: (x == "Under-Resourced").sum() / len(x) * 100
            ),
            pct_adequately_resourced=(
                'capacity_classification',
                lambda x: (x == "Adequately Resourced").sum() / len(x) * 100
            ),
            pct_well_resourced=(
                'capacity_classification',
                lambda x: (x == "Well Resourced").sum() / len(x) * 100
            ),
            avg_adequacy_score=('resource_adequacy_score', 'mean'),
        )
        .round(2)
        .reset_index()
    )
    
    # =========================================================================
    # 9. RETURN RESULTS
    # =========================================================================
    
    return district_results, state_summary


def generate_resource_adequacy_report(
    district_results: pd.DataFrame,
    state_summary: pd.DataFrame,
    output_csv: str = "resource_adequacy_assessment.csv",
    output_summary_csv: str = "resource_adequacy_state_summary.csv"
) -> None:
    """
    Export resource adequacy results to CSV files for dashboards and reports.
    
    Parameters
    ----------
    district_results : pd.DataFrame
        District-level assessment results
    state_summary : pd.DataFrame
        State-level summary statistics
    output_csv : str
        Path for district-level results CSV
    output_summary_csv : str
        Path for state-level summary CSV
    """
    
    district_results.to_csv(output_csv, index=False)
    print(f"✓ District results exported: {output_csv}")
    
    state_summary.to_csv(output_summary_csv, index=False)
    print(f"✓ State summary exported: {output_summary_csv}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("RESOURCE ADEQUACY ASSESSMENT SUMMARY")
    print("=" * 80)
    print(f"\nState-Level Overview:")
    print(state_summary.to_string(index=False))
    
    print(f"\n\nDistrict Classification Breakdown:")
    classification_summary = (
        district_results['capacity_classification']
        .value_counts()
        .sort_index()
    )
    for classification, count in classification_summary.items():
        pct = (count / len(district_results)) * 100
        print(f"  {classification:25s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n\nAbsorption Capability:")
    absorption_summary = (
        district_results['absorption_capability']
        .value_counts()
    )
    for capability, count in absorption_summary.items():
        pct = (count / len(district_results)) * 100
        print(f"  {capability:25s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n\nTop 5 Well-Resourced Districts:")
    top_well = (
        district_results[
            district_results['capacity_classification'] == "Well Resourced"
        ]
        .nlargest(5, 'resource_adequacy_score')[
            ['state', 'district', 'resource_adequacy_score', 'absorption_capability']
        ]
    )
    if len(top_well) > 0:
        print(top_well.to_string(index=False))
    else:
        print("  (No well-resourced districts)")
    
    print(f"\n\nTop 5 Under-Resourced Districts:")
    under = (
        district_results[
            district_results['capacity_classification'] == "Under-Resourced"
        ]
        .nsmallest(5, 'resource_adequacy_score')[
            ['state', 'district', 'resource_adequacy_score', 'absorption_capability']
        ]
    )
    if len(under) > 0:
        print(under.to_string(index=False))
    else:
        print("  (All districts adequately or well resourced)")
    
    print("\n" + "=" * 80)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Run resource adequacy assessment on Aadhaar data.
    
    Expected input DataFrame structure:
    - month: datetime (YYYY-MM or datetime object)
    - state: string
    - district: string
    - demographic_updates: int
    - biometric_updates: int
    - enrolments: int
    - anomaly_flag: boolean
    - anomaly_persistent: boolean
    """
    
    # Load sample data (replace with your actual data source)
    # df = pd.read_csv('your_aadhaar_data.csv')
    
    # Run assessment
    # district_results, state_summary = assess_resource_adequacy(
    #     df, 
    #     lookback_months=6,
    #     demand_absorption_threshold=0.15
    # )
    
    # Export results
    # generate_resource_adequacy_report(
    #     district_results,
    #     state_summary,
    #     output_csv="resource_adequacy_assessment.csv",
    #     output_summary_csv="resource_adequacy_state_summary.csv"
    # )
    
    print("Resource adequacy assessment module ready.")
    print("Call assess_resource_adequacy(df) with your Aadhaar data.")
