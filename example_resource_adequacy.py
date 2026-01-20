"""
Example: Resource Adequacy Assessment for Aadhaar Districts

Demonstrates how to integrate the resource adequacy module with existing
Aadhaar data pipeline to generate operational capacity assessments.

Usage:
    python example_resource_adequacy.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from resource_adequacy_assessment import (
    assess_resource_adequacy,
    generate_resource_adequacy_report
)


def load_aadhaar_data_with_derived_columns() -> pd.DataFrame:
    """
    Load and consolidate Aadhaar data from multiple sources.
    
    Combines demographic, biometric, and enrolment data by month and district.
    Computes derived metrics for resource assessment.
    
    Returns
    -------
    pd.DataFrame
        Consolidated monthly district-level Aadhaar activity with computed metrics
    """
    
    print("=" * 80)
    print("LOADING AADHAAR DATA")
    print("=" * 80)
    
    base_path = Path(__file__).parent
    
    # =========================================================================
    # Load demographic updates
    # =========================================================================
    print("\nLoading demographic data...")
    demo_files = sorted((base_path / "api_data_aadhar_demographic").glob("*.csv"))
    demo_dfs = []
    for file in demo_files:
        try:
            df = pd.read_csv(file)
            demo_dfs.append(df)
            print(f"  ✓ {file.name}: {len(df):,} records")
        except Exception as e:
            print(f"  ✗ {file.name}: {e}")
    
    if demo_dfs:
        demographic_df = pd.concat(demo_dfs, ignore_index=True)
        print(f"  Total demographic records: {len(demographic_df):,}\n")
    else:
        print("  No demographic files found")
        demographic_df = None
    
    # =========================================================================
    # Load biometric updates
    # =========================================================================
    print("Loading biometric data...")
    bio_files = sorted((base_path / "api_data_aadhar_biometric").glob("*.csv"))
    bio_dfs = []
    for file in bio_files:
        try:
            df = pd.read_csv(file)
            bio_dfs.append(df)
            print(f"  ✓ {file.name}: {len(df):,} records")
        except Exception as e:
            print(f"  ✗ {file.name}: {e}")
    
    if bio_dfs:
        biometric_df = pd.concat(bio_dfs, ignore_index=True)
        print(f"  Total biometric records: {len(biometric_df):,}\n")
    else:
        print("  No biometric files found")
        biometric_df = None
    
    # =========================================================================
    # Load enrolment data
    # =========================================================================
    print("Loading enrolment data...")
    enrol_files = sorted((base_path / "api_data_aadhar_enrolment").glob("*.csv"))
    enrol_dfs = []
    for file in enrol_files:
        try:
            df = pd.read_csv(file)
            enrol_dfs.append(df)
            print(f"  ✓ {file.name}: {len(df):,} records")
        except Exception as e:
            print(f"  ✗ {file.name}: {e}")
    
    if enrol_dfs:
        enrolment_df = pd.concat(enrol_dfs, ignore_index=True)
        print(f"  Total enrolment records: {len(enrolment_df):,}\n")
    else:
        print("  No enrolment files found")
        enrolment_df = None
    
    # =========================================================================
    # Aggregate by month, state, district
    # =========================================================================
    print("Aggregating to monthly district level...")
    
    # For demographic: sum all age group columns as activity counts
    if demographic_df is not None:
        demo_counts = demographic_df.copy()
        demo_counts['date'] = pd.to_datetime(demo_counts['date'], format='%d-%m-%Y', errors='coerce')
        demo_counts['month'] = demo_counts['date'].dt.to_period('M')
        # Get all numeric columns except date-based ones
        numeric_cols = [c for c in demo_counts.columns if c not in ['date', 'month', 'state', 'district', 'pincode']]
        demo_agg = (
            demo_counts
            .groupby(['month', 'state', 'district'])
            [numeric_cols]
            .sum()
            .sum(axis=1)
            .reset_index(name='demographic_updates')
        )
    else:
        demo_agg = None
    
    # For biometric: similar aggregation
    if biometric_df is not None:
        bio_counts = biometric_df.copy()
        bio_counts['date'] = pd.to_datetime(bio_counts['date'], format='%d-%m-%Y', errors='coerce')
        bio_counts['month'] = bio_counts['date'].dt.to_period('M')
        numeric_cols = [c for c in bio_counts.columns if c not in ['date', 'month', 'state', 'district', 'pincode']]
        bio_agg = (
            bio_counts
            .groupby(['month', 'state', 'district'])
            [numeric_cols]
            .sum()
            .sum(axis=1)
            .reset_index(name='biometric_updates')
        )
    else:
        bio_agg = None
    
    # For enrolment: similar aggregation
    if enrolment_df is not None:
        enrol_counts = enrolment_df.copy()
        enrol_counts['date'] = pd.to_datetime(enrol_counts['date'], format='%d-%m-%Y', errors='coerce')
        enrol_counts['month'] = enrol_counts['date'].dt.to_period('M')
        numeric_cols = [c for c in enrol_counts.columns if c not in ['date', 'month', 'state', 'district', 'pincode']]
        enrol_agg = (
            enrol_counts
            .groupby(['month', 'state', 'district'])
            [numeric_cols]
            .sum()
            .sum(axis=1)
            .reset_index(name='enrolments')
        )
    else:
        enrol_agg = None
    
    # =========================================================================
    # Merge datasets
    # =========================================================================
    print("Merging datasets by (month, state, district)...")
    
    if demo_agg is not None:
        merged = demo_agg.copy()
        print(f"  Starting with {len(merged):,} month-state-district records (demographic)")
    else:
        merged = None
    
    if bio_agg is not None:
        if merged is None:
            merged = bio_agg.copy()
            print(f"  Starting with {len(merged):,} month-state-district records (biometric)")
        else:
            merged = merged.merge(
                bio_agg,
                on=['month', 'state', 'district'],
                how='outer'
            )
            print(f"  After biometric merge: {len(merged):,} records")
    
    if enrol_agg is not None:
        if merged is None:
            merged = enrol_agg.copy()
            print(f"  Starting with {len(merged):,} month-state-district records (enrolment)")
        else:
            merged = merged.merge(
                enrol_agg,
                on=['month', 'state', 'district'],
                how='outer'
            )
            print(f"  After enrolment merge: {len(merged):,} records")
    
    if merged is None:
        raise ValueError("No data loaded. Check CSV file paths.")
    
    # =========================================================================
    # Fill missing values and compute derived metrics
    # =========================================================================
    print("\nComputing derived metrics...")
    
    # Fill NaNs with 0 for update counts
    merged['demographic_updates'] = merged.get('demographic_updates', 0).fillna(0).astype(int)
    merged['biometric_updates'] = merged.get('biometric_updates', 0).fillna(0).astype(int)
    merged['enrolments'] = merged.get('enrolments', 0).fillna(0).astype(int)
    
    # Convert month to datetime for consistency
    merged['month'] = merged['month'].dt.to_timestamp()
    
    # Compute total updates
    merged['total_updates'] = merged['demographic_updates'] + merged['biometric_updates']
    
    # Compute update rate (prevent division by zero)
    merged['update_rate'] = np.where(
        merged['enrolments'] > 0,
        merged['total_updates'] / merged['enrolments'],
        0
    )
    
    # =========================================================================
    # Detect anomalies
    # =========================================================================
    print("Detecting anomalies...")
    
    def detect_anomalies(group):
        """Detect anomalies for a single district using IQR method."""
        # Use total_updates for anomaly detection
        Q1 = group['total_updates'].quantile(0.25)
        Q3 = group['total_updates'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Anomaly flag: outside normal range
        group['anomaly_flag'] = (
            (group['total_updates'] < lower_bound) |
            (group['total_updates'] > upper_bound)
        )
        
        # Anomaly persistent: same month or immediately following another anomaly
        group['anomaly_persistent'] = (
            group['anomaly_flag'].shift(1) & group['anomaly_flag']
        )
        group['anomaly_persistent'] = group['anomaly_persistent'].fillna(False)
        
        return group
    
    # Keep group columns during apply (drop include_groups=False to retain state/district)
    merged = (
        merged
        .sort_values(['state', 'district', 'month'])
        .groupby(['state', 'district'], group_keys=False)
        .apply(detect_anomalies)
        .reset_index()
    )
    
    # =========================================================================
    # Sort and validate
    # =========================================================================
    merged = merged.sort_values(['month', 'state', 'district']).reset_index(drop=True)
    
    # Ensure state and district columns exist after apply
    if 'state' not in merged.columns or 'district' not in merged.columns:
        raise ValueError("State and district columns were lost during anomaly detection")
    print(f"\n✓ Consolidated dataset: {len(merged):,} records")
    print(f"  Months: {merged['month'].min().strftime('%Y-%m')} to {merged['month'].max().strftime('%Y-%m')}")
    print(f"  States: {merged['state'].nunique()}")
    print(f"  Districts: {merged['district'].nunique()}")
    print(f"  Anomalies detected: {merged['anomaly_flag'].sum():,}")
    
    return merged


def main():
    """
    Execute complete resource adequacy assessment pipeline.
    """
    
    # =========================================================================
    # STEP 1: Load and prepare data
    # =========================================================================
    try:
        df = load_aadhaar_data_with_derived_columns()
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # STEP 2: Run resource adequacy assessment
    # =========================================================================
    print("\n" + "=" * 80)
    print("ASSESSING RESOURCE ADEQUACY")
    print("=" * 80)
    
    try:
        district_results, state_summary = assess_resource_adequacy(
            df,
            lookback_months=6,
            demand_absorption_threshold=0.15
        )
        print("\n✓ Assessment complete")
    except Exception as e:
        print(f"\n✗ Error during assessment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # STEP 3: Generate and export report
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    
    try:
        generate_resource_adequacy_report(
            district_results,
            state_summary,
            output_csv="resource_adequacy_assessment.csv",
            output_summary_csv="resource_adequacy_state_summary.csv"
        )
    except Exception as e:
        print(f"\n✗ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # STEP 4: Additional analysis for policymakers
    # =========================================================================
    print("\n" + "=" * 80)
    print("CAPACITY PLANNING INSIGHTS")
    print("=" * 80)
    
    # Identify districts at risk of congestion
    at_risk = district_results[
        district_results['absorption_capability'] == "At Risk"
    ].sort_values('resource_adequacy_score')
    
    print(f"\nDistricts At Risk for 15% Demand Increase: {len(at_risk)}")
    if len(at_risk) > 0:
        print("\nTop concerns (lowest adequacy scores):")
        print(at_risk[
            ['state', 'district', 'resource_adequacy_score', 'capacity_classification']
        ].head(10).to_string(index=False))
    
    # Identify high-capacity districts
    high_capacity = district_results[
        (district_results['capacity_classification'] == "Well Resourced") &
        (district_results['absorption_capability'] == "Yes")
    ].sort_values('resource_adequacy_score', ascending=False)
    
    print(f"\n\nHigh-Capacity Districts (Ready for Growth): {len(high_capacity)}")
    if len(high_capacity) > 0:
        print("\nTop performers (highest adequacy scores):")
        print(high_capacity[
            ['state', 'district', 'resource_adequacy_score', 'avg_monthly_updates']
        ].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Assessment complete. Results exported to CSV files.")
    print("=" * 80)


if __name__ == "__main__":
    main()
