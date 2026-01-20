"""
Example: Using the Aadhaar Stress Prediction System
=====================================================

This script demonstrates how to load, aggregate, and analyze Aadhaar data
using the stress prediction system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the stress prediction module
from aadhaar_stress_prediction import (
    estimate_district_stress_likelihood,
    summarize_stress_results,
    get_district_details
)


# ============================================================================
# SECTION 1: DATA LOADING AND AGGREGATION
# ============================================================================

def load_and_aggregate_aadhaar_data(
    biometric_dir: str,
    demographic_dir: str,
    enrolment_dir: str
) -> pd.DataFrame:
    """
    Load and aggregate Aadhaar data from CSV files.
    
    This function:
    1. Loads all CSV files from each directory
    2. Concatenates them into single DataFrames per data type
    3. Merges on (month, state, district)
    4. Computes derived columns
    5. Adds anomaly flags (optional)
    
    Args:
        biometric_dir: Path to biometric data directory
        demographic_dir: Path to demographic data directory
        enrolment_dir: Path to enrolment data directory
    
    Returns:
        Aggregated DataFrame with monthly district-level metrics
    """
    
    print("Loading Aadhaar data from source files...")
    
    # Load biometric data
    print(f"  Reading biometric data from: {biometric_dir}")
    biometric_files = list(Path(biometric_dir).glob('*.csv'))
    biometric_dfs = [pd.read_csv(f) for f in sorted(biometric_files)]
    biometric_df = pd.concat(biometric_dfs, ignore_index=True)
    print(f"    Loaded {len(biometric_df)} biometric records")
    
    # Load demographic data
    print(f"  Reading demographic data from: {demographic_dir}")
    demographic_files = list(Path(demographic_dir).glob('*.csv'))
    demographic_dfs = [pd.read_csv(f) for f in sorted(demographic_files)]
    demographic_df = pd.concat(demographic_dfs, ignore_index=True)
    print(f"    Loaded {len(demographic_df)} demographic records")
    
    # Load enrolment data
    print(f"  Reading enrolment data from: {enrolment_dir}")
    enrolment_files = list(Path(enrolment_dir).glob('*.csv'))
    enrolment_dfs = [pd.read_csv(f) for f in sorted(enrolment_files)]
    enrolment_df = pd.concat(enrolment_dfs, ignore_index=True)
    print(f"    Loaded {len(enrolment_df)} enrolment records")
    
    # Convert date columns to datetime and extract month
    print("\nProcessing data...")
    biometric_df['date'] = pd.to_datetime(biometric_df['date'], format='%d-%m-%Y')
    biometric_df['month'] = biometric_df['date'].dt.to_period('M')
    
    demographic_df['date'] = pd.to_datetime(demographic_df['date'], format='%d-%m-%Y')
    demographic_df['month'] = demographic_df['date'].dt.to_period('M')
    
    enrolment_df['date'] = pd.to_datetime(enrolment_df['date'], format='%d-%m-%Y')
    enrolment_df['month'] = enrolment_df['date'].dt.to_period('M')
    
    # Aggregate at monthly district level
    print("Aggregating data at monthly district level...")
    
    # Aggregate biometric data - count rows as proxy for updates
    biometric_agg = biometric_df.groupby(['month', 'state', 'district']).size().reset_index(name='biometric_updates')
    
    # Aggregate demographic data - count rows as proxy for updates
    demographic_agg = demographic_df.groupby(['month', 'state', 'district']).size().reset_index(name='demographic_updates')
    
    # Aggregate enrolments - count rows as proxy for enrolments
    enrolment_agg = enrolment_df.groupby(['month', 'state', 'district']).size().reset_index(name='enrolments')
    
    # Merge all data on (month, state, district)
    df = enrolment_agg.copy()
    df = df.merge(demographic_agg, on=['month', 'state', 'district'], how='left')
    df = df.merge(biometric_agg, on=['month', 'state', 'district'], how='left')
    
    # Fill missing values with 0 (no updates recorded)
    df['biometric_updates'] = df['biometric_updates'].fillna(0).astype(int)
    df['demographic_updates'] = df['demographic_updates'].fillna(0).astype(int)
    df['enrolments'] = df['enrolments'].fillna(0).astype(int)
    
    print(f"  Total district-month records: {len(df)}")
    
    # ========================================================================
    # Compute derived columns
    # ========================================================================
    print("\nComputing derived metrics...")
    
    # Convert month period to timestamp for compatibility
    df['month'] = df['month'].dt.to_timestamp()
    
    # Total updates
    df['total_updates'] = df['demographic_updates'] + df['biometric_updates']
    
    # Indices (handling division by zero)
    df['ssi'] = df['total_updates'] / df['enrolments'].replace(0, np.nan)
    
    # Add placeholder for address_updates if not in data
    if 'address_updates' not in df.columns:
        df['address_updates'] = 0
    df['mpi'] = df['address_updates'] / df['enrolments'].replace(0, np.nan)
    
    # ========================================================================
    # Add anomaly flags (optional - based on statistical outliers)
    # ========================================================================
    print("Detecting anomalies...")
    
    df['anomaly_flag'] = False
    
    # Define anomalies as districts with SSI > 99th percentile or < 1st percentile
    ssi_99 = df['ssi'].quantile(0.99)
    ssi_01 = df['ssi'].quantile(0.01)
    
    df.loc[(df['ssi'] > ssi_99) | (df['ssi'] < ssi_01), 'anomaly_flag'] = True
    
    anomaly_count = df['anomaly_flag'].sum()
    print(f"  Detected {anomaly_count} anomalous records ({100*anomaly_count/len(df):.2f}%)")
    
    # Sort by month for proper time series processing
    df = df.sort_values(['state', 'district', 'month']).reset_index(drop=True)
    
    print(f"\nData loading and aggregation complete.")
    print(f"  States: {df['state'].nunique()}")
    print(f"  Districts: {df['district'].nunique()}")
    print(f"  Time periods: {df['month'].min()} to {df['month'].max()}")
    
    return df


# ============================================================================
# SECTION 2: ANALYSIS AND REPORTING
# ============================================================================

def generate_compact_reports(
    results_df: pd.DataFrame,
    output_dir: str
) -> dict:
    """
    Generate compact, actionable analysis reports.
    
    Produces:
    1. Risk summary table (compact overview)
    2. Top 10 high-risk districts
    3. State-level risk distribution
    
    Args:
        results_df: Output from estimate_district_stress_likelihood()
        output_dir: Directory to save CSV outputs
    
    Returns:
        Dictionary containing all report DataFrames
    """
    
    print("\n" + "="*70)
    print("AADHAAR STRESS PREDICTION REPORT (COMPACT)")
    print("="*70)
    
    # ========================================================================
    # REPORT 1: Risk Summary Table
    # ========================================================================
    risk_summary = pd.DataFrame({
        'Risk_Level': ['High', 'Medium', 'Low', 'Total'],
        'District_Count': [
            (results_df['risk_level'] == 'High').sum(),
            (results_df['risk_level'] == 'Medium').sum(),
            (results_df['risk_level'] == 'Low').sum(),
            len(results_df)
        ]
    })
    risk_summary['Percentage'] = (risk_summary['District_Count'] / len(results_df) * 100).round(1)
    risk_summary['Avg_Stress_Likelihood'] = [
        results_df[results_df['risk_level'] == 'High']['stress_likelihood'].mean(),
        results_df[results_df['risk_level'] == 'Medium']['stress_likelihood'].mean(),
        results_df[results_df['risk_level'] == 'Low']['stress_likelihood'].mean(),
        results_df['stress_likelihood'].mean()
    ]
    risk_summary['Avg_Stress_Likelihood'] = risk_summary['Avg_Stress_Likelihood'].round(4)
    
    print("\n1. RISK SUMMARY TABLE:")
    print(risk_summary.to_string(index=False))
    
    # ========================================================================
    # REPORT 2: Top 10 High-Risk Districts
    # ========================================================================
    top_10_high_risk = results_df.nlargest(10, 'stress_likelihood')[
        ['state', 'district', 'stress_likelihood', 'normal_capacity', 
         'predicted_updates_next_month', 'risk_level']
    ].copy()
    top_10_high_risk.columns = ['State', 'District', 'Stress_Likelihood', 
                                  'Normal_Capacity', 'Predicted_Updates', 'Risk_Level']
    top_10_high_risk = top_10_high_risk.reset_index(drop=True)
    top_10_high_risk.index = top_10_high_risk.index + 1
    
    print("\n2. TOP 10 HIGH-RISK DISTRICTS:")
    print(top_10_high_risk.to_string())
    
    # ========================================================================
    # REPORT 3: State-Level Risk Distribution
    # ========================================================================
    state_risk_dist = results_df.groupby(['state', 'risk_level']).size().unstack(fill_value=0)
    state_risk_dist['Total'] = state_risk_dist.sum(axis=1)
    state_risk_dist = state_risk_dist.sort_values('High', ascending=False)
    
    # Add percentage columns
    for col in ['High', 'Medium', 'Low']:
        if col in state_risk_dist.columns:
            state_risk_dist[f'{col}_%'] = (state_risk_dist[col] / state_risk_dist['Total'] * 100).round(1)
    
    # Calculate average stress likelihood per state
    state_avg_stress = results_df.groupby('state')['stress_likelihood'].mean().round(4)
    state_risk_dist['Avg_Stress_Likelihood'] = state_avg_stress
    
    # Reorder columns for readability
    col_order = ['Total', 'High', 'High_%', 'Medium', 'Medium_%', 'Low', 'Low_%', 'Avg_Stress_Likelihood']
    col_order = [c for c in col_order if c in state_risk_dist.columns]
    state_risk_dist = state_risk_dist[col_order]
    state_risk_dist = state_risk_dist.reset_index()
    state_risk_dist.columns.name = None
    state_risk_dist = state_risk_dist.rename(columns={'state': 'State'})
    
    print("\n3. STATE-LEVEL RISK DISTRIBUTION (Top 15 by High-Risk Count):")
    print(state_risk_dist.head(15).to_string(index=False))
    
    # ========================================================================
    # Save Reports to CSV
    # ========================================================================
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    risk_summary_path = f"{output_dir}/risk_summary.csv"
    top_10_path = f"{output_dir}/top_10_high_risk_districts.csv"
    state_dist_path = f"{output_dir}/state_risk_distribution.csv"
    
    risk_summary.to_csv(risk_summary_path, index=False)
    top_10_high_risk.to_csv(top_10_path, index=True, index_label='Rank')
    state_risk_dist.to_csv(state_dist_path, index=False)
    
    print(f"\n✓ Reports saved:")
    print(f"  - {risk_summary_path}")
    print(f"  - {top_10_path}")
    print(f"  - {state_dist_path}")
    
    print("\n" + "="*70)
    
    # Return all reports as dictionary
    return {
        'risk_summary': risk_summary,
        'top_10_high_risk': top_10_high_risk,
        'state_risk_distribution': state_risk_dist
    }


def detailed_district_analysis(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    state: str,
    district: str
) -> None:
    """
    Print detailed time-series analysis for a specific district.
    
    Args:
        df: Original aggregated DataFrame
        results_df: Results from stress prediction
        state: State name
        district: District name
    """
    
    # Get prediction result
    pred = get_district_details(results_df, state, district)
    
    if pred is None:
        print(f"District '{district}' in '{state}' not found in results.")
        return
    
    # Get time series data
    district_data = df[
        (df['state'] == state) & (df['district'] == district)
    ].sort_values('month')
    
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS: {state} / {district}")
    print(f"{'='*70}")
    
    print(f"\nPREDICTION SUMMARY:")
    print(f"  Risk Level:                   {pred['risk_level']}")
    print(f"  Stress Likelihood:            {pred['stress_likelihood']:.4f}")
    print(f"  Normal Capacity (75th %ile):  {pred['normal_capacity']:.0f} updates/month")
    print(f"  Predicted Next Month Updates: {pred['predicted_updates_next_month']:.0f}")
    print(f"  Reference Month:              {pred['reference_month']}")
    
    print(f"\nLAST 12 MONTHS ACTIVITY:")
    print(f"{'Month':<12} {'Enrolments':<12} {'Total Updates':<15} {'SSI':<12} {'Anomaly':<8}")
    print("-" * 70)
    
    for _, row in district_data.tail(12).iterrows():
        ssi = row['ssi'] if pd.notna(row['ssi']) else 0
        anomaly = "Yes" if row['anomaly_flag'] else "No"
        month_str = row['month'].strftime('%Y-%m')
        print(f"{month_str:<12} {row['enrolments']:<12.0f} {row['total_updates']:<15.0f} {ssi:<12.4f} {anomaly:<8}")
    
    # Compute growth stats
    if len(district_data) >= 2:
        recent_growth = (
            (district_data['total_updates'].iloc[-1] - district_data['total_updates'].iloc[-2]) /
            district_data['total_updates'].iloc[-2]
        ) * 100 if district_data['total_updates'].iloc[-2] > 0 else 0
        
        avg_updates = district_data['total_updates'].mean()
        std_updates = district_data['total_updates'].std()
        
        print(f"\nGROWTH & VOLATILITY:")
        print(f"  Recent Growth (MoM):          {recent_growth:+.2f}%")
        print(f"  Average Monthly Updates:      {avg_updates:.0f}")
        print(f"  Update Volatility (Std Dev):  {std_updates:.0f}")
        print(f"  Coefficient of Variation:     {std_updates/avg_updates if avg_updates > 0 else 0:.4f}")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# SECTION 3: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Define data directories
    BASE_DIR = "/Users/abhinavsudhi/Downloads/DESKTOP2/ML"
    BIOMETRIC_DIR = f"{BASE_DIR}/api_data_aadhar_biometric"
    DEMOGRAPHIC_DIR = f"{BASE_DIR}/api_data_aadhar_demographic"
    ENROLMENT_DIR = f"{BASE_DIR}/api_data_aadhar_enrolment"
    
    # ========================================================================
    # Step 1: Load and aggregate data
    # ========================================================================
    try:
        df = load_and_aggregate_aadhaar_data(
            BIOMETRIC_DIR,
            DEMOGRAPHIC_DIR,
            ENROLMENT_DIR
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Display data overview
    print("\nDATA OVERVIEW:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # ========================================================================
    # Step 2: Run stress prediction
    # ========================================================================
    print("\n" + "="*70)
    print("RUNNING STRESS PREDICTION ANALYSIS")
    print("="*70)
    
    try:
        results_df = estimate_district_stress_likelihood(
            df,
            growth_threshold=0.20,
            window_months=12,
            percentile=75.0,
            lookahead_months=1
        )
    except Exception as e:
        print(f"Error running stress prediction: {e}")
        sys.exit(1)
    
    # ========================================================================
    # Step 3: Generate compact reports
    # ========================================================================
    
    # Generate compact, actionable reports
    reports = generate_compact_reports(
        results_df,
        output_dir=BASE_DIR
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✓ {len(results_df)} districts analyzed")
    print(f"✓ {reports['top_10_high_risk'].shape[0]} top high-risk districts identified")
    print(f"✓ {reports['state_risk_distribution'].shape[0]} states covered")
    print("\n✓ All reports generated successfully!")
