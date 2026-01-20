"""
Demand Elasticity Analysis - Complete Workflow Example
========================================================

This script demonstrates end-to-end usage of the elasticity analyzer with
real Aadhaar data, including data preparation, analysis, and report generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import the elasticity analyzer
from demand_elasticity_analyzer import analyze_demand_elasticity


def load_and_prepare_data(
    biometric_dir: str = 'api_data_aadhar_biometric',
    demographic_dir: str = 'api_data_aadhar_demographic',
    enrolment_dir: str = 'api_data_aadhar_enrolment'
) -> pd.DataFrame:
    """
    Load and combine multiple CSV files from API data directories.
    
    Args:
        biometric_dir: Directory containing biometric update CSVs
        demographic_dir: Directory containing demographic update CSVs
        enrolment_dir: Directory containing enrolment CSVs
    
    Returns:
        Combined and prepared DataFrame
    """
    
    # Load biometric data
    print("Loading biometric data...")
    biometric_files = sorted(Path(biometric_dir).glob('*.csv'))
    biometric_dfs = [pd.read_csv(f) for f in biometric_files]
    biometric_df = pd.concat(biometric_dfs, ignore_index=True)
    print(f"  Loaded {len(biometric_df)} biometric records")
    
    # Load demographic data
    print("Loading demographic data...")
    demographic_files = sorted(Path(demographic_dir).glob('*.csv'))
    demographic_dfs = [pd.read_csv(f) for f in demographic_files]
    demographic_df = pd.concat(demographic_dfs, ignore_index=True)
    print(f"  Loaded {len(demographic_df)} demographic records")
    
    # Load enrolment data
    print("Loading enrolment data...")
    enrolment_files = sorted(Path(enrolment_dir).glob('*.csv'))
    enrolment_dfs = [pd.read_csv(f) for f in enrolment_files]
    enrolment_df = pd.concat(enrolment_dfs, ignore_index=True)
    print(f"  Loaded {len(enrolment_df)} enrolment records")
    
    # Aggregate to monthly district level
    print("\nAggregating to monthly district level...")
    
    # Prepare biometric aggregates
    if 'date' in biometric_df.columns:
        biometric_df['month'] = pd.to_datetime(biometric_df['date']).dt.to_period('M')
    elif 'month' in biometric_df.columns:
        biometric_df['month'] = pd.to_datetime(biometric_df['month']).dt.to_period('M')
    
    biometric_agg = biometric_df.groupby(['month', 'state', 'district']).size().reset_index(name='biometric_updates')
    
    # Prepare demographic aggregates
    if 'date' in demographic_df.columns:
        demographic_df['month'] = pd.to_datetime(demographic_df['date']).dt.to_period('M')
    elif 'month' in demographic_df.columns:
        demographic_df['month'] = pd.to_datetime(demographic_df['month']).dt.to_period('M')
    
    demographic_agg = demographic_df.groupby(['month', 'state', 'district']).size().reset_index(name='demographic_updates')
    
    # Prepare enrolment aggregates
    if 'date' in enrolment_df.columns:
        enrolment_df['month'] = pd.to_datetime(enrolment_df['date']).dt.to_period('M')
    elif 'month' in enrolment_df.columns:
        enrolment_df['month'] = pd.to_datetime(enrolment_df['month']).dt.to_period('M')
    
    enrolment_agg = enrolment_df.groupby(['month', 'state', 'district']).size().reset_index(name='enrolments')
    
    # Merge all data
    df = biometric_agg.merge(demographic_agg, on=['month', 'state', 'district'], how='outer')
    df = df.merge(enrolment_agg, on=['month', 'state', 'district'], how='outer')
    
    # Handle missing values
    df = df.fillna(0)
    
    # Convert to numeric
    df['biometric_updates'] = df['biometric_updates'].astype(int)
    df['demographic_updates'] = df['demographic_updates'].astype(int)
    df['enrolments'] = df['enrolments'].astype(int)
    
    # Derive columns
    df['total_updates'] = df['demographic_updates'] + df['biometric_updates']
    df['update_rate'] = np.where(
        df['enrolments'] > 0,
        df['total_updates'] / df['enrolments'],
        0
    )
    
    # Add anomaly flags (can be derived from data or loaded from existing files)
    # For this example, we'll create synthetic anomaly flags based on statistical outliers
    df['anomaly_flag'] = False
    df['anomaly_persistent'] = False
    
    # Detect anomalies using z-score method within state-district groups
    for (state, district), group_idx in df.groupby(['state', 'district']).groups.items():
        if len(group_idx) > 2:
            total_updates = df.loc[group_idx, 'total_updates']
            z_scores = np.abs((total_updates - total_updates.mean()) / (total_updates.std() + 1e-8))
            anomaly_mask = z_scores > 2
            df.loc[group_idx[anomaly_mask], 'anomaly_flag'] = True
    
    # Convert month to datetime
    df['month'] = df['month'].dt.to_timestamp()
    
    # Sort
    df = df.sort_values(['state', 'district', 'month']).reset_index(drop=True)
    
    print(f"  Combined data shape: {df.shape}")
    print(f"  Date range: {df['month'].min()} to {df['month'].max()}")
    print(f"  States: {df['state'].nunique()}")
    print(f"  Districts: {df['district'].nunique()}")
    
    return df


def run_elasticity_analysis(df: pd.DataFrame) -> tuple:
    """
    Execute elasticity analysis on prepared data.
    
    Args:
        df: Prepared DataFrame
    
    Returns:
        Tuple of (district_summary, state_summary, insight_generator)
    """
    print("\nRunning elasticity analysis...")
    district_summary, state_summary, generate_insight = analyze_demand_elasticity(df)
    print(f"  Analysis complete.")
    print(f"  Districts analyzed: {len(district_summary)}")
    
    return district_summary, state_summary, generate_insight


def generate_reports(
    district_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    generate_insight: callable,
    output_dir: str = '.'
):
    """
    Generate and save governance reports.
    
    Args:
        district_summary: District-level elasticity metrics
        state_summary: State-level summaries
        generate_insight: Insight generation function
        output_dir: Directory for output files
    """
    print("\nGenerating reports...")
    
    # Save district summary
    district_output = Path(output_dir) / 'elasticity_district_summary.csv'
    district_summary.to_csv(district_output, index=False)
    print(f"  ✓ District summary: {district_output}")
    
    # Save state summary
    state_output = Path(output_dir) / 'elasticity_state_summary.csv'
    state_summary.to_csv(state_output, index=False)
    print(f"  ✓ State summary: {state_output}")
    
    # Generate governance report
    report_path = Path(output_dir) / 'elasticity_governance_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AADHAAR DEMAND ELASTICITY ANALYSIS - GOVERNANCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        # State-level insights
        f.write("\nState-Level Findings:\n\n")
        for _, row in state_summary.iterrows():
            f.write(f"State: {row['state']}\n")
            f.write(f"  • Total Districts Analyzed: {row['total_districts_analyzed']}\n")
            f.write(f"  • High Elasticity Districts: {row['high_elasticity_districts']}\n")
            f.write(f"  • Moderate Elasticity Districts: {row['moderate_elasticity_districts']}\n")
            f.write(f"  • Low Elasticity Districts: {row['low_elasticity_districts']}\n")
            f.write(f"  • Capacity Responsive Districts: {row['capacity_responsive_pct']:.1f}%\n")
            f.write(f"  • Low Elasticity Districts: {row['low_elasticity_pct']:.1f}%\n")
            f.write(f"  • Average Elasticity Score: {row['avg_elasticity_score']:.3f}\n\n")
        
        # Top performing districts
        f.write("\nTop Performing Districts (High Elasticity):\n")
        f.write("-" * 80 + "\n")
        high_elasticity = district_summary[
            district_summary['elasticity_classification'] == 'High'
        ].sort_values('elasticity_score')
        
        if len(high_elasticity) > 0:
            for _, row in high_elasticity.head(10).iterrows():
                insight = generate_insight(row['state'], row['district'])
                f.write(f"\n{row['district']}, {row['state']}\n")
                f.write(f"  Elasticity Score: {row['elasticity_score']:.3f}\n")
                f.write(f"  Insight: {insight}\n")
        else:
            f.write("  No districts with high elasticity identified.\n")
        
        # Districts requiring attention
        f.write("\n\nDistricts Requiring Attention (Low Elasticity):\n")
        f.write("-" * 80 + "\n")
        low_elasticity = district_summary[
            district_summary['elasticity_classification'] == 'Low'
        ].sort_values('elasticity_score', ascending=False)
        
        if len(low_elasticity) > 0:
            for _, row in low_elasticity.head(10).iterrows():
                insight = generate_insight(row['state'], row['district'])
                f.write(f"\n{row['district']}, {row['state']}\n")
                f.write(f"  Elasticity Score: {row['elasticity_score']:.3f}\n")
                f.write(f"  Policy Recommendation: {row['policy_recommendation']}\n")
                f.write(f"  Insight: {insight}\n")
        else:
            f.write("  No districts with low elasticity identified.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ Governance report: {report_path}")
    print("\nReport generation complete.")


def main():
    """Main execution flow."""
    print("=" * 80)
    print("DEMAND ELASTICITY ANALYSIS - WORKFLOW")
    print("=" * 80 + "\n")
    
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data()
        
        # Step 2: Run elasticity analysis
        district_summary, state_summary, generate_insight = run_elasticity_analysis(df)
        
        # Step 3: Generate reports
        generate_reports(district_summary, state_summary, generate_insight)
        
        # Step 4: Display key findings
        print("\n" + "=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)
        
        print("\nState Summary:")
        print(state_summary.to_string(index=False))
        
        print("\n\nDistrict Summary (First 10 rows):")
        print(district_summary.head(10).to_string(index=False))
        
        print("\n" + "=" * 80)
        print("✓ Analysis complete. Check output CSV files for full results.")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
