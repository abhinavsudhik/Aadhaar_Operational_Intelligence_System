"""
Data Loading & Biometric Surge Predictor Runner
================================================

Loads Aadhaar data from CSV files and runs biometric surge predictions.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

# Import the predictor module
from biometric_surge_predictor import generate_prediction_report


def load_and_combine_aadhaar_data(base_path: str) -> pd.DataFrame:
    """
    Load and combine all Aadhaar CSV files from the data folders.
    
    Parameters:
    -----------
    base_path : str
        Base path to the ML folder
    
    Returns:
    --------
    pd.DataFrame
        Combined dataset with all relevant columns
    """
    
    print("=" * 70)
    print("LOADING AADHAAR DATA")
    print("=" * 70)
    
    # Load biometric data
    print("\n📊 Loading biometric data...")
    biometric_files = glob.glob(os.path.join(base_path, 'api_data_aadhar_biometric', '*.csv'))
    biometric_dfs = []
    for file in sorted(biometric_files):
        print(f"  ✓ {Path(file).name}")
        df = pd.read_csv(file)
        biometric_dfs.append(df)
    biometric_data = pd.concat(biometric_dfs, ignore_index=True) if biometric_dfs else pd.DataFrame()
    print(f"  Total biometric records: {len(biometric_data):,}")
    
    # Load demographic data
    print("\n👥 Loading demographic data...")
    demographic_files = glob.glob(os.path.join(base_path, 'api_data_aadhar_demographic', '*.csv'))
    demographic_dfs = []
    for file in sorted(demographic_files):
        print(f"  ✓ {Path(file).name}")
        df = pd.read_csv(file)
        demographic_dfs.append(df)
    demographic_data = pd.concat(demographic_dfs, ignore_index=True) if demographic_dfs else pd.DataFrame()
    print(f"  Total demographic records: {len(demographic_data):,}")
    
    # Load enrolment data
    print("\n📝 Loading enrolment data...")
    enrolment_files = glob.glob(os.path.join(base_path, 'api_data_aadhar_enrolment', '*.csv'))
    enrolment_dfs = []
    for file in sorted(enrolment_files):
        print(f"  ✓ {Path(file).name}")
        df = pd.read_csv(file)
        enrolment_dfs.append(df)
    enrolment_data = pd.concat(enrolment_dfs, ignore_index=True) if enrolment_dfs else pd.DataFrame()
    print(f"  Total enrolment records: {len(enrolment_data):,}")
    
    # Print column info
    print("\n📋 Data Schema:")
    print("\nBiometric columns:", biometric_data.columns.tolist())
    print("Demographic columns:", demographic_data.columns.tolist())
    print("Enrolment columns:", enrolment_data.columns.tolist())
    
    # Combine datasets
    print("\n🔗 Combining datasets...")
    
    # Start with enrolment data as base
    df = enrolment_data.copy() if len(enrolment_data) > 0 else pd.DataFrame()
    
    # Merge biometric data
    if len(df) > 0 and len(biometric_data) > 0:
        # Find common columns for merge
        common_cols = list(set(df.columns) & set(biometric_data.columns))
        if common_cols:
            df = df.merge(biometric_data, on=common_cols, how='outer')
            print(f"  ✓ Merged biometric data on: {common_cols}")
    
    # Merge demographic data
    if len(df) > 0 and len(demographic_data) > 0:
        common_cols = list(set(df.columns) & set(demographic_data.columns))
        if common_cols:
            df = df.merge(demographic_data, on=common_cols, how='outer')
            print(f"  ✓ Merged demographic data on: {common_cols}")
    
    print(f"\n✅ Combined dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    return df


def prepare_predictor_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data into predictor-ready format.
    
    Required columns for predictor:
    - month (datetime or YYYY-MM)
    - state (string)
    - district (string)
    - age_group (string)
    - biometric_updates (int)
    - enrolments (int)
    
    Parameters:
    -----------
    raw_df : pd.DataFrame
        Raw combined data
    
    Returns:
    --------
    pd.DataFrame
        Prepared dataset for predictor
    """
    
    print("\n" + "=" * 70)
    print("PREPARING DATA FOR PREDICTOR")
    print("=" * 70)
    
    df = raw_df.copy()
    
    # Identify and standardize key columns
    print("\n🔍 Analyzing columns for mapping...")
    print(f"   Available columns: {df.columns.tolist()}")
    
    # Month column
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['month'].dt.to_period('M').astype(str)
    
    # State, district already present
    # Age group: We need to pivot age columns into age_group dimension
    
    # MAP ENROLMENT COLUMNS TO AGE GROUPS
    age_enrol_mapping = {
        'age_0_5': '0-5',
        'age_5_17': '5-17',
        'age_18_greater': '18+'
    }
    
    # MAP BIOMETRIC COLUMNS TO AGE GROUPS
    age_biometric_mapping = {
        'bio_age_5_17': '5-17',
        'bio_age_17_': '18+'
    }
    
    print(f"   Enrolment age columns: {[col for col in age_enrol_mapping.keys() if col in df.columns]}")
    print(f"   Biometric age columns: {[col for col in age_biometric_mapping.keys() if col in df.columns]}")
    
    # Melt enrolment data to long format
    enrol_cols = [col for col in age_enrol_mapping.keys() if col in df.columns]
    if enrol_cols:
        df_enrol = df[['month', 'state', 'district'] + enrol_cols].copy()
        df_enrol = df_enrol.melt(
            id_vars=['month', 'state', 'district'],
            value_vars=enrol_cols,
            var_name='age_col',
            value_name='enrolments'
        )
        df_enrol['age_group'] = df_enrol['age_col'].map(age_enrol_mapping)
        df_enrol = df_enrol.drop('age_col', axis=1)
    else:
        df_enrol = pd.DataFrame()
    
    # Melt biometric data to long format
    biom_cols = [col for col in age_biometric_mapping.keys() if col in df.columns]
    if biom_cols:
        df_biom = df[['month', 'state', 'district'] + biom_cols].copy()
        df_biom = df_biom.melt(
            id_vars=['month', 'state', 'district'],
            value_vars=biom_cols,
            var_name='age_col',
            value_name='biometric_updates'
        )
        df_biom['age_group'] = df_biom['age_col'].map(age_biometric_mapping)
        df_biom = df_biom.drop('age_col', axis=1)
    else:
        df_biom = pd.DataFrame()
    
    # Merge enrolment and biometric data
    if len(df_enrol) > 0 and len(df_biom) > 0:
        df_final = df_enrol.merge(
            df_biom,
            on=['month', 'state', 'district', 'age_group'],
            how='outer'
        )
    elif len(df_enrol) > 0:
        df_final = df_enrol
        df_final['biometric_updates'] = 0
    elif len(df_biom) > 0:
        df_final = df_biom
        df_final['enrolments'] = 0
    else:
        df_final = pd.DataFrame()
    
    # Convert to numeric and fill NaN
    if len(df_final) > 0:
        df_final['enrolments'] = pd.to_numeric(df_final['enrolments'], errors='coerce').fillna(0)
        df_final['biometric_updates'] = pd.to_numeric(df_final['biometric_updates'], errors='coerce').fillna(0)
        
        # Remove rows with null month, state, or district
        df_final = df_final.dropna(subset=['month', 'state', 'district'])
        
        # Convert month to datetime
        df_final['month'] = pd.to_datetime(df_final['month'], errors='coerce')
        df_final = df_final.dropna(subset=['month'])
        
        # Aggregate by month, state, district, age_group to remove duplicates
        df_final = df_final.groupby(['month', 'state', 'district', 'age_group'], as_index=False)[
            ['biometric_updates', 'enrolments']
        ].sum()
        
        # Sort by month
        df_final = df_final.sort_values('month').reset_index(drop=True)
        
        # Derive biometric_update_rate
        df_final['biometric_update_rate'] = (
            df_final['biometric_updates'] / (df_final['enrolments'] + 1)
        )
        df_final['anomaly_flag'] = False
    
    print(f"\n✅ Prepared dataset shape: {df_final.shape}")
    if len(df_final) > 0:
        print(f"   Month range: {df_final['month'].min()} to {df_final['month'].max()}")
        print(f"   States: {df_final['state'].nunique()}")
        print(f"   Districts: {df_final['district'].nunique()}")
        print(f"   Age groups: {df_final['age_group'].nunique()}")
    
    return df_final


def run_predictions(df: pd.DataFrame) -> None:
    """
    Run biometric surge predictions and display results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Prepared dataset for predictor
    """
    
    if len(df) == 0:
        print("\n❌ No data available for predictions.")
        return
    
    print("\n" + "=" * 70)
    print("RUNNING BIOMETRIC SURGE PREDICTIONS")
    print("=" * 70)
    
    # Generate report
    print("\n⚙️  Processing predictions (this may take a moment)...\n")
    report = generate_prediction_report(df)
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"   Reference Month: {report['reference_month']}")
    print(f"   Total Districts Analyzed: {report['total_districts_analyzed']}")
    print(f"   Districts with Expected Surge: {report['districts_with_surge']}")
    print(f"   Overall Surge Percentage: {report['overall_surge_percentage']:.1f}%")
    
    # State summary
    print(f"\n🏛️  STATE-LEVEL SUMMARY")
    print(report['state_summary'].to_string(index=False))
    
    # Top surge districts
    print(f"\n⚠️  TOP HIGH-RISK DISTRICTS (Expected Surge)")
    if len(report['top_surge_districts']) > 0:
        for idx, row in report['top_surge_districts'].iterrows():
            print(f"\n   {row['district']} ({row['state']})")
            print(f"   - Surge Probability: {row['surge_probability']*100:.1f}%")
            print(f"   - Impact Level: {row['expected_impact_level']}")
            print(f"   - Window: {row['expected_surge_window']}")
    else:
        print("   No districts with expected surge.")
    
    # Operational insights
    print(f"\n💡 OPERATIONAL INSIGHTS")
    for i, insight in enumerate(report['top_surge_insights'][:5], 1):
        print(f"\n   {i}. {insight}")
    
    # Full district predictions
    print(f"\n📋 FULL DISTRICT PREDICTIONS (Top 15)")
    print(report['district_predictions'].head(15).to_string(index=False))
    
    # Save results
    output_dir = Path(__file__).parent
    
    # Save district predictions
    pred_file = output_dir / 'biometric_surge_district_predictions.csv'
    report['district_predictions'].to_csv(pred_file, index=False)
    print(f"\n✅ Saved district predictions to: {pred_file.name}")
    
    # Save state summary
    state_file = output_dir / 'biometric_surge_state_summary.csv'
    report['state_summary'].to_csv(state_file, index=False)
    print(f"✅ Saved state summary to: {state_file.name}")
    
    print("\n" + "=" * 70)
    print("✅ PREDICTIONS COMPLETE")
    print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    base_path = '/Users/abhinavsudhi/Downloads/DESKTOP2/ML'
    
    # Load data
    raw_df = load_and_combine_aadhaar_data(base_path)
    
    # Prepare for predictor
    prepared_df = prepare_predictor_dataset(raw_df)
    
    # Run predictions
    if len(prepared_df) > 0:
        run_predictions(prepared_df)
    else:
        print("\n❌ Failed to prepare data for predictions.")
