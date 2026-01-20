"""
Biometric Surge Predictor - SIMPLIFIED DEMO
Loads sample of data and runs predictions for demonstration
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

# Import the predictor
from biometric_surge_predictor import generate_prediction_report

def load_sample_aadhaar_data(base_path: str, sample_size: int = 100000) -> pd.DataFrame:
    """Load a sample of Aadhaar data for faster processing."""
    
    print("=" * 70)
    print("LOADING SAMPLE AADHAAR DATA")
    print("=" * 70)
    
    # Load first CSV from each folder
    biometric_file = glob.glob(os.path.join(base_path, 'api_data_aadhar_biometric', '*.csv'))[0]
    enrolment_file = glob.glob(os.path.join(base_path, 'api_data_aadhar_enrolment', '*.csv'))[0]
    demographic_file = glob.glob(os.path.join(base_path, 'api_data_aadhar_demographic', '*.csv'))[0]
    
    print(f"\n📊 Loading biometric sample from: {Path(biometric_file).name}")
    biometric_df = pd.read_csv(biometric_file, nrows=sample_size)
    print(f"   Loaded {len(biometric_df):,} records")
    
    print(f"📝 Loading enrolment sample from: {Path(enrolment_file).name}")
    enrolment_df = pd.read_csv(enrolment_file, nrows=sample_size)
    print(f"   Loaded {len(enrolment_df):,} records")
    
    print(f"👥 Loading demographic sample from: {Path(demographic_file).name}")
    demographic_df = pd.read_csv(demographic_file, nrows=sample_size)
    print(f"   Loaded {len(demographic_df):,} records")
    
    # Merge on common columns
    print("\n🔗 Merging datasets...")
    df = enrolment_df.copy()
    common_cols = list(set(df.columns) & set(biometric_df.columns))
    df = df.merge(biometric_df, on=common_cols, how='outer')
    common_cols = list(set(df.columns) & set(demographic_df.columns))
    df = df.merge(demographic_df, on=common_cols, how='outer')
    
    print(f"✅ Combined shape: {df.shape}")
    return df


def prepare_data_for_predictor(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw data into predictor format."""
    
    print("\n" + "=" * 70)
    print("PREPARING DATA FOR PREDICTOR")
    print("=" * 70)
    
    df = raw_df.copy()
    
    # Create month from date
    if 'date' in df.columns:
        df['month'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['month'].dt.to_period('M').astype(str)
    
    # Map age columns to age groups
    age_enrol_mapping = {
        'age_0_5': '0-5',
        'age_5_17': '5-17',
        'age_18_greater': '18+'
    }
    
    age_biometric_mapping = {
        'bio_age_5_17': '5-17',
        'bio_age_17_': '18+'
    }
    
    enrol_cols = [col for col in age_enrol_mapping.keys() if col in df.columns]
    biom_cols = [col for col in age_biometric_mapping.keys() if col in df.columns]
    
    print(f"\n📋 Available age columns:")
    print(f"   Enrolment: {enrol_cols}")
    print(f"   Biometric: {biom_cols}")
    
    # Melt to long format
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
    
    # Merge
    if len(df_enrol) > 0 and len(df_biom) > 0:
        df_final = df_enrol.merge(df_biom, on=['month', 'state', 'district', 'age_group'], how='outer')
    elif len(df_enrol) > 0:
        df_final = df_enrol
        df_final['biometric_updates'] = 0
    else:
        df_final = df_biom
        df_final['enrolments'] = 0
    
    # Clean and aggregate
    df_final['enrolments'] = pd.to_numeric(df_final['enrolments'], errors='coerce').fillna(0)
    df_final['biometric_updates'] = pd.to_numeric(df_final['biometric_updates'], errors='coerce').fillna(0)
    df_final = df_final.dropna(subset=['month', 'state', 'district'])
    df_final['month'] = pd.to_datetime(df_final['month'], errors='coerce')
    df_final = df_final.dropna(subset=['month'])
    
    # Aggregate
    df_final = df_final.groupby(['month', 'state', 'district', 'age_group'], as_index=False)[
        ['biometric_updates', 'enrolments']
    ].sum()
    
    df_final = df_final.sort_values('month').reset_index(drop=True)
    df_final['biometric_update_rate'] = df_final['biometric_updates'] / (df_final['enrolments'] + 1)
    df_final['anomaly_flag'] = False
    
    print(f"\n✅ Prepared dataset: {df_final.shape}")
    print(f"   Month range: {df_final['month'].min()} to {df_final['month'].max()}")
    print(f"   States: {df_final['state'].nunique()}")
    print(f"   Districts: {df_final['district'].nunique()}")
    print(f"   Age groups: {df_final['age_group'].nunique()}")
    
    return df_final


if __name__ == '__main__':
    
    base_path = '/Users/abhinavsudhi/Downloads/DESKTOP2/ML'
    
    # Load sample data
    raw_df = load_sample_aadhaar_data(base_path, sample_size=50000)
    
    # Prepare
    df = prepare_data_for_predictor(raw_df)
    
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("RUNNING BIOMETRIC SURGE PREDICTIONS")
        print("=" * 70)
        
        # Generate predictions
        print("\n⚙️  Processing predictions...")
        report = generate_prediction_report(df)
        
        # Display results
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Total Districts: {report['total_districts_analyzed']}")
        print(f"   Districts with Expected Surge: {report['districts_with_surge']}")
        print(f"   Surge Percentage: {report['overall_surge_percentage']:.1f}%")
        
        print(f"\n🏛️  State-Level Summary:")
        print(report['state_summary'].to_string(index=False))
        
        print(f"\n⚠️  Top High-Risk Districts:")
        if len(report['top_surge_districts']) > 0:
            for _, row in report['top_surge_districts'].head(10).iterrows():
                print(f"\n   {row['district']} ({row['state']})")
                print(f"   - Probability: {row['surge_probability']*100:.0f}%")
                print(f"   - Impact: {row['expected_impact_level']}")
                print(f"   - Window: {row['expected_surge_window']}")
        
        print(f"\n💡 Operational Insights (Top 5):")
        for i, insight in enumerate(report['top_surge_insights'][:5], 1):
            print(f"\n   {i}. {insight}")
        
        # Save results
        output_dir = Path(__file__).parent
        pred_file = output_dir / 'biometric_surge_predictions_sample.csv'
        report['district_predictions'].to_csv(pred_file, index=False)
        print(f"\n✅ Saved to: {pred_file.name}")
        
        print("\n" + "=" * 70)
        print("✅ PREDICTIONS COMPLETE")
        print("=" * 70)
    else:
        print("❌ Failed to prepare data.")
