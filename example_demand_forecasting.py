"""
Example script demonstrating Aadhaar demand forecasting.

This script shows how to:
1. Load and prepare Aadhaar data from multiple CSV files
2. Generate district-level demand forecasts
3. Create state-level summaries
4. Export results for dashboards and reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from aadhaar_demand_forecaster import (
    forecast_district_demand, 
    generate_forecast_report
)


def load_aadhaar_data() -> pd.DataFrame:
    """
    Load Aadhaar data from demographic, biometric, and enrolment files.
    
    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with monthly district-level data
    """
    
    print("Loading Aadhaar data files...")
    
    # Define paths
    base_path = Path(__file__).parent
    
    # Load demographic data
    demo_files = list((base_path / "api_data_aadhar_demographic").glob("*.csv"))
    demo_dfs = []
    for file in demo_files:
        df = pd.read_csv(file)
        demo_dfs.append(df)
    demographic_df = pd.concat(demo_dfs, ignore_index=True)
    print(f"  Loaded {len(demographic_df):,} demographic records")
    
    # Load biometric data
    bio_files = list((base_path / "api_data_aadhar_biometric").glob("*.csv"))
    bio_dfs = []
    for file in bio_files:
        df = pd.read_csv(file)
        bio_dfs.append(df)
    biometric_df = pd.concat(bio_dfs, ignore_index=True)
    print(f"  Loaded {len(biometric_df):,} biometric records")
    
    return demographic_df, biometric_df


def prepare_forecast_data(demographic_df: pd.DataFrame, 
                          biometric_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data into format required for forecasting.
    
    Expected columns in input:
    - Some date/month identifier
    - State/district information
    - Update counts
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: month, state, district, 
        demographic_updates, biometric_updates, total_updates
    """
    
    print("\nPreparing data for forecasting...")
    
    # This is a template - adjust based on your actual column names
    # Inspect the first few rows to understand structure
    print("\nDemographic columns:", demographic_df.columns.tolist())
    print("Biometric columns:", biometric_df.columns.tolist())
    
    # Example aggregation (ADJUST THIS BASED ON YOUR ACTUAL DATA STRUCTURE)
    # Assuming your data has columns like: date, state, district, count/updates
    
    # For demonstration, let's create a mock aggregated dataset
    # In production, replace this with actual aggregation logic
    
    # Check if data has the expected structure
    required_cols_demo = ['state', 'district']  # Add actual column names
    required_cols_bio = ['state', 'district']   # Add actual column names
    
    # Create monthly aggregates
    # This is pseudocode - adapt to your data structure
    """
    # Example aggregation pattern:
    demo_monthly = demographic_df.groupby(['month', 'state', 'district']).agg({
        'update_count': 'sum'
    }).reset_index()
    demo_monthly.rename(columns={'update_count': 'demographic_updates'}, inplace=True)
    
    bio_monthly = biometric_df.groupby(['month', 'state', 'district']).agg({
        'update_count': 'sum'
    }).reset_index()
    bio_monthly.rename(columns={'update_count': 'biometric_updates'}, inplace=True)
    
    # Merge demographic and biometric
    df = pd.merge(
        demo_monthly, 
        bio_monthly,
        on=['month', 'state', 'district'],
        how='outer'
    ).fillna(0)
    """
    
    # For now, return a sample structure message
    print("\n⚠️  IMPORTANT: Update the prepare_forecast_data() function")
    print("   with actual column names from your CSV files.")
    print("\n   Inspect your data structure first:")
    print(f"   - Demographic sample: {demographic_df.head(2).to_dict()}")
    print(f"   - Biometric sample: {biometric_df.head(2).to_dict()}")
    
    return None


def create_sample_data() -> pd.DataFrame:
    """
    Create sample data for testing the forecaster.
    Remove this once you have real data prepared.
    """
    
    print("\nGenerating sample data for demonstration...")
    
    np.random.seed(42)
    
    states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'West Bengal']
    districts_per_state = 3
    months = pd.date_range('2023-01-01', '2025-12-01', freq='MS')
    
    data = []
    
    for state in states:
        for dist_idx in range(districts_per_state):
            district = f"{state}_District_{dist_idx+1}"
            
            # Generate time series with trend and seasonality
            base_updates = np.random.randint(5000, 15000)
            trend = np.linspace(0, 2000, len(months))
            seasonality = 1000 * np.sin(np.arange(len(months)) * 2 * np.pi / 12)
            noise = np.random.normal(0, 800, len(months))
            
            total_updates = base_updates + trend + seasonality + noise
            total_updates = np.maximum(total_updates, 0).astype(int)
            
            # Split into demographic and biometric (roughly 60/40)
            demographic = (total_updates * 0.6).astype(int)
            biometric = total_updates - demographic
            
            for month, demo, bio, total in zip(months, demographic, biometric, total_updates):
                data.append({
                    'month': month,
                    'state': state,
                    'district': district,
                    'demographic_updates': demo,
                    'biometric_updates': bio,
                    'total_updates': total
                })
    
    df = pd.DataFrame(data)
    print(f"  Created {len(df):,} sample records")
    print(f"  Date range: {df['month'].min()} to {df['month'].max()}")
    print(f"  States: {df['state'].nunique()}, Districts: {df['district'].nunique()}")
    
    return df


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("AADHAAR DEMAND FORECASTING - EXAMPLE SCRIPT")
    print("=" * 70)
    
    # Option 1: Load real data (uncomment and modify when ready)
    """
    demographic_df, biometric_df = load_aadhaar_data()
    df = prepare_forecast_data(demographic_df, biometric_df)
    
    if df is None:
        print("\n❌ Data preparation incomplete. Please update the script.")
        return
    """
    
    # Option 2: Use sample data for testing
    df = create_sample_data()
    
    # Generate forecasts
    print("\n" + "=" * 70)
    print("GENERATING FORECASTS")
    print("=" * 70)
    
    district_forecast, state_summary = forecast_district_demand(
        df,
        lookback_months=12,
        forecast_horizon=3
    )
    
    print(f"\n✓ Generated forecasts for {len(district_forecast)} districts")
    
    # Display sample results
    print("\n" + "-" * 70)
    print("DISTRICT FORECAST SAMPLE (First 5 rows)")
    print("-" * 70)
    print(district_forecast.head().to_string())
    
    print("\n" + "-" * 70)
    print("STATE SUMMARY")
    print("-" * 70)
    print(state_summary.to_string())
    
    # Generate text report
    print("\n" + "=" * 70)
    report = generate_forecast_report(district_forecast, state_summary)
    print(report)
    
    # Export results
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)
    
    output_files = {
        'district_demand_forecast.csv': district_forecast,
        'state_demand_summary.csv': state_summary,
        'demand_forecast_report.txt': report
    }
    
    for filename, content in output_files.items():
        if isinstance(content, pd.DataFrame):
            content.to_csv(filename, index=False)
            print(f"  ✓ Saved {filename}")
        else:
            with open(filename, 'w') as f:
                f.write(content)
            print(f"  ✓ Saved {filename}")
    
    # Analysis insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    high_demand = district_forecast[district_forecast['demand_level'] == 'High']
    print(f"  • {len(high_demand)} districts require immediate attention (High demand)")
    
    if len(high_demand) > 0:
        top_district = high_demand.nlargest(1, 'forecast_1m').iloc[0]
        print(f"  • Highest forecast: {top_district['district']} ({top_district['state']})")
        print(f"    Expected updates next month: {top_district['forecast_1m']:,}")
    
    critical_states = state_summary[state_summary['high_demand_districts'] > 0]
    print(f"  • {len(critical_states)} states have high-demand districts")
    
    print("\n✓ Forecasting complete!")


if __name__ == "__main__":
    main()
