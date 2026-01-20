import pandas as pd
import numpy as np
from pathlib import Path

base_path = Path(".")

# Load and aggregate demo data
demo_files = sorted((base_path / "api_data_aadhar_demographic").glob("*.csv"))
demo_dfs = [pd.read_csv(f) for f in demo_files[:1]]
demo_df = pd.concat(demo_dfs, ignore_index=True)

demo_df['date'] = pd.to_datetime(demo_df['date'], format='%d-%m-%Y')
demo_df['month'] = demo_df['date'].dt.to_period('M')
numeric_cols = [c for c in demo_df.columns if c not in ['date', 'month', 'state', 'district', 'pincode']]

print("Numeric cols:", numeric_cols)

demo_agg = (
    demo_df
    .groupby(['month', 'state', 'district'])
    [numeric_cols]
    .sum()
    .sum(axis=1)
    .reset_index(name='demographic_updates')
)

print("Columns after agg:", demo_agg.columns.tolist())
print(demo_agg.head())
