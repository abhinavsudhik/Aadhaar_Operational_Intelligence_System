import pandas as pd
import numpy as np
from pathlib import Path

# Simplified version of the data loading to test groupby behavior

# Create test data
data = {
    'month': pd.to_datetime(['2025-03-01', '2025-03-01', '2025-03-01', '2025-04-01', '2025-04-01']),
    'state': ['State A', 'State A', 'State B', 'State A', 'State B'],
    'district': ['D1', 'D2', 'D1', 'D1', 'D1'],
    'total_updates': [100, 200, 150, 120, 180]
}
merged = pd.DataFrame(data)
print("Original merged:")
print(merged)
print()

def detect_anomalies(group):
    """Detect anomalies for a single district using IQR method."""
    Q1 = group['total_updates'].quantile(0.25)
    Q3 = group['total_updates'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    group['anomaly_flag'] = (
        (group['total_updates'] < lower_bound) |
        (group['total_updates'] > upper_bound)
    )
    group['anomaly_persistent'] = (
        group['anomaly_flag'].shift(1) & group['anomaly_flag']
    )
    group['anomaly_persistent'] = group['anomaly_persistent'].fillna(False)
    
    return group

result = (
    merged
    .sort_values(['state', 'district', 'month'])
    .groupby(['state', 'district'], group_keys=False)
    .apply(detect_anomalies, include_groups=False)
)

print("After groupby apply:")
print(result)
print("Columns:", result.columns.tolist())
