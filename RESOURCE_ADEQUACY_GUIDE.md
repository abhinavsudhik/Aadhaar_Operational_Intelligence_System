# Resource Adequacy Assessment for Aadhaar Districts

## Overview

The **Resource Adequacy Assessment** module evaluates whether districts are **under-resourced**, **adequately resourced**, or **well-resourced** based on observed Aadhaar update activity patterns and operational stability.

This decision-support system helps policymakers and system planners:

- Identify capacity constraints and bottlenecks
- Allocate staffing and infrastructure investments
- Assess readiness for demand absorption
- Prioritize operational improvements

---

## Key Insight

**Higher and more stable demographic + biometric update activity indicates better operational capacity**, reflecting:

- Staffing levels and expertise
- Infrastructure quality
- Process maturity and reliability
- Ability to handle demand fluctuations

This is **not merely a demand indicator**—it's a proxy for operational health and resilience.

---

## Methodology

### 1. Metric Computation (Per District)

For each district, the system analyzes the **last 6 months** of activity:

#### **Metric 1: Update Volume**

- **Calculation**: Average total_updates (demographic + biometric) per month
- **Interpretation**: Higher volume → more active operations → better baseline capacity
- **Example**: District averaging 15,000 updates/month vs. 2,000 updates/month

#### **Metric 2: Update Stability**

- **Calculation**: Coefficient of Variation (CV) = σ(total_updates) / μ(total_updates)
- **Conversion to Score** (0-100): `100 × exp(-2 × CV)`
  - CV = 0.1 → score ≈ 82 (stable)
  - CV = 0.5 → score ≈ 37 (volatile)
  - CV = 1.0 → score ≈ 14 (unstable)
- **Interpretation**: Lower volatility → mature, reliable operations → higher capacity
- **Why it matters**: Volatile operations indicate process instability, staffing gaps, or infrastructure issues

#### **Metric 3: Anomaly Resolution Rate**

- **Calculation**: (1 - persistent_anomalies / detected_anomalies)
- **Interpretation**: High resolution rate (>80%) → responsive operations → better management
- **Example**: 10 anomalies detected, 8 resolved next month → resolution_rate = 0.80

#### **Metric 4: Data Completeness** (Hygiene Factor)

- **Calculation**: (months_with_data / lookback_months)
- **Interpretation**: Sparse data suggests operational challenges or missing infrastructure
- **Example**: Only 4 months of data in 6-month window → completeness = 0.67

---

### 2. Normalization (0-100 Scale)

All metrics are normalized to 0-100 using **percentile-based scaling** (5th to 95th percentile):

$$\text{Normalized Score} = \frac{\text{Value} - P_5}{\text{P}_{95} - P_5} \times 100$$

**Why percentile-based?**

- Prevents outlier districts from compressing the scale
- Ensures scores reflect relative capacity (not absolute activity)
- Makes thresholds meaningful and comparable across all states

---

### 3. Composite Resource Adequacy Score (0-100)

A weighted combination of normalized metrics:

$$\text{Adequacy Score} = 0.40 \times V + 0.35 \times S + 0.15 \times R + 0.10 \times C$$

Where:

- **V** = Update Volume Score (40% weight)
- **S** = Stability Score (35% weight)
- **R** = Anomaly Resolution Score (15% weight)
- **C** = Data Completeness Score (10% weight)

**Weight Rationale**:

- Volume (40%) and Stability (35%) = operational foundation
- Resolution (15%) = management effectiveness (important but secondary)
- Completeness (10%) = data hygiene (minimum bar)

---

### 4. Capacity Classification

Districts classified into three tiers based on adequacy score:

| Classification           | Score Range | Operational Profile                                                                                                                 |
| ------------------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Under-Resourced**      | < 35        | Limited activity, unstable operations, or sparse data. Requires process review, staffing analysis, or infrastructure investment.    |
| **Adequately Resourced** | 35–65       | Moderate activity, acceptable stability. Baseline capacity exists. Room for improvement in process reliability or anomaly handling. |
| **Well Resourced**       | ≥ 65        | Strong activity, high stability, effective anomaly resolution. Mature, reliable operations.                                         |

---

### 5. Demand Absorption Capability

**Question**: _Can this district absorb a 15–20% increase in update demand without operational congestion?_

**Decision Rule**: A district can absorb demand increase if:

- Adequacy Score ≥ 65 (strong baseline), **AND**
- Stability Score ≥ 60 (low volatility / consistent performance)

**Result**:

- **"Yes"** → Sufficient buffer; growth-ready
- **"At Risk"** → Limited buffer; monitor closely; may need reinforcement before major demand spike

---

## Output Format

### 1. District-Level Results

**File**: `resource_adequacy_assessment.csv`

One row per district with:

| Column                    | Type  | Example                                                                                                      |
| ------------------------- | ----- | ------------------------------------------------------------------------------------------------------------ |
| `state`                   | str   | "Karnataka"                                                                                                  |
| `district`                | str   | "Bangalore Urban"                                                                                            |
| `avg_monthly_updates`     | float | 18,542.3                                                                                                     |
| `update_stability_score`  | float | 76.4                                                                                                         |
| `anomaly_resolution_rate` | float | 0.85                                                                                                         |
| `resource_adequacy_score` | float | 68.2                                                                                                         |
| `capacity_classification` | str   | "Well Resourced"                                                                                             |
| `absorption_capability`   | str   | "Yes"                                                                                                        |
| `explanatory_note`        | str   | "Strong operational capacity with 18,542 avg monthly updates and 76/100 stability. Ready for demand growth." |

---

### 2. State-Level Summary

**File**: `resource_adequacy_state_summary.csv`

One row per state showing distribution:

| Column                     | Type  | Example     |
| -------------------------- | ----- | ----------- |
| `state`                    | str   | "Karnataka" |
| `total_districts`          | int   | 31          |
| `pct_under_resourced`      | float | 9.7         |
| `pct_adequately_resourced` | float | 45.2        |
| `pct_well_resourced`       | float | 45.2        |
| `avg_adequacy_score`       | float | 52.1        |

**Interpretation**: Karnataka has 45% of districts in strong capacity, 45% acceptable, 10% needing attention.

---

## Usage

### Basic Integration

```python
import pandas as pd
from resource_adequacy_assessment import (
    assess_resource_adequacy,
    generate_resource_adequacy_report
)

# Load your Aadhaar data
df = pd.read_csv('aadhaar_consolidated.csv')

# Run assessment (6-month lookback, 15% demand threshold)
district_results, state_summary = assess_resource_adequacy(
    df,
    lookback_months=6,
    demand_absorption_threshold=0.15
)

# Export results
generate_resource_adequacy_report(
    district_results,
    state_summary,
    output_csv="resource_adequacy_assessment.csv",
    output_summary_csv="resource_adequacy_state_summary.csv"
)
```

### Complete Example

```bash
python example_resource_adequacy.py
```

This script:

1. Loads all Aadhaar data from CSV files
2. Aggregates to monthly district level
3. Detects anomalies using IQR method
4. Runs resource adequacy assessment
5. Exports results and generates console report

---

## Interpretation Guide

### For Policymakers

**High-Priority Intervention**:

- Under-Resourced + At Risk absorption
- Volatility score < 40 (unstable operations)
- Anomaly resolution < 60% (process issues)

**Action Items**:

- Review staffing levels (usually 40-60% of capacity constraints)
- Assess infrastructure (power, connectivity, equipment)
- Identify process bottlenecks (training, SOPs)
- Pilot improvement initiatives in test districts

**Reallocation Opportunity**:

- Well Resourced + Absorption "Yes"
- Can absorb demand transferred from under-resourced districts
- Good test sites for new initiatives

---

### For System Planners

**Demand Forecasting**:

- Pair resource adequacy scores with demand forecasts
- Under-Resourced + high demand = congestion risk
- Well Resourced + stable demand = low risk

**Infrastructure Planning**:

- Stability score < 50 → likely infrastructure bottlenecks
- Completion score < 70 → missing reporting or process gaps
- Use to prioritize data center upgrades, connectivity improvements

**Training & Process**:

- Anomaly resolution < 75% → need better exception handling
- Coordinate with operational teams to identify root causes

---

## Key Assumptions

1. **Last 6 months is representative**: Adjust `lookback_months` if seasonal patterns dominate
2. **Anomalies indicate operational stress**: The model assumes detected anomalies reflect real operational issues
3. **Higher activity = better capacity**: Assumes staffing, infrastructure scale with activity
4. **No significant external shocks**: Model may need retraining if major disruptions occur (e.g., system outages)

---

## Limitations & Considerations

1. **Correlation vs. Causation**: High update activity may reflect demand rather than capacity. Context needed.
2. **Missing Data**: Districts with < 3 months of historical data receive conservative scores.
3. **Structural Changes**: If a district implemented major process changes, historical data may not reflect current capacity.
4. **Seasonality**: If updates are highly seasonal, consider filtering to same season or adjusting weights.

---

## Customization

### Adjust Assessment Period

```python
# Use last 3 months for fast-changing operations
district_results, state_summary = assess_resource_adequacy(df, lookback_months=3)

# Use 12 months for stable, mature operations
district_results, state_summary = assess_resource_adequacy(df, lookback_months=12)
```

### Adjust Metric Weights

Edit the composite score calculation in `assess_resource_adequacy()`:

```python
# Example: Emphasize stability over volume
district_metrics['resource_adequacy_score'] = (
    0.30 * district_metrics['update_volume_score'] +
    0.45 * district_metrics['stability_score'] +  # ← Increased
    0.15 * district_metrics['anomaly_resolution_score'] +
    0.10 * district_metrics['completeness_score']
).round(1)
```

### Adjust Classification Thresholds

```python
# Change from 35/65 split to 40/70
def classify_capacity(score: float) -> str:
    if score < 40:
        return "Under-Resourced"
    elif score < 70:
        return "Adequately Resourced"
    else:
        return "Well Resourced"
```

---

## Troubleshooting

### Issue: All districts classified as "Under-Resourced"

**Causes**:

- Data from very early rollout period (low activity is expected)
- Significant missing data (adjust threshold)
- All districts genuinely have limited capacity

**Solutions**:

- Check date range: `df['month'].min()` to `df['month'].max()`
- Review data completeness: `df.groupby('district')['month'].count()`
- Verify anomaly detection isn't too aggressive: `df['anomaly_flag'].value_counts()`

### Issue: Volatile scores between assessment runs

**Causes**:

- Percentile-based normalization shifts when new data added
- New districts or time periods change the reference distribution

**Solution**:

- Use fixed normalization bounds instead of percentiles
- Document baseline assessment (e.g., "Q1 2025 baseline")
- Compare relative rankings rather than absolute scores

### Issue: Explanatory notes are generic

**Improve**:

- Add state or region-specific context
- Incorporate external data (e.g., recent infrastructure projects)
- Customize thresholds based on domain knowledge

---

## Next Steps

1. **Validation**: Compare adequacy scores against expert assessments or field audits
2. **Dashboard Integration**: Embed district results in real-time operational dashboards
3. **Predictive Modeling**: Use adequacy scores as input to demand forecasting (see `aadhaar_demand_forecaster.py`)
4. **Continuous Monitoring**: Run assessment monthly to track improvement or degradation
5. **Intervention Tracking**: Correlate score changes with specific policy interventions to measure impact

---

## Questions & Support

For questions about:

- **Methodology**: See "Methodology" section and inline code comments
- **Data Preparation**: See `example_resource_adequacy.py`
- **Interpretation**: See "Interpretation Guide" section
- **Customization**: Modify weights and thresholds in `assess_resource_adequacy()` function

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Owner**: Decision Support System, Aadhaar Operations
