# Aadhaar Stress Prediction System

## Comprehensive Technical Documentation

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Conceptual Framework](#conceptual-framework)
3. [Function Reference](#function-reference)
4. [Usage Examples](#usage-examples)
5. [Interpretation Guide](#interpretation-guide)
6. [Edge Cases & Limitations](#edge-cases--limitations)

---

## System Overview

### Purpose

The Aadhaar Stress Prediction System estimates the likelihood that districts will exceed their normal Aadhaar update capacity in the near term (next 1 month). This enables proactive governance interventions and resource planning.

### Key Design Principles

- **Empirical Probability First**: Uses historical data patterns when sufficient history exists
- **Intelligent Fallback**: Employs heuristic risk scoring when historical patterns are insufficient
- **Interpretability**: Results are explainable for policy audiences, not black-box predictions
- **Robustness**: Handles missing data, sparse districts, and edge cases gracefully
- **Efficiency**: Uses only pandas and numpy; no heavy ML dependencies

---

## Conceptual Framework

### Definitions

#### 1. Normal Capacity

**Definition**: The 75th percentile of `total_updates` over a rolling 12-month window.

**Rationale**:

- Represents the "comfortable" operational level for a district
- 75th percentile captures typical high-activity scenarios without being affected by rare extremes
- Rolling window adapts to seasonal patterns and structural changes

**Computation**:

```
normal_capacity[t] = quantile(total_updates[t-11:t], 0.75)
```

#### 2. Stress Event

**Definition**: A month where `total_updates > normal_capacity`.

**Interpretation**:

- The district exceeded its normal operational capacity
- Indicates potential strain on administrative infrastructure
- May trigger cascading delays in biometric/demographic updates

#### 3. Recent Spike

**Definition**: Month-over-month growth in `total_updates > 20%` OR `anomaly_flag == True`.

**Rationale**:

- 20% threshold captures meaningful, policy-relevant growth
- Anomaly flag allows domain experts to flag unusual patterns
- Spikes are early indicators of potential stress events

**Computation**:

```
spike[t] = (total_updates[t] - total_updates[t-1]) / total_updates[t-1] > 0.20
           OR anomaly_flag[t] == True
```

#### 4. Stress Likelihood

**Definition**: The empirical probability that a spike is followed by a stress event within 1 month.

**Computation**:

```
stress_likelihood = (# spikes followed by stress) / (# total spikes)
```

**Example**:

- If there were 5 spikes historically, and 3 were followed by stress events:
- stress_likelihood = 3/5 = 0.60 (60% probability)

#### 5. Risk Level

**Mapping**:

- **Low**: stress_likelihood < 0.33 (below 33%)
- **Medium**: 0.33 ≤ stress_likelihood < 0.67
- **High**: stress_likelihood ≥ 0.67 (67% or above)

---

## Function Reference

### Core Analysis Functions

#### `estimate_district_stress_likelihood(df, reference_month=None, ...)`

**Purpose**: Main orchestration function that runs complete stress analysis.

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Aggregated monthly district-level Aadhaar data |
| `reference_month` | Timestamp | None | Reference month for predictions (uses last month if None) |
| `growth_threshold` | float | 0.20 | MoM growth threshold for spike detection (0-1) |
| `window_months` | int | 12 | Rolling window size for capacity (months) |
| `percentile` | float | 75.0 | Percentile for normal capacity (0-100) |
| `min_spike_history` | int | 2 | Minimum spikes for empirical probability |
| `lookahead_months` | int | 1 | Months ahead to check for stress events |

**Returns**: DataFrame with columns:

- `state`: State name
- `district`: District name
- `reference_month`: Reference month for analysis
- `normal_capacity`: District's 75th percentile threshold
- `predicted_updates_next_month`: Forecasted updates
- `stress_likelihood`: Probability (0-1)
- `risk_level`: Categorical risk (Low/Medium/High)

**Example**:

```python
results = estimate_district_stress_likelihood(
    df,
    reference_month=pd.Timestamp('2024-12-31'),
    growth_threshold=0.20
)
print(results.head())
```

---

### Helper Functions

#### `compute_rolling_normal_capacity(district_data, window_months=12, percentile=75.0)`

**Purpose**: Compute rolling 75th percentile for a single district.

**Returns**: Series of normal capacity values (NaN for insufficient history).

**Notes**:

- Uses `min_periods=3` to ensure minimum data points
- Avoids unstable estimates with very few observations

---

#### `identify_stress_events(district_data, normal_capacity)`

**Purpose**: Mark months where actual updates exceed normal capacity.

**Returns**: Boolean Series (True = stress event).

---

#### `identify_recent_spikes(district_data, growth_threshold=0.20, use_anomaly_flag=True)`

**Purpose**: Identify months with significant growth or flagged anomalies.

**Returns**: Boolean Series (True = recent spike).

---

#### `compute_empirical_stress_probability(district_data, recent_spikes, stress_events, lookahead_months=1)`

**Purpose**: Calculate historical probability of stress following spikes.

**Returns**:

- Float (0-1) if sufficient spike history exists
- None if fewer than 2 spikes

---

#### `compute_heuristic_risk_score(district_data, recent_months=6)`

**Purpose**: Fallback scoring when empirical history is insufficient.

**Components**:

1. **Recent Growth Rate** (40% weight)
   - Maps 50% growth → 1.0, 0% growth → 0.0
2. **Update Volatility** (40% weight)
   - Coefficient of Variation of last 6 months
   - CV > 0.5 → 1.0
3. **Anomaly Presence** (20% weight)
   - Proportion of anomalous records in recent months

**Returns**: Risk score (0-1).

---

#### `predict_next_month_updates(district_data, method='mean')`

**Purpose**: Forecast next month's total updates.

**Methods**:

- `'mean'`: Simple 3-month average (robust, conservative)
- `'trend'`: Linear extrapolation from last 6 months (captures trends)

**Returns**: Predicted updates (non-negative integer).

---

#### `map_likelihood_to_risk_level(stress_likelihood)`

**Purpose**: Convert probability to interpretable risk category.

**Returns**: 'Low', 'Medium', or 'High'.

---

### Reporting Functions

#### `summarize_stress_results(results_df)`

**Purpose**: Generate summary statistics for policy makers.

**Returns**: Dictionary with:

- Total districts analyzed
- Count by risk level
- Mean/median/max stress likelihood
- Top high-risk districts

**Example**:

```python
summary = summarize_stress_results(results)
print(f"High-risk districts: {summary['high_risk_count']}")
```

---

#### `get_district_details(results_df, state, district)`

**Purpose**: Retrieve detailed prediction for a specific district.

**Returns**: Dictionary or None if not found.

---

---

## Usage Examples

### Example 1: Basic Analysis

```python
import pandas as pd
from aadhaar_stress_prediction import estimate_district_stress_likelihood

# Load pre-aggregated data (month, state, district, total_updates, anomaly_flag)
df = pd.read_csv('aadhaar_monthly_aggregated.csv')
df['month'] = pd.to_datetime(df['month'])

# Run stress prediction
results = estimate_district_stress_likelihood(df)

# View top high-risk districts
high_risk = results[results['risk_level'] == 'High']
print(high_risk[['state', 'district', 'stress_likelihood', 'predicted_updates_next_month']])

# Save results
results.to_csv('stress_predictions.csv', index=False)
```

---

### Example 2: Scenario Analysis

```python
# What if we use a stricter growth threshold (10% instead of 20%)?
results_strict = estimate_district_stress_likelihood(
    df,
    growth_threshold=0.10
)
print(f"High-risk with strict threshold: {(results_strict['risk_level']=='High').sum()}")

# What if we look ahead 2 months instead of 1?
results_2month = estimate_district_stress_likelihood(
    df,
    lookahead_months=2
)
print(f"High-risk within 2 months: {(results_2month['risk_level']=='High').sum()}")
```

---

### Example 3: District-Level Deep Dive

```python
from aadhaar_stress_prediction import get_district_details

# Get detailed prediction for a specific district
details = get_district_details(results, state='Maharashtra', district='Mumbai')

print(f"Risk Level: {details['risk_level']}")
print(f"Normal Capacity: {details['normal_capacity']:.0f} updates/month")
print(f"Predicted Next Month: {details['predicted_updates_next_month']:.0f}")
print(f"Stress Likelihood: {details['stress_likelihood']:.2%}")
```

---

### Example 4: Temporal Forecasting

```python
# Run analysis for multiple reference months
for month in pd.date_range('2024-09-01', '2024-12-01', freq='M'):
    results_month = estimate_district_stress_likelihood(df, reference_month=month)
    high_risk_count = (results_month['risk_level'] == 'High').sum()
    print(f"{month.strftime('%Y-%m')}: {high_risk_count} high-risk districts")
```

---

## Interpretation Guide

### For Policy Makers

#### Understanding Risk Levels

**High Risk (Stress Likelihood ≥ 0.67)**

- Immediate intervention likely needed
- Allocate additional administrative resources
- Prepare contingency plans for delays
- Consider load balancing across nearby districts

**Medium Risk (0.33 ≤ Likelihood < 0.67)**

- Monitor closely
- Have resources on standby
- Plan efficiency improvements
- Train staff for peak periods

**Low Risk (Likelihood < 0.33)**

- Routine operations
- Standard resource allocation
- Plan preventive maintenance

#### Interpreting Predictions

**What does "Stress Likelihood = 0.45" mean?**

- Based on historical patterns, when a similar spike occurred in the past, there was a 45% probability it was followed by a stress event
- Not a guaranteed outcome, but a meaningful risk indicator
- Suggests Medium risk level and warrants monitoring

**Why is normal_capacity important?**

- It's the "comfort zone" for the district
- Exceeding it historically leads to performance degradation
- Use it as a planning target for capacity expansion

**How to use predicted_updates_next_month?**

- Compare to normal_capacity to assess headroom
- If predicted > capacity: initiate contingency planning
- If predicted >> capacity: significant resource constraints likely
- Track actual vs. predicted to recalibrate models

---

### For Data Analysts

#### Methodology Transparency

**Empirical Probability Method** (Primary):

1. Look at all historical "spike" events
2. For each spike, check if stress event occurred within 1 month
3. Calculate proportion as stress likelihood
4. More reliable if 5+ spike events exist

**Heuristic Fallback** (When < 2 spikes):

1. Recent growth rate: How much did updates grow last month?
2. Volatility: How unstable are district updates?
3. Anomaly flag: Has this district been unusual?
4. Combine with equal weight to get risk score

**Why 75th percentile?**

- 50th (median): Too conservative, ignores legitimate high periods
- 90th (or higher): Too lenient, misses real stress signals
- 75th: Captures "upper-normal" range, good signal-to-noise ratio

---

#### Data Quality Checks

```python
# Ensure complete data coverage
print(f"Missing months per district:")
missing = df.groupby('district')['month'].apply(
    lambda x: (x.max() - x.min()).days / 30 - len(x)
)
print(missing[missing > 0])

# Check for anomalies in aggregation
print(f"Districts with zero enrolments: {(df['enrolments']==0).sum()}")
print(f"Districts with zero updates: {(df['total_updates']==0).sum()}")

# Validate SSI ranges
print(f"SSI statistics:\n{df['ssi'].describe()}")
```

---

## Edge Cases & Limitations

### Handled Edge Cases

| Scenario                          | Handling                                         |
| --------------------------------- | ------------------------------------------------ |
| **District with < 2 months data** | Skipped (insufficient for rolling windows)       |
| **Missing months (sparse data)**  | Rolling window only uses available data          |
| **All months below capacity**     | Empirical probability = 0.0                      |
| **All months in spike condition** | Empirical probability = 1.0                      |
| **Divide-by-zero (SSI, MPI)**     | Uses NaN; handled in downstream calculations     |
| **Negative growth**               | Correctly identified as non-spike (< 20% growth) |
| **No anomaly_flag column**        | Gracefully ignored; spike only by growth         |

### Known Limitations

1. **Seasonality Not Modeled**
   - System doesn't explicitly account for annual patterns
   - Solution: Run separate analyses by quarter if strong seasonality exists

2. **Regime Changes**
   - If policy changed (e.g., new enrollment drives), historical patterns may not apply
   - Solution: Use data from consistent policy regime; reset window after major changes

3. **Sparse Districts**
   - Districts with few spikes rely heavily on heuristic fallback
   - Less reliable for districts with < 12 months history

4. **Correlation Between Districts**
   - System treats each district independently
   - May miss state-level or regional stress cascades
   - Solution: Post-process results to identify state-level patterns

5. **External Shocks**
   - System learns from historical data; cannot predict unprecedented events
   - Example: COVID-19 lockdowns, policy reversals

### Recommendations for Robustness

1. **Validation Against Actual Events**

   ```python
   # Compare historical predictions to actual outcomes
   historical_results = estimate_district_stress_likelihood(df_historical)
   actual_stress = df_actual['stress_event'].values
   calibration_accuracy = (historical_results['stress_likelihood'] > 0.5) == actual_stress
   print(f"Calibration accuracy: {calibration_accuracy.mean():.2%}")
   ```

2. **Sensitivity Analysis**

   ```python
   # Test how results change with different thresholds
   for growth_threshold in [0.10, 0.15, 0.20, 0.25]:
       results = estimate_district_stress_likelihood(df, growth_threshold=growth_threshold)
       print(f"Growth threshold {growth_threshold:.0%}: {(results['risk_level']=='High').sum()} high-risk")
   ```

3. **Regular Retraining**
   - Recompute predictions monthly as new data arrives
   - Allows calibration to changing district patterns
   - Detect and flag districts with sudden behavior shifts

---

## References

### Definitions and Methodology

- **Percentile-based thresholds**: Standard practice in capacity planning
- **Empirical probability**: Frequentist approach; transparent and auditable
- **Heuristic fallback**: Combines interpretable components (growth, volatility)

### Related Work

- District-level governance analytics: Public Financial Management Systems
- Anomaly detection: Statistical process control (SPC) principles
- Forecasting: Simple exponential smoothing and linear trend methods

---

## Contact & Support

For questions about implementation, interpretation, or customization:

- Review helper function docstrings for detailed parameter explanations
- Modify `growth_threshold` and `window_months` to match your policy goals
- Extend `compute_heuristic_risk_score()` to include additional signals

---

**Last Updated**: January 2025
**Version**: 1.0
**Status**: Production-Ready
