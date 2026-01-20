"""
QUICK START GUIDE - Demand Elasticity Analyzer
===============================================

A step-by-step guide to using the elasticity analysis system.
"""

# ============================================================================

# 1. BASIC USAGE (with prepared DataFrame)

# ============================================================================

"""
If you already have a pandas DataFrame with the required columns:

- month (datetime)
- state (string)
- district (string)
- demographic_updates (int)
- biometric_updates (int)
- enrolments (int)
- anomaly_flag (boolean)
- total_updates (int) [derived: demographic_updates + biometric_updates]
- update_rate (float) [derived: total_updates / enrolments]

## Usage:

"""

from demand_elasticity_analyzer import analyze_demand_elasticity

# Assuming you have df loaded

district_summary, state_summary, generate_insight = analyze_demand_elasticity(df)

# View district-level results

print(district_summary)

# View state-level results

print(state_summary)

# Generate insight for a specific district

insight = generate_insight('Karnataka', 'Bangalore')
print(insight)

# ============================================================================

# 2. COMPLETE WORKFLOW (with raw CSV data)

# ============================================================================

"""
If you have raw CSV files in separate directories:

## Usage:

"""

from run_elasticity_analysis import (
load_and_prepare_data,
run_elasticity_analysis,
generate_reports
)

# Step 1: Load and combine multiple CSV files

df = load_and_prepare_data(
biometric_dir='api_data_aadhar_biometric',
demographic_dir='api_data_aadhar_demographic',
enrolment_dir='api_data_aadhar_enrolment'
)

# Step 2: Run elasticity analysis

district_summary, state_summary, generate_insight = run_elasticity_analysis(df)

# Step 3: Generate governance reports

generate_reports(district_summary, state_summary, generate_insight)

# This creates:

# - elasticity_district_summary.csv

# - elasticity_state_summary.csv

# - elasticity_governance_report.txt

# ============================================================================

# 3. USING TEST DATA (for validation)

# ============================================================================

"""
Generate synthetic test data and run analysis:
"""

from test_elasticity_analyzer import create_synthetic_district_data
from demand_elasticity_analyzer import analyze_demand_elasticity
import pandas as pd

# Create synthetic districts with different patterns

df = pd.concat([
create_synthetic_district_data(months=24, state='TestState', district='HighElasticity', pattern='increasing'),
create_synthetic_district_data(months=24, state='TestState', district='LowElasticity', pattern='stable'),
create_synthetic_district_data(months=24, state='TestState', district='Decreasing', pattern='decreasing')
], ignore_index=True)

# Run analysis

district_summary, state_summary, generate_insight = analyze_demand_elasticity(df)
print(district_summary)

# ============================================================================

# 4. CUSTOMIZING PARAMETERS

# ============================================================================

"""
Adjust sensitivity of capacity increase detection:
"""

# More sensitive (detects smaller increases)

district*summary, state_summary, * = analyze_demand_elasticity(
df,
min_sustained_months=1 # Single month counts
)

# More strict (requires larger increases over longer periods)

district*summary, state_summary, * = analyze_demand_elasticity(
df,
min_sustained_months=6 # Requires 6 consecutive months
)

# ============================================================================

# 5. VALIDATION

# ============================================================================

"""
Validate outputs:
"""

from test_elasticity_analyzer import validate_output_dataframes

issues = validate_output_dataframes(district_summary, state_summary)

if issues:
print("Validation issues found:")
for issue in issues:
print(f" - {issue}")
else:
print("✓ All validations passed!")

# ============================================================================

# 6. INTERPRETING RESULTS

# ============================================================================

"""
Understanding elasticity classifications:

HIGH ELASTICITY (score < 0.33):
✓ Meaning: Capacity increases clearly reduce anomalies
✓ Implication: System is responsive to improvements
✓ Action: Continue and expand capacity investments
✓ Example: A district where doubling update capacity cuts anomalies by 50%

MODERATE ELASTICITY (0.33 <= score < 0.67):
⚠ Meaning: Capacity helps, but some anomalies persist
⚠ Implication: Structural issues may exist alongside capacity constraints
⚠ Action: Investigate root causes while expanding capacity
⚠ Common causes: - Data quality issues - Process inefficiencies - External factors

LOW ELASTICITY (score >= 0.67):
✗ Meaning: Capacity alone won't fix this
✗ Implication: Problem is structural, not capacity-driven
✗ Action: Investigate and address root causes
✗ Common causes: - Faulty biometric equipment - Outdated processes - Data quality at source - System design flaws
"""

# ============================================================================

# 7. COMMON TASKS

# ============================================================================

# Task 1: Find all high-elasticity districts

high_elasticity = district_summary[
district_summary['elasticity_classification'] == 'High'
].sort_values('elasticity_score')

print("High Elasticity Districts:")
print(high_elasticity)

# Task 2: Identify districts needing attention

low_elasticity = district_summary[
district_summary['elasticity_classification'] == 'Low'
].sort_values('elasticity_score', ascending=False)

print("Districts Requiring Attention:")
for \_, row in low_elasticity.iterrows():
insight = generate_insight(row['state'], row['district'])
print(f"\n{row['district']}, {row['state']}")
print(f" Insight: {insight}")

# Task 3: State-level policy analysis

for \_, row in state_summary.iterrows():
print(f"\nState: {row['state']}")
print(f" Capacity-responsive: {row['capacity_responsive_pct']:.1f}%")
print(f" Low elasticity: {row['low_elasticity_pct']:.1f}%")

    if row['capacity_responsive_pct'] > 70:
        print("  → Recommendation: Expand capacity investment program")
    elif row['low_elasticity_pct'] > 30:
        print("  → Recommendation: Focus on structural improvements")

# Task 4: Export custom reports

custom_report = district_summary[
(district_summary['state'] == 'Karnataka') &
(district_summary['elasticity_classification'].isin(['High', 'Low']))
].to_csv('karnataka_critical_districts.csv', index=False)

# Task 5: Get summary statistics

print("\nElasticity Distribution:")
print(district_summary['elasticity_classification'].value_counts())
print(f"\nAverage Elasticity Score: {district_summary['elasticity_score'].mean():.3f}")

# ============================================================================

# 8. TROUBLESHOOTING

# ============================================================================

"""
Issue: All districts show "Insufficient Data"
Fix: Ensure each district has at least 3 months of history
Check: len(df.groupby('district')) and minimum months per district

Issue: No districts show "High Elasticity"
Fix: Capacity patterns may be unclear in data
Solution: Review percentile_threshold parameter or adjust min_sustained_months

Issue: Missing required columns error
Fix: Verify DataFrame has all required columns:
Required: month, state, district, demographic_updates, biometric_updates,
enrolments, anomaly_flag
Derived: total_updates, update_rate

Issue: Results seem incorrect
Fix: Validate data preparation:

- Ensure data is sorted by month
- Check that anomaly_flag contains True/False (not 0/1)
- Verify no null values in key columns
- Run validate_output_dataframes() to check results
  """

# ============================================================================

# 9. PRODUCTION DEPLOYMENT

# ============================================================================

"""
For production use:

1. Data preparation:
   - Validate and clean source data
   - Ensure consistent date format
   - Check for duplicates at (state, district, month) level
   - Document any filtering or exclusions

2. Configuration:
   - Set min_sustained_months based on domain knowledge
   - Document elasticity classification thresholds
   - Version control the analysis parameters

3. Monitoring:
   - Track elasticity scores over time
   - Set alerts for districts moving to low elasticity
   - Periodically review insights vs actual outcomes

4. Governance:
   - Document decisions made based on analysis
   - Track follow-up actions and results
   - Use insights in capacity planning meetings
     """

# ============================================================================

# 10. API REFERENCE

# ============================================================================

"""
Key Functions:

1. analyze_demand_elasticity(df, min_sustained_months=2)
   Input: DataFrame with required columns
   Output: (district_summary, state_summary, generate_insight)
   Use: Main entry point for analysis

2. generate_insight(state, district)
   Input: State and district names
   Output: Plain-language insight string
   Use: Generate human-readable explanations

3. validate_output_dataframes(district_summary, state_summary)
   Input: Analysis output DataFrames
   Output: List of validation issues
   Use: QA and testing

4. load_and_prepare_data(biometric_dir, demographic_dir, enrolment_dir)
   Input: Directory paths with CSV files
   Output: Prepared DataFrame
   Use: Loading raw data (requires specific directory structure)

Output Columns:

District Summary:

- state: State name
- district: District name
- elasticity_score: 0–1 (0=high elasticity)
- elasticity_classification: High/Moderate/Low
- expected_intervention_effectiveness: Effective/Moderately Effective/Limited
- policy_recommendation: Actionable guidance

State Summary:

- state: State name
- total_districts_analyzed: Count
- high/moderate/low_elasticity_districts: Counts
- capacity_responsive_pct: % with effective interventions
- low_elasticity_pct: % with low elasticity
- avg_elasticity_score: Mean score
  """
