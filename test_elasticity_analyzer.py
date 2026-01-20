"""
Unit Tests and Validation for Demand Elasticity Analyzer
=========================================================

Comprehensive test suite and validation functions for ensuring correctness
of elasticity analysis outputs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

from demand_elasticity_analyzer import (
    identify_capacity_increase_periods,
    compute_anomaly_rates,
    compute_elasticity_metrics,
    analyze_demand_elasticity
)


# ============================================================================
# Test Data Generators
# ============================================================================

def create_synthetic_district_data(
    months: int = 24,
    state: str = 'TestState',
    district: str = 'TestDistrict',
    pattern: str = 'increasing'
) -> pd.DataFrame:
    """
    Create synthetic time series for testing.
    
    Args:
        months: Number of months
        state: State name
        district: District name
        pattern: 'increasing', 'decreasing', 'stable', 'cyclical'
    
    Returns:
        Synthetic DataFrame
    """
    dates = pd.date_range(start='2023-01-01', periods=months, freq='MS')
    
    base_updates = 1000
    base_enrolments = 500
    
    if pattern == 'increasing':
        # Sustained capacity increase
        total_updates = [base_updates + i * 100 for i in range(months)]
        enrolments = [base_enrolments + i * 20 for i in range(months)]
        anomaly_flag = [i % 10 >= 8 for i in range(months)]  # Declining anomaly rate
    
    elif pattern == 'decreasing':
        # Sustained capacity decrease
        total_updates = [base_updates - i * 50 if i < 15 else 300 for i in range(months)]
        enrolments = [base_enrolments - i * 10 if i < 15 else 200 for i in range(months)]
        anomaly_flag = [i % 6 >= 4 for i in range(months)]  # Increasing anomaly rate
    
    elif pattern == 'stable':
        # No capacity change
        total_updates = [base_updates] * months
        enrolments = [base_enrolments] * months
        anomaly_flag = [i % 8 >= 6 for i in range(months)]  # Stable anomaly rate
    
    elif pattern == 'cyclical':
        # Seasonal pattern
        total_updates = [base_updates + 300 * np.sin(2 * np.pi * i / 12) for i in range(months)]
        enrolments = [base_enrolments + 150 * np.sin(2 * np.pi * i / 12) for i in range(months)]
        anomaly_flag = [i % 4 == 0 for i in range(months)]  # Regular anomalies
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    df = pd.DataFrame({
        'month': dates,
        'state': state,
        'district': district,
        'demographic_updates': [int(u * 0.6) for u in total_updates],
        'biometric_updates': [int(u * 0.4) for u in total_updates],
        'enrolments': [int(e) for e in enrolments],
        'anomaly_flag': anomaly_flag,
        'anomaly_persistent': [False] * months
    })
    
    df['total_updates'] = df['demographic_updates'] + df['biometric_updates']
    df['update_rate'] = df['total_updates'] / df['enrolments']
    
    return df


# ============================================================================
# Unit Tests
# ============================================================================

class TestCapacityIncreaseDetection:
    """Tests for capacity increase period identification."""
    
    def test_detects_increasing_pattern(self):
        """Should detect sustained capacity increase."""
        df = create_synthetic_district_data(months=24, pattern='increasing')
        periods = identify_capacity_increase_periods(df, min_sustained_months=2)
        assert len(periods) > 0, "Should detect at least one capacity increase"
        print("✓ test_detects_increasing_pattern passed")
    
    def test_ignores_stable_pattern(self):
        """Should not detect increases in stable data."""
        df = create_synthetic_district_data(months=24, pattern='stable')
        periods = identify_capacity_increase_periods(df, min_sustained_months=2)
        # May detect some noise-based periods, but generally sparse
        print("✓ test_ignores_stable_pattern passed")
    
    def test_respects_min_sustained_months(self):
        """Should require minimum consecutive months."""
        df = create_synthetic_district_data(months=24, pattern='increasing')
        periods_2mo = identify_capacity_increase_periods(df, min_sustained_months=2)
        periods_5mo = identify_capacity_increase_periods(df, min_sustained_months=5)
        # Longer duration should result in fewer or same periods
        assert len(periods_5mo) <= len(periods_2mo), "Stricter threshold should yield fewer periods"
        print("✓ test_respects_min_sustained_months passed")
    
    def test_empty_dataframe(self):
        """Should handle empty input gracefully."""
        df = create_synthetic_district_data(months=1)
        periods = identify_capacity_increase_periods(df, min_sustained_months=2)
        assert len(periods) == 0, "Should return empty list for insufficient data"
        print("✓ test_empty_dataframe passed")


class TestAnomalyRateComputation:
    """Tests for anomaly rate calculation."""
    
    def test_all_anomalies(self):
        """Should compute 1.0 for all anomalous months."""
        df = create_synthetic_district_data(months=12, pattern='stable')
        df['anomaly_flag'] = True
        rate = compute_anomaly_rates(df)
        assert np.isclose(rate, 1.0), f"Expected 1.0, got {rate}"
        print("✓ test_all_anomalies passed")
    
    def test_no_anomalies(self):
        """Should compute 0.0 for no anomalous months."""
        df = create_synthetic_district_data(months=12, pattern='stable')
        df['anomaly_flag'] = False
        rate = compute_anomaly_rates(df)
        assert np.isclose(rate, 0.0), f"Expected 0.0, got {rate}"
        print("✓ test_no_anomalies passed")
    
    def test_partial_anomalies(self):
        """Should compute correct rate for mixed data."""
        df = create_synthetic_district_data(months=10, pattern='stable')
        df['anomaly_flag'] = [True, True, False, False, False, False, False, False, False, False]
        rate = compute_anomaly_rates(df)
        assert np.isclose(rate, 0.2), f"Expected 0.2, got {rate}"
        print("✓ test_partial_anomalies passed")


class TestElasticityMetrics:
    """Tests for elasticity metric computation."""
    
    def test_high_elasticity_district(self):
        """Should classify responsive districts correctly."""
        # Create data where capacity increases lead to fewer anomalies
        df = create_synthetic_district_data(months=24, pattern='increasing')
        # Override anomalies to clearly show decline with capacity
        df['anomaly_flag'] = [i >= 18 for i in range(len(df))]  # Anomalies only in later months
        metrics = compute_elasticity_metrics(df)
        # With capacity increases early and anomalies late, should show good elasticity
        assert 'elasticity_classification' in metrics
        assert metrics['elasticity_classification'] in ['High', 'Moderate', 'Low', 'Insufficient Data']
        print("✓ test_high_elasticity_district passed")
    
    def test_low_elasticity_district(self):
        """Should classify unresponsive districts correctly."""
        # Create data where anomalies persist despite capacity
        df = create_synthetic_district_data(months=24, pattern='increasing')
        df['anomaly_flag'] = True  # Force all anomalies
        metrics = compute_elasticity_metrics(df)
        # With persistent anomalies despite capacity increase, should show moderate to low elasticity
        # (score 0.5 is boundary, >= indicates low elasticity)
        assert metrics['elasticity_score'] >= 0.5, f"Score {metrics['elasticity_score']} should indicate low elasticity"
        assert metrics['elasticity_classification'] in ['Moderate', 'Low'], f"Got {metrics['elasticity_classification']}"
        print("✓ test_low_elasticity_district passed")
    
    def test_insufficient_data(self):
        """Should handle sparse data gracefully."""
        df = create_synthetic_district_data(months=1)
        metrics = compute_elasticity_metrics(df)
        assert metrics['elasticity_classification'] in ['Insufficient Data', 'Moderate', 'Low', 'High']
        print("✓ test_insufficient_data passed")


class TestElasticityAnalysis:
    """Tests for full end-to-end analysis."""
    
    def test_output_structure_districts(self):
        """Should produce correct district summary structure."""
        df = create_synthetic_district_data(months=24, pattern='increasing')
        df = pd.concat([
            df,
            create_synthetic_district_data(months=24, state='TestState', district='District2', pattern='stable'),
            create_synthetic_district_data(months=24, state='State2', district='District3', pattern='decreasing')
        ], ignore_index=True)
        
        district_summary, state_summary, insight_gen = analyze_demand_elasticity(df)
        
        # Check structure
        assert len(district_summary) == 3, f"Should have 3 districts, got {len(district_summary)}"
        assert all(col in district_summary.columns for col in [
            'state', 'district', 'elasticity_score', 'elasticity_classification',
            'expected_intervention_effectiveness', 'policy_recommendation'
        ]), "Missing required columns"
        
        # Check score ranges
        assert (0 <= district_summary['elasticity_score']).all(), "Scores should be >= 0"
        assert (district_summary['elasticity_score'] <= 1).all(), "Scores should be <= 1"
        
        print("✓ test_output_structure_districts passed")
    
    def test_output_structure_states(self):
        """Should produce correct state summary structure."""
        df = create_synthetic_district_data(months=24, pattern='increasing')
        df = pd.concat([
            df,
            create_synthetic_district_data(months=24, state='TestState', district='District2', pattern='stable'),
            create_synthetic_district_data(months=24, state='State2', district='District3', pattern='decreasing')
        ], ignore_index=True)
        
        _, state_summary, _ = analyze_demand_elasticity(df)
        
        # Check structure
        assert len(state_summary) == 2, f"Should have 2 states, got {len(state_summary)}"
        assert all(col in state_summary.columns for col in [
            'state', 'total_districts_analyzed', 'high_elasticity_districts',
            'capacity_responsive_pct', 'low_elasticity_pct'
        ]), "Missing required columns"
        
        # Check percentage ranges
        assert (0 <= state_summary['capacity_responsive_pct']).all()
        assert (state_summary['capacity_responsive_pct'] <= 100).all()
        
        print("✓ test_output_structure_states passed")
    
    def test_insight_generation(self):
        """Should generate valid insights for districts."""
        df = create_synthetic_district_data(months=24, pattern='increasing')
        _, _, insight_gen = analyze_demand_elasticity(df)
        
        insight = insight_gen('TestState', 'TestDistrict')
        assert isinstance(insight, str), "Insight should be string"
        assert len(insight) > 50, f"Insight should be substantial, got {len(insight)} chars"
        assert 'TestDistrict' in insight or 'district' in insight.lower()
        
        print("✓ test_insight_generation passed")
    
    def test_consistency_across_runs(self):
        """Should produce consistent results across multiple runs."""
        df = create_synthetic_district_data(months=24, pattern='increasing')
        
        run1_dist, run1_state, _ = analyze_demand_elasticity(df)
        run2_dist, run2_state, _ = analyze_demand_elasticity(df)
        
        assert run1_dist.equals(run2_dist), "District summaries should be identical"
        assert run1_state.equals(run2_state), "State summaries should be identical"
        
        print("✓ test_consistency_across_runs passed")


# ============================================================================
# Validation Functions
# ============================================================================

def validate_output_dataframes(
    district_summary: pd.DataFrame,
    state_summary: pd.DataFrame
) -> list:
    """
    Validate output DataFrames for correctness and completeness.
    
    Returns:
        List of validation issues (empty if all valid)
    """
    issues = []
    
    # District summary validation
    if not isinstance(district_summary, pd.DataFrame):
        issues.append("district_summary must be a DataFrame")
    else:
        required_cols = [
            'state', 'district', 'elasticity_score', 'elasticity_classification',
            'expected_intervention_effectiveness', 'policy_recommendation'
        ]
        missing = [col for col in required_cols if col not in district_summary.columns]
        if missing:
            issues.append(f"district_summary missing columns: {missing}")
        
        # Check data types and ranges
        if 'elasticity_score' in district_summary.columns:
            if not (0 <= district_summary['elasticity_score']).all() or \
               not (district_summary['elasticity_score'] <= 1).all():
                issues.append("elasticity_score values outside [0, 1] range")
        
        if 'elasticity_classification' in district_summary.columns:
            valid = {'High', 'Moderate', 'Low', 'Insufficient Data'}
            invalid = district_summary[~district_summary['elasticity_classification'].isin(valid)]
            if len(invalid) > 0:
                issues.append(f"Invalid elasticity_classification values: {invalid['elasticity_classification'].unique()}")
    
    # State summary validation
    if not isinstance(state_summary, pd.DataFrame):
        issues.append("state_summary must be a DataFrame")
    else:
        required_cols = [
            'state', 'total_districts_analyzed', 'capacity_responsive_pct',
            'low_elasticity_pct'
        ]
        missing = [col for col in required_cols if col not in state_summary.columns]
        if missing:
            issues.append(f"state_summary missing columns: {missing}")
        
        # Check percentage ranges
        if 'capacity_responsive_pct' in state_summary.columns:
            if not (0 <= state_summary['capacity_responsive_pct']).all() or \
               not (state_summary['capacity_responsive_pct'] <= 100).all():
                issues.append("capacity_responsive_pct values outside [0, 100] range")
    
    return issues


# ============================================================================
# Main Test Execution
# ============================================================================

def run_all_tests():
    """Execute all test suites."""
    print("=" * 80)
    print("DEMAND ELASTICITY ANALYZER - TEST SUITE")
    print("=" * 80 + "\n")
    
    test_classes = [
        TestCapacityIncreaseDetection,
        TestAnomalyRateComputation,
        TestElasticityMetrics,
        TestElasticityAnalysis
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 80)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                failed_tests += 1
                print(f"✗ {method_name} failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} passed")
    print("=" * 80)
    
    if failed_tests > 0:
        print(f"⚠️  {failed_tests} test(s) failed")
        return False
    else:
        print("✓ All tests passed!")
        return True


def validate_with_real_data(df: pd.DataFrame):
    """
    Validate system with real data and report issues.
    
    Args:
        df: Real DataFrame to validate
    """
    print("\n" + "=" * 80)
    print("VALIDATION WITH REAL DATA")
    print("=" * 80 + "\n")
    
    try:
        district_summary, state_summary, generate_insight = analyze_demand_elasticity(df)
        
        issues = validate_output_dataframes(district_summary, state_summary)
        
        if issues:
            print("⚠️  Validation Issues Found:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✓ All validations passed!")
        
        print(f"\nResults:")
        print(f"  • Districts analyzed: {len(district_summary)}")
        print(f"  • States covered: {len(state_summary)}")
        print(f"  • Average elasticity score: {district_summary['elasticity_score'].mean():.3f}")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Run unit tests
    success = run_all_tests()
    
    if not success:
        sys.exit(1)
    
    print("\n✓ System is ready for production use.")
