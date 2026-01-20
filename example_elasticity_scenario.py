"""
Real-World Example Scenario: Aadhaar Service Elasticity Analysis
================================================================

A complete worked example showing how to use the elasticity analyzer
in a realistic governance scenario.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from demand_elasticity_analyzer import analyze_demand_elasticity


def create_realistic_scenario():
    """
    Create realistic Aadhaar service data for 5 states, 3 districts each,
    demonstrating different elasticity patterns.
    """
    
    # Define scenarios
    scenarios = {
        'Karnataka': {
            'Bangalore': {
                'pattern': 'responsive',  # Capacity increases → anomaly reduction
                'description': 'Well-managed urban center'
            },
            'Belgaum': {
                'pattern': 'partial',  # Some improvement
                'description': 'Mixed capacity and structural issues'
            },
            'Mangalore': {
                'pattern': 'unresponsive',  # Capacity doesn't help
                'description': 'Systemic issues requiring intervention'
            }
        },
        'Maharashtra': {
            'Mumbai': {
                'pattern': 'responsive',
                'description': 'High-performing metropolitan area'
            },
            'Pune': {
                'pattern': 'responsive',
                'description': 'Growing tech hub with good infrastructure'
            },
            'Aurangabad': {
                'pattern': 'partial',
                'description': 'Moderate capacity challenges'
            }
        },
        'Tamil Nadu': {
            'Chennai': {
                'pattern': 'responsive',
                'description': 'Metropolitan efficiency'
            },
            'Coimbatore': {
                'pattern': 'partial',
                'description': 'Industrial area with mixed results'
            },
            'Villupuram': {
                'pattern': 'unresponsive',
                'description': 'Rural area with systemic issues'
            }
        },
        'Uttar Pradesh': {
            'Lucknow': {
                'pattern': 'partial',
                'description': 'State capital, mixed results'
            },
            'Kanpur': {
                'pattern': 'unresponsive',
                'description': 'Industrial area with data quality issues'
            },
            'Agra': {
                'pattern': 'partial',
                'description': 'Tourist destination, volatile patterns'
            }
        },
        'West Bengal': {
            'Kolkata': {
                'pattern': 'responsive',
                'description': 'Metro area with good governance'
            },
            'Howrah': {
                'pattern': 'partial',
                'description': 'Suburban challenges'
            },
            'Durgapur': {
                'pattern': 'unresponsive',
                'description': 'Industrial area requiring attention'
            }
        }
    }
    
    # Create 24 months of data (2 years)
    months = 24
    base_date = datetime(2022, 1, 1)
    
    data = []
    
    for state, districts in scenarios.items():
        for district, config in districts.items():
            pattern = config['pattern']
            
            # Base values
            if 'Mumbai' in district or 'Bangalore' in district or 'Chennai' in district:
                base_enrol = 5000  # Large urban center
                base_capacity = 2000
            elif 'Pune' in district or 'Kanpur' in district or 'Kolkata' in district:
                base_enrol = 3000  # Medium city
                base_capacity = 1200
            else:
                base_enrol = 1500  # Smaller district
                base_capacity = 600
            
            for month_idx in range(months):
                month = base_date + timedelta(days=30*month_idx)
                
                # Generate time series based on pattern
                if pattern == 'responsive':
                    # Capacity increases over time
                    enrolments = int(base_enrol * (1 + 0.02 * month_idx))
                    capacity = int(base_capacity * (1 + 0.03 * month_idx))
                    # Anomalies decrease with capacity
                    anomaly_prob = max(0.05, 0.20 - 0.004 * month_idx)
                
                elif pattern == 'partial':
                    # Slow capacity increase
                    enrolments = int(base_enrol * (1 + 0.01 * month_idx))
                    capacity = int(base_capacity * (1 + 0.015 * month_idx))
                    # Anomalies decrease slowly
                    anomaly_prob = max(0.15, 0.30 - 0.002 * month_idx)
                
                elif pattern == 'unresponsive':
                    # Capacity varies but doesn't help
                    enrolments = int(base_enrol * (1 + 0.005 * month_idx + np.random.normal(0, 0.05)))
                    capacity = int(base_capacity * (1 + 0.02 * month_idx + np.random.normal(0, 0.08)))
                    # Anomalies stay high regardless
                    anomaly_prob = 0.30 + np.random.normal(0, 0.05)
                
                # Add noise
                enrolments = max(100, int(enrolments * (1 + np.random.normal(0, 0.05))))
                capacity = max(50, int(capacity * (1 + np.random.normal(0, 0.05))))
                
                # Split capacity
                demographic = int(capacity * 0.6)
                biometric = int(capacity * 0.4)
                
                # Generate anomaly flag
                anomaly_flag = np.random.random() < anomaly_prob
                
                data.append({
                    'month': month,
                    'state': state,
                    'district': district,
                    'demographic_updates': demographic,
                    'biometric_updates': biometric,
                    'enrolments': enrolments,
                    'anomaly_flag': anomaly_flag,
                    'anomaly_persistent': anomaly_flag and month_idx > 12  # Persistence check
                })
    
    df = pd.DataFrame(data)
    df['total_updates'] = df['demographic_updates'] + df['biometric_updates']
    df['update_rate'] = df['total_updates'] / df['enrolments']
    df = df.sort_values(['state', 'district', 'month']).reset_index(drop=True)
    
    return df, scenarios


def run_realistic_example():
    """Execute complete elasticity analysis on realistic scenario."""
    
    print("=" * 80)
    print("AADHAAR SERVICE ELASTICITY ANALYSIS - REALISTIC SCENARIO")
    print("=" * 80 + "\n")
    
    # Step 1: Create scenario data
    print("STEP 1: Creating realistic scenario data...")
    df, scenarios = create_realistic_scenario()
    print(f"✓ Generated {len(df)} records across 15 districts in 5 states")
    print(f"  Date range: {df['month'].min().date()} to {df['month'].max().date()}")
    print(f"  States: {df['state'].nunique()}")
    print(f"  Districts: {df['district'].nunique()}\n")
    
    # Step 2: Run elasticity analysis
    print("STEP 2: Running elasticity analysis...")
    district_summary, state_summary, generate_insight = analyze_demand_elasticity(df)
    print("✓ Analysis complete\n")
    
    # Step 3: Display results
    print("=" * 80)
    print("DISTRICT-LEVEL RESULTS")
    print("=" * 80)
    print(district_summary.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("STATE-LEVEL SUMMARY")
    print("=" * 80)
    print(state_summary.to_string(index=False))
    
    # Step 4: Governance insights
    print("\n" + "=" * 80)
    print("GOVERNANCE INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)
    
    for state in district_summary['state'].unique():
        state_data = district_summary[district_summary['state'] == state]
        print(f"\n{state}:")
        print(f"  Average Elasticity: {state_data['elasticity_score'].mean():.3f}")
        
        high = len(state_data[state_data['elasticity_classification'] == 'High'])
        low = len(state_data[state_data['elasticity_classification'] == 'Low'])
        
        if high > 1:
            print(f"  ✓ {high} districts with high elasticity - exemplary for capacity strategy")
        
        if low > 1:
            print(f"  ⚠ {low} districts with low elasticity - require structural intervention")
        
        # Show specific recommendations
        low_districts = state_data[state_data['elasticity_classification'] == 'Low']
        if len(low_districts) > 0:
            print(f"\n  Districts requiring attention:")
            for _, row in low_districts.iterrows():
                insight = generate_insight(row['state'], row['district'])
                print(f"    • {row['district']}")
                print(f"      {row['policy_recommendation']}")
    
    # Step 5: Strategic recommendations
    print("\n" + "=" * 80)
    print("STRATEGIC RECOMMENDATIONS")
    print("=" * 80)
    
    overall_capacity_responsive = state_summary['capacity_responsive_pct'].mean()
    overall_low_elasticity = state_summary['low_elasticity_pct'].mean()
    
    print(f"\nNational Overview:")
    print(f"  • Capacity-responsive districts: {overall_capacity_responsive:.1f}%")
    print(f"  • Low-elasticity districts: {overall_low_elasticity:.1f}%")
    
    if overall_capacity_responsive > 70:
        print("\n  RECOMMENDATION: Increase capacity investment across all states")
        print("  RATIONALE: Strong elasticity indicates capacity is effective lever")
    elif overall_capacity_responsive > 50:
        print("\n  RECOMMENDATION: Targeted capacity investment with structural audit")
        print("  RATIONALE: Mixed results suggest some capacity + some structural issues")
    else:
        print("\n  RECOMMENDATION: Pause capacity expansion, focus on structural improvements")
        print("  RATIONALE: Low elasticity indicates capacity alone won't solve issues")
    
    # High-performing states
    best_states = state_summary.nlargest(2, 'capacity_responsive_pct')
    if len(best_states) > 0:
        print(f"\nBest-Performing States (adopt as model):")
        for _, row in best_states.iterrows():
            print(f"  • {row['state']}: {row['capacity_responsive_pct']:.1f}% capacity-responsive")
    
    # States needing help
    struggling_states = state_summary[state_summary['low_elasticity_pct'] > 40]
    if len(struggling_states) > 0:
        print(f"\nStates Requiring Intervention:")
        for _, row in struggling_states.iterrows():
            print(f"  • {row['state']}: {row['low_elasticity_pct']:.1f}% low-elasticity districts")
            print(f"    → Recommend: Structural audit + process improvement focus")
    
    # Step 6: Implementation roadmap
    print("\n" + "=" * 80)
    print("90-DAY IMPLEMENTATION ROADMAP")
    print("=" * 80)
    
    print("""
PHASE 1 (Days 1-30): Analysis & Planning
  □ Share results with state/district leadership
  □ Validate findings with field teams
  □ Identify root causes in low-elasticity districts
  □ Plan interventions (capacity vs. structural)

PHASE 2 (Days 31-60): Targeted Interventions
  □ Launch capacity expansion in responsive districts
  □ Initiate root cause analysis in unresponsive districts
  □ Implement process improvements in partial-response districts
  □ Establish KPI tracking and monitoring

PHASE 3 (Days 61-90): Monitoring & Optimization
  □ Track elasticity scores weekly
  □ Adjust interventions based on results
  □ Document lessons learned
  □ Plan next analysis cycle (3 months)

SUCCESS METRICS:
  • 10% increase in capacity-responsive districts
  • 20% reduction in low-elasticity districts
  • Anomaly frequency down 15% in responsive districts
  • Root causes identified in 100% of low-elasticity districts
""")
    
    # Step 7: Save outputs
    print("=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)
    
    district_summary.to_csv('elasticity_district_summary_example.csv', index=False)
    state_summary.to_csv('elasticity_state_summary_example.csv', index=False)
    print("\n✓ Results saved to:")
    print("  • elasticity_district_summary_example.csv")
    print("  • elasticity_state_summary_example.csv")
    
    return district_summary, state_summary, generate_insight


def demonstrate_insights(district_summary, generate_insight):
    """Show insight generation examples."""
    
    print("\n" + "=" * 80)
    print("SAMPLE INSIGHTS FOR POLICYMAKERS")
    print("=" * 80)
    
    # Pick interesting examples
    high = district_summary[district_summary['elasticity_classification'] == 'High'].iloc[0]
    low = district_summary[district_summary['elasticity_classification'] == 'Low'].iloc[0]
    moderate = district_summary[district_summary['elasticity_classification'] == 'Moderate'].iloc[0]
    
    for row in [high, low, moderate]:
        state, district = row['state'], row['district']
        insight = generate_insight(state, district)
        classification = row['elasticity_classification']
        score = row['elasticity_score']
        
        print(f"\n{district}, {state}")
        print(f"Classification: {classification}")
        print(f"Elasticity Score: {score:.3f}")
        print(f"Insight:")
        print(f"  {insight}")


if __name__ == '__main__':
    # Run the complete scenario
    district_summary, state_summary, generate_insight = run_realistic_example()
    
    # Show sample insights
    demonstrate_insights(district_summary, generate_insight)
    
    print("\n" + "=" * 80)
    print("✓ SCENARIO ANALYSIS COMPLETE")
    print("=" * 80)
