import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function to calculate metric multiplier
def calculate_multiplier(achievement, is_signed_mm=False, tcv_achievement=0):
    if achievement < 85:
        return 0.0
    
    if is_signed_mm and tcv_achievement < 50:
        return 0.0
    
    if achievement < 100:
        # Linear interpolation between 85% (0.5) and 100% (1.0)
        return 0.5 + (achievement - 85) * (0.5 / 15)
    
    if achievement >= 100:
        # 0.5 multiplier increase for each 1% above 100%
        multiplier = 1.0 + (achievement - 100) * 0.5
        return min(multiplier, 2.0)
    
    return 0.0

# Calculate quarterly bonus
def calculate_quarterly_bonus(otb, quarter, metrics, apply_qtr_cap=True):
    # Calculate achievement percentages
    indiv_rev_ach = (metrics['indiv_rev_actual'] / metrics['indiv_rev_target']) * 100 if metrics['indiv_rev_target'] else 0
    tcv_ach = (metrics['tcv_actual'] / metrics['tcv_target']) * 100 if metrics['tcv_target'] else 0
    signed_mm_ach = (metrics['signed_mm_actual'] / metrics['signed_mm_target']) * 100 if metrics['signed_mm_target'] else 0
    sl_rev_ach = (metrics['sl_rev_actual'] / metrics['sl_rev_target']) * 100 if metrics['sl_rev_target'] else 0
    margin_ach = (metrics['margin_actual'] / metrics['margin_target']) * 100 if metrics['margin_target'] else 0
    
    # Calculate multipliers
    rev_mult = calculate_multiplier(indiv_rev_ach)
    tcv_mult = calculate_multiplier(tcv_ach)
    signed_mm_mult = calculate_multiplier(signed_mm_ach, is_signed_mm=True, tcv_achievement=tcv_ach)
    sl_rev_mult = calculate_multiplier(sl_rev_ach)
    margin_mult = calculate_multiplier(margin_ach)
    
    # Apply quarterly cap for Q1-Q3
    if quarter < 4 and apply_qtr_cap:
        rev_mult = min(rev_mult, 1.0)
        tcv_mult = min(tcv_mult, 1.0)
        signed_mm_mult = min(signed_mm_mult, 1.0)
        sl_rev_mult = min(sl_rev_mult, 1.0)
        margin_mult = min(margin_mult, 1.0)
    
    # Calculate weighted scores
    individual_score = (rev_mult * 0.4 + tcv_mult * 0.3 + signed_mm_mult * 0.3) * 0.7
    service_score = (sl_rev_mult * 0.5 + margin_mult * 0.5) * 0.3
    overall_percentage = individual_score + service_score
    
    # Quarterly bonus (assume OTB is annual, so divide by 4 for quarterly portion)
    quarterly_bonus = (otb / 4) * overall_percentage
    
    return {
        'quarterly_bonus': quarterly_bonus,
        'overall_percentage': overall_percentage * 100,
        'multipliers': {
            'indiv_rev': rev_mult,
            'tcv': tcv_mult,
            'signed_mm': signed_mm_mult,
            'sl_rev': sl_rev_mult,
            'margin': margin_mult
        },
        'achievements': {
            'indiv_rev': indiv_rev_ach,
            'tcv': tcv_ach,
            'signed_mm': signed_mm_ach,
            'sl_rev': sl_rev_ach,
            'margin': margin_ach
        }
    }

# Streamlit app
def main():
    st.set_page_config(page_title="Bonus Payout Calculator", layout="wide")
    st.title("🏆 Quarterly Bonus Payout Calculator")
    st.markdown("Calculate bonus payouts based on policy guidelines with quarterly projections and full year reconciliation.")
    
    # Initialize session state for quarterly data
    if 'quarterly_data' not in st.session_state:
        st.session_state.quarterly_data = {q: {} for q in range(1, 5)}
    
    # Input section
    with st.sidebar:
        st.header("Global Parameters")
        otb = st.number_input("Annual On Target Bonus (OTB)", min_value=0, value=10000, step=1000)
        
    # Tabs for Quarterly Calculator and Full Year Projection
    tab1, tab2 = st.tabs(["Quarterly Calculator", "Full Year Projection"])
    
    with tab1:
        st.header("Quarterly Bonus Calculation")
        col1, col2 = st.columns(2)
        
        with col1:
            quarter = st.selectbox("Select Quarter", [1, 2, 3, 4], key="qtr_select")
            st.subheader(f"Q{quarter} Metrics")
            
            # Individual metrics
            st.markdown("**Individual Metrics (70%)**")
            indiv_rev_target = st.number_input("Reported Revenue - Target", min_value=0.0, value=100.0, key=f"indiv_rev_t_q{quarter}")
            indiv_rev_actual = st.number_input("Reported Revenue - Actual", min_value=0.0, value=95.0, key=f"indiv_rev_a_q{quarter}")
            
            tcv_target = st.number_input("New Deal Closure TCV - Target", min_value=0.0, value=100.0, key=f"tcv_t_q{quarter}")
            tcv_actual = st.number_input("New Deal Closure TCV - Actual", min_value=0.0, value=105.0, key=f"tcv_a_q{quarter}")
            
            signed_mm_target = st.number_input("Signed MM for New Deals - Target", min_value=0.0, value=100.0, key=f"mm_t_q{quarter}")
            signed_mm_actual = st.number_input("Signed MM for New Deals - Actual", min_value=0.0, value=110.0, key=f"mm_a_q{quarter}")
            
            # Service line metrics
            st.markdown("**Service Line Metrics (30%)**")
            sl_rev_target = st.number_input("Service Line Revenue - Target", min_value=0.0, value=100.0, key=f"sl_rev_t_q{quarter}")
            sl_rev_actual = st.number_input("Service Line Revenue - Actual", min_value=0.0, value=120.0, key=f"sl_rev_a_q{quarter}")
            
            margin_target = st.number_input("Managed Margin - Target", min_value=0.0, value=100.0, key=f"margin_t_q{quarter}")
            margin_actual = st.number_input("Managed Margin - Actual", min_value=0.0, value=90.0, key=f"margin_a_q{quarter}")
            
            # Store data
            st.session_state.quarterly_data[quarter] = {
                'indiv_rev_target': indiv_rev_target,
                'indiv_rev_actual': indiv_rev_actual,
                'tcv_target': tcv_target,
                'tcv_actual': tcv_actual,
                'signed_mm_target': signed_mm_target,
                'signed_mm_actual': signed_mm_actual,
                'sl_rev_target': sl_rev_target,
                'sl_rev_actual': sl_rev_actual,
                'margin_target': margin_target,
                'margin_actual': margin_actual
            }
        
        with col2:
            if st.button("Calculate Quarterly Bonus", key=f"calc_q{quarter}"):
                metrics = {
                    'indiv_rev_target': indiv_rev_target,
                    'indiv_rev_actual': indiv_rev_actual,
                    'tcv_target': tcv_target,
                    'tcv_actual': tcv_actual,
                    'signed_mm_target': signed_mm_target,
                    'signed_mm_actual': signed_mm_actual,
                    'sl_rev_target': sl_rev_target,
                    'sl_rev_actual': sl_rev_actual,
                    'margin_target': margin_target,
                    'margin_actual': margin_actual
                }
                
                result = calculate_quarterly_bonus(otb, quarter, metrics)
                
                st.subheader(f"Q{quarter} Bonus Results")
                
                # Display multipliers and achievements
                st.markdown("**Metric Performance**")
                metric_data = []
                for metric in ['indiv_rev', 'tcv', 'signed_mm', 'sl_rev', 'margin']:
                    metric_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Achievement %': f"{result['achievements'][metric]:.1f}%",
                        'Multiplier': f"{result['multipliers'][metric]:.2f}",
                        'Capped': 'Capped' if quarter < 4 and result['multipliers'][metric] > 1.0 else '' 
                    })
                
                st.table(pd.DataFrame(metric_data))
                
                # Display scores
                st.markdown("**Performance Scores**")
                score_data = {
                    'Component': ['Individual Performance (70%)', 'Service Line Performance (30%)', 'Overall Score'],
                    'Value': [
                        f"{result['overall_percentage'] * 0.7:.1f}%", 
                        f"{result['overall_percentage'] * 0.3:.1f}%", 
                        f"{result['overall_percentage']:.1f}%"
                    ]
                }
                st.table(pd.DataFrame(score_data))
                
                # Display bonus
                st.markdown("**Bonus Calculation**")
                st.metric("Quarterly Bonus Amount", f"${result['quarterly_bonus']:,.2f}")
                st.metric("Quarterly Bonus as % of Quarterly OTB", f"{result['overall_percentage']:.1f}%")
                
                # Visualize achievement
                fig, ax = plt.subplots(figsize=(8, 4))
                metrics_list = ['Reported Rev', 'New Deal TCV', 'Signed MM', 'SL Revenue', 'Margin']
                achievements = [result['achievements']['indiv_rev'], 
                              result['achievements']['tcv'], 
                              result['achievements']['signed_mm'], 
                              result['achievements']['sl_rev'], 
                              result['achievements']['margin']]
                
                ax.bar(metrics_list, achievements, color='skyblue')
                ax.axhline(y=85, color='r', linestyle='--', label='Gate (85%)')
                ax.set_ylabel('Achievement %')
                ax.set_title('Metric Achievements')
                ax.legend()
                st.pyplot(fig)
    
    with tab2:
        st.header("Full Year Bonus Projection")
        st.info("Enter data for all quarters to see the full year projection and reconciliation")
        
        # Check if all quarters have data
        all_data_entered = all(st.session_state.quarterly_data[q] for q in range(1, 5))
        
        if not all_data_entered:
            st.warning("Please enter data for all quarters in the Quarterly Calculator tab")
            return
        
        # Calculate quarterly bonuses with and without caps
        quarterly_results = {}
        quarterly_paid = {}
        quarterly_true = {}
        
        for q in range(1, 5):
            metrics = st.session_state.quarterly_data[q]
            # With quarterly cap (as paid)
            paid_result = calculate_quarterly_bonus(otb, q, metrics, apply_qtr_cap=True)
            quarterly_paid[q] = paid_result['quarterly_bonus']
            
            # Without quarterly cap (true performance)
            true_result = calculate_quarterly_bonus(otb, q, metrics, apply_qtr_cap=False)
            quarterly_true[q] = true_result['quarterly_bonus']
            
            quarterly_results[q] = {
                'paid': paid_result,
                'true': true_result
            }
        
        # Calculate annual totals
        total_paid_q1q3 = sum(quarterly_paid[q] for q in range(1, 4))
        total_true = sum(quarterly_true.values())
        
        # Apply annual cap (200% of OTB)
        capped_bonus = min(total_true, 2 * otb)
        
        # Calculate Q4 payout (true minus what was already paid in Q1-Q3)
        q4_payout = max(0, capped_bonus - total_paid_q1q3)
        
        # Display results
        st.subheader("Quarterly Bonus Summary")
        q_data = []
        for q in range(1, 5):
            q_data.append({
                'Quarter': f"Q{q}",
                'Paid Bonus': f"${quarterly_paid[q]:,.2f}",
                'True Bonus': f"${quarterly_true[q]:,.2f}",
                'Difference': f"${quarterly_true[q] - quarterly_paid[q]:,.2f}"
            })
        st.table(pd.DataFrame(q_data))
        
        st.subheader("Full Year Reconciliation")
        st.markdown(f"**Annual OTB:** ${otb:,.2f}")
        st.markdown(f"**Total Paid in Q1-Q3:** ${total_paid_q1q3:,.2f}")
        st.markdown(f"**Total True Bonus (without caps):** ${total_true:,.2f}")
        st.markdown(f"**200% OTB Cap:** ${2 * otb:,.2f}")
        st.markdown(f"**Capped Annual Bonus:** ${capped_bonus:,.2f}")
        
        st.success(f"**Q4 Payout Amount:** ${q4_payout:,.2f}")
        st.success(f"**Total Annual Bonus:** ${total_paid_q1q3 + q4_payout:,.2f}")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        quarters = [f"Q{q}" for q in range(1, 5)]
        paid_values = [quarterly_paid[q] for q in range(1, 5)]
        true_values = [quarterly_true[q] for q in range(1, 5)]
        
        bar_width = 0.35
        index = np.arange(len(quarters))
        
        ax.bar(index, paid_values, bar_width, label='Paid (with caps)')
        ax.bar(index + bar_width, true_values, bar_width, label='True (without caps)')
        
        ax.axhline(y=2 * otb, color='r', linestyle='--', label='200% OTB Cap')
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Bonus Amount ($)')
        ax.set_title('Quarterly Bonus Comparison')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(quarters)
        ax.legend()
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()