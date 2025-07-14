import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

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
    
    # Calculate RAW multipliers (before quarterly cap)
    rev_raw = calculate_multiplier(indiv_rev_ach)
    tcv_raw = calculate_multiplier(tcv_ach)
    signed_mm_raw = calculate_multiplier(signed_mm_ach, is_signed_mm=True, tcv_achievement=tcv_ach)
    sl_rev_raw = calculate_multiplier(sl_rev_ach)
    margin_raw = calculate_multiplier(margin_ach)
    
    # Apply quarterly cap for Q1-Q3
    if quarter < 4 and apply_qtr_cap:
        rev_final = min(rev_raw, 1.0)
        tcv_final = min(tcv_raw, 1.0)
        signed_mm_final = min(signed_mm_raw, 1.0)
        sl_rev_final = min(sl_rev_raw, 1.0)
        margin_final = min(margin_raw, 1.0)
    else:
        rev_final = rev_raw
        tcv_final = tcv_raw
        signed_mm_final = signed_mm_raw
        sl_rev_final = sl_rev_raw
        margin_final = margin_raw
    
    # Calculate weighted scores
    individual_score = (rev_final * 0.4 + tcv_final * 0.3 + signed_mm_final * 0.3) * 0.7
    service_score = (sl_rev_final * 0.5 + margin_final * 0.5) * 0.3
    overall_percentage = individual_score + service_score
    
    # Quarterly bonus (assume OTB is annual, so divide by 4 for quarterly portion)
    quarterly_bonus = (otb / 4) * overall_percentage
    
    return {
        'quarterly_bonus': quarterly_bonus,
        'overall_percentage': overall_percentage * 100,
        'raw_multipliers': {  # Store raw values for capping detection
            'indiv_rev': rev_raw,
            'tcv': tcv_raw,
            'signed_mm': signed_mm_raw,
            'sl_rev': sl_rev_raw,
            'margin': margin_raw
        },
        'final_multipliers': {
            'indiv_rev': rev_final,
            'tcv': tcv_final,
            'signed_mm': signed_mm_final,
            'sl_rev': sl_rev_final,
            'margin': margin_final
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
    
    # Initialize session state
    if 'annual_targets' not in st.session_state:
        st.session_state.annual_targets = {
            'indiv_rev': {'annual': 400.0, 'q1': 100.0, 'q2': 100.0, 'q3': 100.0, 'q4': 100.0},
            'tcv': {'annual': 400.0, 'q1': 100.0, 'q2': 100.0, 'q3': 100.0, 'q4': 100.0},
            'signed_mm': {'annual': 400.0, 'q1': 100.0, 'q2': 100.0, 'q3': 100.0, 'q4': 100.0},
            'sl_rev': {'annual': 400.0, 'q1': 100.0, 'q2': 100.0, 'q3': 100.0, 'q4': 100.0},
            'margin': {'annual': 400.0, 'q1': 100.0, 'q2': 100.0, 'q3': 100.0, 'q4': 100.0}
        }
    
    if 'quarterly_data' not in st.session_state:
        st.session_state.quarterly_data = {
            1: {'indiv_rev_target': 100.0, 'indiv_rev_actual': 95.0, 
                'tcv_target': 100.0, 'tcv_actual': 105.0,
                'signed_mm_target': 100.0, 'signed_mm_actual': 110.0,
                'sl_rev_target': 100.0, 'sl_rev_actual': 120.0,
                'margin_target': 100.0, 'margin_actual': 90.0},
            2: {'indiv_rev_target': 100.0, 'indiv_rev_actual': 0.0, 
                'tcv_target': 100.0, 'tcv_actual': 0.0,
                'signed_mm_target': 100.0, 'signed_mm_actual': 0.0,
                'sl_rev_target': 100.0, 'sl_rev_actual': 0.0,
                'margin_target': 100.0, 'margin_actual': 0.0},
            3: {'indiv_rev_target': 100.0, 'indiv_rev_actual': 0.0, 
                'tcv_target': 100.0, 'tcv_actual': 0.0,
                'signed_mm_target': 100.0, 'signed_mm_actual': 0.0,
                'sl_rev_target': 100.0, 'sl_rev_actual': 0.0,
                'margin_target': 100.0, 'margin_actual': 0.0},
            4: {'indiv_rev_target': 100.0, 'indiv_rev_actual': 0.0, 
                'tcv_target': 100.0, 'tcv_actual': 0.0,
                'signed_mm_target': 100.0, 'signed_mm_actual': 0.0,
                'sl_rev_target': 100.0, 'sl_rev_actual': 0.0,
                'margin_target': 100.0, 'margin_actual': 0.0}
        }
    
    # Input section
    with st.sidebar:
        st.header("Global Parameters")
        otb = st.number_input("Annual On Target Bonus (OTB)", min_value=0, value=10000, step=1000)
    
    # Tabs structure
    tab1, tab2, tab3 = st.tabs([
        "Annual Target Planning", 
        "Quarterly Calculator", 
        "Full Year Projection"
    ])
    
    # Tab 1: Annual Target Planning
    with tab1:
        st.header("📊 Annual Target Planning")
        st.info("Set annual targets and quarterly breakdowns for all metrics")
        
        col1, col2 = st.columns(2)
        
        # Metric names mapping
        metric_names = {
            'indiv_rev': "Individual Reported Revenue",
            'tcv': "New Deal Closure TCV",
            'signed_mm': "Signed MM for New Deals",
            'sl_rev': "Service Line Revenue",
            'margin': "Managed Margin"
        }
        
        with col1:
            st.subheader("Annual Targets")
            for metric in st.session_state.annual_targets.keys():
                annual_val = st.number_input(
                    f"{metric_names[metric]} - Annual Target",
                    min_value=0.0,
                    value=st.session_state.annual_targets[metric]['annual'],
                    key=f"annual_{metric}"
                )
                st.session_state.annual_targets[metric]['annual'] = annual_val
        
        with col2:
            st.subheader("Quarterly Distribution")
            for metric in st.session_state.annual_targets.keys():
                st.markdown(f"**{metric_names[metric]}**")
                cols = st.columns(4)
                for i, qtr in enumerate(['q1', 'q2', 'q3', 'q4']):
                    with cols[i]:
                        qtr_val = st.number_input(
                            f"Q{i+1}",
                            min_value=0.0,
                            value=st.session_state.annual_targets[metric][qtr],
                            key=f"{metric}_{qtr}"
                        )
                        st.session_state.annual_targets[metric][qtr] = qtr_val
                
                # Show allocation percentage
                total_alloc = sum([
                    st.session_state.annual_targets[metric]['q1'],
                    st.session_state.annual_targets[metric]['q2'],
                    st.session_state.annual_targets[metric]['q3'],
                    st.session_state.annual_targets[metric]['q4']
                ])
                
                alloc_pct = (total_alloc / st.session_state.annual_targets[metric]['annual'] * 100 
                             if st.session_state.annual_targets[metric]['annual'] > 0 else 0)
                
                st.progress(min(alloc_pct/100, 1.0))
                st.caption(f"Allocated: {total_alloc:,.0f} of {st.session_state.annual_targets[metric]['annual']:,.0f} ({alloc_pct:.1f}%)")
        
        # Save annual targets to session state
        st.success("Annual targets saved automatically")
        
        # Update quarterly targets
        for q in range(1, 5):
            q_key = f'q{q}'
            st.session_state.quarterly_data[q]['indiv_rev_target'] = st.session_state.annual_targets['indiv_rev'][q_key]
            st.session_state.quarterly_data[q]['tcv_target'] = st.session_state.annual_targets['tcv'][q_key]
            st.session_state.quarterly_data[q]['signed_mm_target'] = st.session_state.annual_targets['signed_mm'][q_key]
            st.session_state.quarterly_data[q]['sl_rev_target'] = st.session_state.annual_targets['sl_rev'][q_key]
            st.session_state.quarterly_data[q]['margin_target'] = st.session_state.annual_targets['margin'][q_key]
    
    # Tab 2: Quarterly Calculator
    with tab2:
        st.header("📈 Quarterly Calculator")
        quarter = st.selectbox("Select Quarter", [1, 2, 3, 4], key="qtr_select")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Q{quarter} Metrics")
            
            # Individual metrics
            st.markdown("**Individual Metrics (70%)**")
            indiv_rev_target = st.number_input(
                "Reported Revenue - Target", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['indiv_rev_target'],
                key=f"indiv_rev_t_q{quarter}"
            )
            indiv_rev_actual = st.number_input(
                "Reported Revenue - Actual", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['indiv_rev_actual'],
                key=f"indiv_rev_a_q{quarter}"
            )
            
            tcv_target = st.number_input(
                "New Deal Closure TCV - Target", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['tcv_target'],
                key=f"tcv_t_q{quarter}"
            )
            tcv_actual = st.number_input(
                "New Deal Closure TCV - Actual", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['tcv_actual'],
                key=f"tcv_a_q{quarter}"
            )
            
            signed_mm_target = st.number_input(
                "Signed MM for New Deals - Target", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['signed_mm_target'],
                key=f"mm_t_q{quarter}"
            )
            signed_mm_actual = st.number_input(
                "Signed MM for New Deals - Actual", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['signed_mm_actual'],
                key=f"mm_a_q{quarter}"
            )
            
            # Service line metrics
            st.markdown("**Service Line Metrics (30%)**")
            sl_rev_target = st.number_input(
                "Service Line Revenue - Target", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['sl_rev_target'],
                key=f"sl_rev_t_q{quarter}"
            )
            sl_rev_actual = st.number_input(
                "Service Line Revenue - Actual", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['sl_rev_actual'],
                key=f"sl_rev_a_q{quarter}"
            )
            
            margin_target = st.number_input(
                "Managed Margin - Target", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['margin_target'],
                key=f"margin_t_q{quarter}"
            )
            margin_actual = st.number_input(
                "Managed Margin - Actual", 
                min_value=0.0, 
                value=st.session_state.quarterly_data[quarter]['margin_actual'],
                key=f"margin_a_q{quarter}"
            )
            
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
                metrics = st.session_state.quarterly_data[quarter]
                result = calculate_quarterly_bonus(otb, quarter, metrics)
                
                st.subheader(f"Q{quarter} Bonus Results")
                
                # Display multipliers and achievements
                st.markdown("**Metric Performance**")
                metric_data = []
                for metric in ['indiv_rev', 'tcv', 'signed_mm', 'sl_rev', 'margin']:
                    # Check if capped: Only for Q1-Q3 and when raw multiplier > 1.0
                    capped_note = "Capped" if (quarter < 4 and 
                                              result['raw_multipliers'][metric] > 1.0 and 
                                              result['final_multipliers'][metric] == 1.0) else ""
                    
                    metric_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Achievement %': f"{result['achievements'][metric]:.1f}%",
                        'Multiplier': f"{result['final_multipliers'][metric]:.2f}",
                        'Note': capped_note
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
    
    # Tab 3: Full Year Projection
    with tab3:
        st.header("📤 Full Year Projection & Export")
        st.info("Enter data for all quarters to see the full year projection and reconciliation")
        
        # Check if all quarters have data
        all_data_entered = all(st.session_state.quarterly_data[q] for q in range(1, 5))
        
        if not all_data_entered:
            st.warning("Please enter data for all quarters in the Quarterly Calculator tab")
            st.stop()
        
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
        
        # Export section
        st.subheader("Export Results")
        
        if st.button("Generate Full Year Report"):
            # Create comprehensive data structure
            report_data = []
            
            # Add annual targets
            for metric, data in st.session_state.annual_targets.items():
                report_data.append({
                    'Type': 'Target',
                    'Period': 'Annual',
                    'Metric': metric_names[metric],
                    'Value': data['annual']
                })
                for q in range(1, 5):
                    report_data.append({
                        'Type': 'Target',
                        'Period': f'Q{q}',
                        'Metric': metric_names[metric],
                        'Value': data[f'q{q}']
                    })
            
            # Add actuals and calculations
            for q in range(1, 5):
                metrics = st.session_state.quarterly_data[q]
                result = quarterly_results[q]['true']
                
                # Add actuals
                for metric_key in ['indiv_rev', 'tcv', 'signed_mm', 'sl_rev', 'margin']:
                    report_data.append({
                        'Type': 'Actual',
                        'Period': f'Q{q}',
                        'Metric': metric_names[metric_key],
                        'Value': metrics[f'{metric_key}_actual']
                    })
                
                # Add calculations
                for metric_key in ['indiv_rev', 'tcv', 'signed_mm', 'sl_rev', 'margin']:
                    report_data.append({
                        'Type': 'Achievement %',
                        'Period': f'Q{q}',
                        'Metric': metric_names[metric_key],
                        'Value': result['achievements'][metric_key]
                    })
                    report_data.append({
                        'Type': 'Multiplier',
                        'Period': f'Q{q}',
                        'Metric': metric_names[metric_key],
                        'Value': result['final_multipliers'][metric_key]
                    })
                
                # Add bonus results
                report_data.append({
                    'Type': 'Bonus',
                    'Period': f'Q{q}',
                    'Metric': 'Quarterly Bonus',
                    'Value': quarterly_true[q]
                })
            
            # Add full year summary
            report_data.append({
                'Type': 'Summary',
                'Period': 'Annual',
                'Metric': 'Total Bonus Before Cap',
                'Value': total_true
            })
            report_data.append({
                'Type': 'Summary',
                'Period': 'Annual',
                'Metric': 'Capped Bonus (200% OTB)',
                'Value': capped_bonus
            })
            report_data.append({
                'Type': 'Summary',
                'Period': 'Annual',
                'Metric': 'Q4 Payout Amount',
                'Value': q4_payout
            })
            report_data.append({
                'Type': 'Summary',
                'Period': 'Annual',
                'Metric': 'Total Annual Bonus',
                'Value': total_paid_q1q3 + q4_payout
            })
            
            # Create DataFrame
            df_report = pd.DataFrame(report_data)
            
            # Pivot for better Excel format
            pivot_df = df_report.pivot_table(
                index=['Metric', 'Type'], 
                columns='Period', 
                values='Value',
                aggfunc='first'
            ).reset_index()
            
            # Export options
            st.success("Report generated successfully!")
            
            # Excel export
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                pivot_df.to_excel(writer, sheet_name='Bonus Report', index=False)
                
                # Add summary sheet
                summary_df = df_report[df_report['Type'] == 'Summary']
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add raw data sheet
                df_report.to_excel(writer, sheet_name='Raw Data', index=False)
            
            excel_buffer.seek(0)
            
            # CSV export
            csv = df_report.to_csv(index=False).encode('utf-8')
            
            # Download buttons
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_buffer,
                file_name="bonus_calculations.xlsx",
                mime="application/vnd.ms-excel"
            )
            
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name="bonus_calculations.csv",
                mime="text/csv"
            )
            
            # Show preview
            st.subheader("Report Preview")
            st.dataframe(pivot_df.head(20))

if __name__ == "__main__":
    main()