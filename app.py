import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import subprocess
import threading
import time
from datetime import datetime
import sys
import io
import random
import numpy as np
import socket
from collections import defaultdict

# Page config
st.set_page_config(
    page_title="STS Claims Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-approved { background-color: #28a745; }
    .status-denied { background-color: #dc3545; }
    .status-open { background-color: #ffc107; }
    .status-impasse { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Try to get password from secrets, fallback to default for local development
        try:
            correct_password = st.secrets["password"]
        except:
            correct_password = "STS2025Dashboard!"
            
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown('<h1 class="main-header">🔐 STS Claims Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.info("**Demo Password:** STS2025Dashboard!")
        st.write("*Contact your administrator for access credentials*")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown('<h1 class="main-header">🔐 STS Claims Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True

def main():
    # Authentication check
    if not check_password():
        return
        
    st.markdown('<h1 class="main-header">STS Claims Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize demo data if no data exists
    if 'claims_data' not in st.session_state:
        load_demo_data()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Sharing information
        st.markdown("---")
        st.subheader("🌐 Share Dashboard")
        
        try:
            # Get local IP for sharing
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            share_url = f"http://{local_ip}:8501"
            st.success("**Share this URL:**")
            st.code(share_url, language="text")
            st.info("💡 Works for team members on the same network")
            
            if st.button("📋 Copy URL"):
                st.write("URL copied! (Manually copy the URL above)")
                
        except:
            st.warning("Run with network access to get shareable URL")
            
        st.markdown("**For public access:** See DEPLOYMENT_GUIDE.md")
        
        st.markdown("---")
        
        # Demo mode toggle
        demo_mode = st.toggle("Demo Mode", value=True, help="Use sample data for testing")
        
        if demo_mode:
            if st.button("🔄 Refresh Demo Data"):
                load_demo_data()
            
            # Export section for demo mode
            if 'claims_data' in st.session_state and len(st.session_state.claims_data) > 0:
                st.markdown("---")
                st.subheader("📥 Export Data")
                export_data()
        else:
            # Configuration section
            st.subheader("Configuration")
            relief_rate = st.number_input("Relief Rate ($/hour)", value=320.47, step=0.01)
            export_path = st.text_input("Export Path", value="", placeholder="C:\\path\\to\\your\\export\\folder")
            
            # Authentication section
            st.subheader("Authentication")
            username = st.text_input("Username", value="", placeholder="Enter your username")
            password = st.text_input("Password", type="password", value="", placeholder="Enter your password")
            
            # Run options
            st.subheader("Run Options")
            headless_mode = st.checkbox("Run in headless mode", value=False)
            max_pages = st.number_input("Max pages to scrape (0 = all)", value=0, min_value=0)
            
            # Action buttons
            if st.button("🚀 Start Data Collection", type="primary"):
                run_data_collection(relief_rate, export_path, username, password, headless_mode, max_pages)
            
            if st.button("📊 Load Latest Data"):
                load_latest_data(export_path)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📊 Analytics", "💰 Financial", "📋 Claims Details", "🔄 Real-time Status"])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_analytics_tab()
    
    with tab3:
        show_financial_tab()
    
    with tab4:
        show_claims_details_tab()
    
    with tab5:
        show_realtime_status_tab()

def relief_dollars(minutes, relief_rate=320.47):
    """Calculate relief dollars from minutes"""
    return (minutes / 60) * relief_rate if minutes else 0

def group_subject_key(violation):
    """Group violations into categories (from original script)"""
    if pd.isna(violation) or not violation:
        return "Unknown"
        
    v = str(violation).lower()
    if 'rest' in v:
        return 'Rest'
    if any(x in v for x in ['12.t', 'yellow slip', '12t', 'ys']):
        return 'Yellow Slip / 12.T'
    if any(x in v for x in ['11.f', '11f']):
        return '11.F'
    if any(x in v for x in ['payback day', '23.s.11', '23s11']):
        return 'Payback Day / 23.S.11'
    if any(x in v for x in ['23.q', '23q', 'green slip', 'gs']):
        return 'Green Slip / 23.Q'
    if 'green slip' in v:
        return 'Green Slip / 23.Q'
    if 'sc' in v or 'short call' in v:
        return 'Short Call'
    if 'long call' in v or 'lc' in v:
        return 'Long Call'
    if '23.o' in v or '23o' in v:
        return '23.O'
    if 'rotation coverage sequence' in v:
        return 'Rotation Coverage Sequence'
    if 'inverse assignment' in v or '23.r' in v:
        return 'Inverse Assignment / 23.R'
    if any(x in v for x in ['deadhead', '8.d', '8.d.3', '8d3']):
        return 'Deadhead / 8.D'
    if 'swap with the pot' in v:
        return 'Swap With The Pot'
    if '23.j' in v:
        return '23.J'
    if '4.f' in v:
        return '4.F'
    if 'arcos' in v or '23.z' in v:
        return 'ARCOS / 23.Z'
    if any(x in v for x in ['white slip', '23.p']):
        return 'White Slip / 23.P'
    if any(x in v for x in ['reroute', '23.l', '23l']):
        return 'Reroute / 23.L'
    if 'illegal rotation' in v:
        return 'Illegal Rotation'
    if '23.k' in v:
        return '23.K'
    if 'mou 24-01' in v:
        return 'MOU 24-01'
    return str(violation).strip() or "Unknown"

def status_canonical(status):
    """Normalize status values (from original script)"""
    if pd.isna(status) or not status:
        return "unknown"
        
    s = str(status).strip().lower()
    if s == "submitted to company":
        return "open"
    if s == "impasse":
        return "impasse"
    if s == "closed without payment":
        return "denied"
    if s == "paid":
        return "approved"
    if s in ("in review", "archived", "contested"):
        return s
    if s in ("open", "approved", "denied", "impasse"):
        return s
    return s

def calculate_comprehensive_analytics(df, relief_rate=320.47):
    """Calculate all analytics from original script"""
    # Add grouped subject and canonical status
    df = df.copy()
    df['Subject_Grouped'] = df['Subject Violations'].apply(group_subject_key)
    df['Status_Canonical'] = df['Status'].apply(status_canonical)
    df['Relief_Dollars'] = df['Relief Minutes'].apply(lambda x: relief_dollars(x, relief_rate))
    
    # Get all statuses
    all_statuses = sorted(df['Status_Canonical'].unique())
    
    # Subject grouped stats with dollars per status
    subject_stats = {}
    
    for subject in df['Subject_Grouped'].unique():
        subject_data = df[df['Subject_Grouped'] == subject]
        stats = {"count": len(subject_data)}
        
        for status in all_statuses:
            status_data = subject_data[subject_data['Status_Canonical'] == status]
            stats[f"{status}_count"] = len(status_data)
            stats[f"{status}_minutes"] = status_data['Relief Minutes'].sum()
            stats[f"{status}_dollars"] = status_data['Relief_Dollars'].sum()
            stats[f"{status}_pct"] = (len(status_data) / len(subject_data) * 100) if len(subject_data) > 0 else 0
            
        subject_stats[subject] = stats
    
    return subject_stats, all_statuses, df

def load_demo_data():
    """Load demo data for testing the dashboard"""
    import random
    import numpy as np
    
    # Generate sample claims data
    subjects = [
        'Rest', '11.F', 'Yellow Slip / 12.T', 'Green Slip / 23.Q', 
        'Short Call', 'Long Call', '23.O', 'Deadhead / 8.D',
        'Payback Day / 23.S.11', 'White Slip / 23.P', 'Reroute / 23.L'
    ]
    
    statuses = ['approved', 'denied', 'open', 'in review', 'impasse']
    
    # Generate 100 sample claims
    data = []
    for i in range(100):
        data.append({
            'Ticket #': f'12345{i:03d}',
            'Status': random.choice(statuses),
            'Relief Minutes': random.randint(30, 480),  # 30 min to 8 hours
            'Subject Violations': random.choice(subjects),
            'Emp #': f'N{100000 + i}',
            'Last Interaction': f'2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
            'Assignee': f'Assignee {random.randint(1,10)}',
            'Dispute #': f'D{1000 + i}',
            'Incident Date Rot #': f'2025-{random.randint(1,12):02d}-{random.randint(1,28):02d} ROT{i}'
        })
    
    st.session_state.claims_data = pd.DataFrame(data)
    st.session_state.demo_mode = True

def show_overview_tab():
    st.header("📈 STS Claims Overview")
    
    # Check if data exists
    if 'claims_data' not in st.session_state:
        st.info("No data loaded. Please run data collection or load existing data from the sidebar.")
        return
    
    df = st.session_state.claims_data
    relief_rate = 320.47
    
    # Show demo mode indicator
    if st.session_state.get('demo_mode', False):
        st.info("🎯 **Demo Mode Active** - Showing sample data for testing purposes")
    
    # Calculate comprehensive analytics
    subject_stats, all_statuses, enhanced_df = calculate_comprehensive_analytics(df, relief_rate)
    
    # === TOP-LEVEL METRICS ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_claims = len(df)
        st.metric("📋 Total Claims", f"{total_claims:,}")
    
    with col2:
        total_relief_hours = df['Relief Minutes'].sum() / 60
        st.metric("⏰ Total Relief Hours", f"{total_relief_hours:,.1f}")
    
    with col3:
        avg_relief_per_claim = df['Relief Minutes'].mean() / 60
        st.metric("📊 Avg Relief/Claim", f"{avg_relief_per_claim:.1f}h")
    
    with col4:
        total_value = enhanced_df['Relief_Dollars'].sum()
        st.metric("💰 Total Value", f"${total_value:,.0f}")
    
    with col5:
        unique_subjects = len(subject_stats)
        st.metric("🏷️ Violation Types", unique_subjects)
    
    # === STATUS BREAKDOWN ===
    st.subheader("📊 Status Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status pie chart with exact counts
        status_counts = enhanced_df['Status_Canonical'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, 
                    title="Claims by Status")
        # Add percentage and count labels
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Status metrics
        for status in all_statuses:
            count = len(enhanced_df[enhanced_df['Status_Canonical'] == status])
            pct = count / len(enhanced_df) * 100 if len(enhanced_df) > 0 else 0
            value = enhanced_df[enhanced_df['Status_Canonical'] == status]['Relief_Dollars'].sum()
            
            if count > 0:
                st.metric(
                    f"{status.title()} Cases", 
                    f"{count} ({pct:.1f}%)",
                    delta=f"${value:,.0f} total value"
                )
    
    with col2:
        # Top 10 subject violations
        subject_counts = enhanced_df['Subject_Grouped'].value_counts().head(10)
        fig = px.bar(x=subject_counts.values, y=subject_counts.index, 
                    orientation='h', title="Top 10 Subject Violations")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # === FINANCIAL OVERVIEW ===
    st.subheader("💰 Financial Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        approved_value = enhanced_df[enhanced_df['Status_Canonical'] == 'approved']['Relief_Dollars'].sum()
        approved_count = len(enhanced_df[enhanced_df['Status_Canonical'] == 'approved'])
        st.metric(
            "💚 Approved Value",
            f"${approved_value:,.0f}",
            delta=f"{approved_count} cases"
        )
    
    with col2:
        pending_value = enhanced_df[enhanced_df['Status_Canonical'].isin(['open', 'in review'])]['Relief_Dollars'].sum()
        pending_count = len(enhanced_df[enhanced_df['Status_Canonical'].isin(['open', 'in review'])])
        st.metric(
            "🟡 Pending Value",
            f"${pending_value:,.0f}",
            delta=f"{pending_count} cases"
        )
    
    with col3:
        denied_value = enhanced_df[enhanced_df['Status_Canonical'] == 'denied']['Relief_Dollars'].sum()
        denied_count = len(enhanced_df[enhanced_df['Status_Canonical'] == 'denied'])
        st.metric(
            "❌ Denied Value", 
            f"${denied_value:,.0f}",
            delta=f"{denied_count} cases"
        )
    
    with col4:
        # Calculate overall approval rate
        decided_cases = enhanced_df[enhanced_df['Status_Canonical'].isin(['approved', 'denied'])]
        if len(decided_cases) > 0:
            approval_rate = len(decided_cases[decided_cases['Status_Canonical'] == 'approved']) / len(decided_cases) * 100
        else:
            approval_rate = 0
        st.metric(
            "📈 Approval Rate",
            f"{approval_rate:.1f}%",
            delta=f"{len(decided_cases)} decided cases"
        )
    
    # === TOP PERFORMING SUBJECTS ===
    st.subheader("🏆 Top Performing Subject Violations")
    
    # Create performance summary
    performance_data = []
    for subject, stats in subject_stats.items():
        total_value = sum([stats.get(f"{status}_dollars", 0) for status in all_statuses])
        approved_count = stats.get("approved_count", 0)
        total_count = stats["count"]
        
        performance_data.append({
            'Subject': subject,
            'Total Cases': total_count,
            'Total Value': total_value,
            'Approved Cases': approved_count,
            'Value per Case': total_value / max(total_count, 1)
        })
    
    # Top by volume
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top by Case Volume:**")
        top_volume = sorted(performance_data, key=lambda x: x['Total Cases'], reverse=True)[:5]
        for item in top_volume:
            st.write(f"• **{item['Subject']}**: {item['Total Cases']} cases (${item['Total Value']:,.0f})")
    
    with col2:
        st.write("**Top by Total Value:**")
        top_value = sorted(performance_data, key=lambda x: x['Total Value'], reverse=True)[:5]
        for item in top_value:
            st.write(f"• **{item['Subject']}**: ${item['Total Value']:,.0f} ({item['Total Cases']} cases)")
    
    # === QUICK INSIGHTS ===
    st.subheader("💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **📈 Key Statistics:**
        • Total claims processed: **{total_claims:,}**
        • Average relief per claim: **{avg_relief_per_claim:.1f} hours**
        • Total financial exposure: **${total_value:,.0f}**
        • Most common violation: **{enhanced_df['Subject_Grouped'].mode().iloc[0] if len(enhanced_df) > 0 else 'N/A'}**
        """)
    
    with col2:
        highest_value_subject = max(performance_data, key=lambda x: x['Total Value']) if performance_data else None
        highest_approval_rate = 0
        highest_approval_subject = "N/A"
        
        for subject, stats in subject_stats.items():
            approved = stats.get("approved_count", 0)
            denied = stats.get("denied_count", 0)
            total_decided = approved + denied
            if total_decided >= 5:  # Only consider subjects with at least 5 decided cases
                rate = approved / total_decided
                if rate > highest_approval_rate:
                    highest_approval_rate = rate
                    highest_approval_subject = subject
        
        if highest_value_subject:
            st.success(f"""
            **🎯 Performance Highlights:**
            • Highest value subject: **{highest_value_subject['Subject']}**
            • Best approval rate: **{highest_approval_subject}** ({highest_approval_rate:.1%})
            • Pending claims value: **${pending_value:,.0f}**
            • Current approval rate: **{approval_rate:.1f}%**
            """)

def show_analytics_tab():
    st.header("📊 Comprehensive Analytics")
    
    if 'claims_data' not in st.session_state:
        st.info("No data loaded. Please run data collection first.")
        return
    
    df = st.session_state.claims_data
    relief_rate = 320.47
    
    # Calculate comprehensive analytics
    subject_stats, all_statuses, enhanced_df = calculate_comprehensive_analytics(df, relief_rate)
    
    # === SUBJECT BREAKDOWN BY STATUS ===
    st.subheader("📋 Complete Breakdown by Subject Violation")
    
    # Create detailed breakdown table
    breakdown_data = []
    for subject, stats in subject_stats.items():
        row = [subject, stats["count"]]
        for status in all_statuses:
            count = stats.get(f"{status}_count", 0)
            dollars = stats.get(f"{status}_dollars", 0)
            pct = stats.get(f"{status}_pct", 0)
            row.extend([count, f"${dollars:,.0f}", f"{pct:.1f}%"])
        breakdown_data.append(row)
    
    # Create column headers
    columns = ["Subject", "Total Cases"]
    for status in all_statuses:
        columns.extend([f"{status.title()} Count", f"{status.title()} Dollars", f"{status.title()} %"])
    
    breakdown_df = pd.DataFrame(breakdown_data, columns=columns)
    breakdown_df = breakdown_df.sort_values("Total Cases", ascending=False)
    
    st.dataframe(breakdown_df, use_container_width=True, height=400)
    
    # Export button for breakdown
    if st.button("📥 Export Complete Breakdown"):
        csv = breakdown_df.to_csv(index=False)
        st.download_button(
            label="Download Complete Analytics CSV",
            data=csv,
            file_name=f"sts_complete_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # === PROBABILITY OF PAYMENT ===
    st.subheader("🎯 Payment Probability Analysis")
    
    probability_data = []
    for subject, stats in subject_stats.items():
        approved = stats.get("approved_count", 0)
        denied = stats.get("denied_count", 0)
        total_decided = approved + denied
        
        if total_decided > 0:
            probability = approved / total_decided
            probability_data.append({
                'Subject': subject,
                'Approved': approved,
                'Denied': denied,
                'Total Decided': total_decided,
                'Probability': probability,
                'Probability %': f"{probability:.1%}"
            })
    
    if probability_data:
        prob_df = pd.DataFrame(probability_data).sort_values('Probability', ascending=False)
        
        # Probability chart
        fig = px.bar(prob_df, x='Subject', y='Probability', 
                    title="Payment Approval Probability by Subject",
                    labels={'Probability': 'Approval Rate'})
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability table
        st.dataframe(prob_df[['Subject', 'Approved', 'Denied', 'Total Decided', 'Probability %']], 
                    use_container_width=True)
    
    # === ESTIMATED APPROVALS (FORECASTING) ===
    st.subheader("🔮 Forecasting: Estimated Future Approvals")
    
    forecasting_data = []
    for subject, stats in subject_stats.items():
        approved = stats.get("approved_count", 0)
        denied = stats.get("denied_count", 0)
        total_decided = approved + denied
        probability = approved / total_decided if total_decided > 0 else 0
        
        open_cases = stats.get("open_count", 0)
        in_review_cases = stats.get("in review_count", 0)
        pending_cases = open_cases + in_review_cases
        
        estimated_approvals = probability * pending_cases
        estimated_dollars = estimated_approvals * (stats.get("approved_dollars", 0) / max(approved, 1))
        
        if pending_cases > 0:
            forecasting_data.append({
                'Subject': subject,
                'Open Cases': open_cases,
                'In Review': in_review_cases,
                'Total Pending': pending_cases,
                'Historical Approval Rate': f"{probability:.1%}",
                'Estimated Approvals': f"{estimated_approvals:.1f}",
                'Estimated Value': f"${estimated_dollars:,.0f}"
            })
    
    if forecasting_data:
        forecast_df = pd.DataFrame(forecasting_data).sort_values('Total Pending', ascending=False)
        
        # Forecasting chart
        fig = px.bar(forecast_df, x='Subject', y=[col for col in forecast_df.columns if 'Estimated Approvals' in col],
                    title="Estimated Future Approvals by Subject")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(forecast_df, use_container_width=True)
    
    # === FINANCIAL PROJECTIONS BY STATUS ===
    st.subheader("💰 Actual and Projected Dollars by Status")
    
    financial_data = []
    for subject, stats in subject_stats.items():
        row = {'Subject': subject}
        for status in all_statuses:
            dollars = stats.get(f"{status}_dollars", 0)
            row[f"{status.title()} $"] = f"${dollars:,.0f}"
        financial_data.append(row)
    
    financial_df = pd.DataFrame(financial_data)
    st.dataframe(financial_df, use_container_width=True)
    
    # === SUMMARY STATISTICS ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unique_subjects = len(subject_stats)
        st.metric("Unique Violation Types", unique_subjects)
    
    with col2:
        total_relief_dollars = enhanced_df['Relief_Dollars'].sum()
        st.metric("Total Relief Value", f"${total_relief_dollars:,.0f}")
    
    with col3:
        avg_probability = np.mean([stats.get("approved_count", 0) / max(stats.get("approved_count", 0) + stats.get("denied_count", 0), 1) 
                                  for stats in subject_stats.values()]) if subject_stats else 0
        st.metric("Average Approval Rate", f"{avg_probability:.1%}")
    
    with col4:
        total_pending = sum([stats.get("open_count", 0) + stats.get("in review_count", 0) 
                           for stats in subject_stats.values()])
        st.metric("Total Pending Cases", total_pending)

def show_financial_tab():
    st.header("💰 Comprehensive Financial Analysis")
    
    if 'claims_data' not in st.session_state:
        st.info("No data loaded. Please run data collection first.")
        return
    
    df = st.session_state.claims_data
    relief_rate = 320.47
    
    # Calculate comprehensive analytics
    subject_stats, all_statuses, enhanced_df = calculate_comprehensive_analytics(df, relief_rate)
    
    # === TOP-LEVEL FINANCIAL METRICS ===
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_approved = enhanced_df[enhanced_df['Status_Canonical'] == 'approved']['Relief_Dollars'].sum()
        st.metric("💚 Approved Value", f"${total_approved:,.0f}")
    
    with col2:
        total_pending = enhanced_df[enhanced_df['Status_Canonical'].isin(['open', 'in review'])]['Relief_Dollars'].sum()
        st.metric("🟡 Pending Value", f"${total_pending:,.0f}")
    
    with col3:
        total_denied = enhanced_df[enhanced_df['Status_Canonical'] == 'denied']['Relief_Dollars'].sum()
        st.metric("❌ Denied Value", f"${total_denied:,.0f}")
    
    with col4:
        total_value = enhanced_df['Relief_Dollars'].sum()
        st.metric("📊 Total Value", f"${total_value:,.0f}")
    
    # === FINANCIAL BREAKDOWN BY SUBJECT ===
    st.subheader("📋 Financial Breakdown by Subject Violation")
    
    # Create financial summary
    financial_summary = []
    for subject, stats in subject_stats.items():
        total_dollars = sum([stats.get(f"{status}_dollars", 0) for status in all_statuses])
        approved_dollars = stats.get("approved_dollars", 0)
        pending_dollars = stats.get("open_dollars", 0) + stats.get("in review_dollars", 0)
        denied_dollars = stats.get("denied_dollars", 0)
        
        financial_summary.append({
            'Subject': subject,
            'Total Cases': stats["count"],
            'Total Value': f"${total_dollars:,.0f}",
            'Approved Value': f"${approved_dollars:,.0f}",
            'Pending Value': f"${pending_dollars:,.0f}",
            'Denied Value': f"${denied_dollars:,.0f}",
            'Avg per Case': f"${total_dollars/max(stats['count'], 1):,.0f}"
        })
    
    financial_df = pd.DataFrame(financial_summary)
    financial_df = financial_df.sort_values('Total Cases', ascending=False)
    
    st.dataframe(financial_df, use_container_width=True, height=400)
    
    # === FINANCIAL CHARTS ===
    col1, col2 = st.columns(2)
    
    with col1:
        # Top subjects by total value
        top_subjects = financial_df.head(10)
        fig = px.bar(top_subjects, x='Subject', y='Total Value',
                    title="Top 10 Subjects by Total Financial Value")
        fig.update_layout(xaxis_tickangle=-45)
        # Remove $ and , for plotting
        top_subjects['Value_Numeric'] = top_subjects['Total Value'].str.replace('$', '').str.replace(',', '').astype(float)
        fig = px.bar(top_subjects, x='Subject', y='Value_Numeric',
                    title="Top 10 Subjects by Total Financial Value")
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Total Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Status distribution by dollars
        status_totals = []
        for status in all_statuses:
            total = enhanced_df[enhanced_df['Status_Canonical'] == status]['Relief_Dollars'].sum()
            if total > 0:
                status_totals.append({'Status': status.title(), 'Total': total})
        
        if status_totals:
            status_df = pd.DataFrame(status_totals)
            fig = px.pie(status_df, values='Total', names='Status',
                        title="Financial Value Distribution by Status")
            st.plotly_chart(fig, use_container_width=True)
    
    # === DETAILED FINANCIAL TABLE BY STATUS ===
    st.subheader("💰 Detailed Dollars by Subject and Status")
    
    detailed_financial = []
    for subject, stats in subject_stats.items():
        row = {'Subject': subject}
        for status in all_statuses:
            dollars = stats.get(f"{status}_dollars", 0)
            row[f"{status.title()}"] = f"${dollars:,.0f}"
        detailed_financial.append(row)
    
    detailed_df = pd.DataFrame(detailed_financial)
    st.dataframe(detailed_df, use_container_width=True)
    
    # === PROJECTED FINANCIAL IMPACT ===
    st.subheader("🔮 Projected Financial Impact")
    
    projections = []
    total_projected_approvals = 0
    total_projected_value = 0
    
    for subject, stats in subject_stats.items():
        approved = stats.get("approved_count", 0)
        denied = stats.get("denied_count", 0)
        total_decided = approved + denied
        probability = approved / total_decided if total_decided > 0 else 0
        
        open_cases = stats.get("open_count", 0)
        in_review_cases = stats.get("in review_count", 0)
        pending_cases = open_cases + in_review_cases
        
        if pending_cases > 0 and approved > 0:
            avg_approved_value = stats.get("approved_dollars", 0) / approved
            estimated_approvals = probability * pending_cases
            estimated_value = estimated_approvals * avg_approved_value
            
            total_projected_approvals += estimated_approvals
            total_projected_value += estimated_value
            
            projections.append({
                'Subject': subject,
                'Pending Cases': pending_cases,
                'Historical Approval Rate': f"{probability:.1%}",
                'Estimated Approvals': f"{estimated_approvals:.1f}",
                'Avg Approved Value': f"${avg_approved_value:,.0f}",
                'Projected Value': f"${estimated_value:,.0f}"
            })
    
    if projections:
        projection_df = pd.DataFrame(projections)
        st.dataframe(projection_df, use_container_width=True)
        
        # Summary metrics for projections
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Projected Approvals", f"{total_projected_approvals:.0f}")
        with col2:
            st.metric("💰 Projected Value", f"${total_projected_value:,.0f}")
        with col3:
            current_approval_rate = len(enhanced_df[enhanced_df['Status_Canonical'] == 'approved']) / len(enhanced_df) * 100
            st.metric("📈 Overall Approval Rate", f"{current_approval_rate:.1f}%")
    
    # === EXPORT OPTIONS ===
    st.subheader("📥 Export Financial Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Financial Summary"):
            csv = financial_df.to_csv(index=False)
            st.download_button(
                label="Download Financial Summary CSV",
                data=csv,
                file_name=f"financial_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Detailed Breakdown"):
            csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Breakdown CSV",
                data=csv,
                file_name=f"detailed_financial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if projections and st.button("Export Projections"):
            csv = projection_df.to_csv(index=False)
            st.download_button(
                label="Download Projections CSV",
                data=csv,
                file_name=f"financial_projections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_claims_details_tab():
    st.header("📋 Claims Details")
    
    if 'claims_data' not in st.session_state:
        st.info("No data loaded. Please run data collection first.")
        return
    
    df = st.session_state.claims_data
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect("Filter by Status", 
                                     options=df['Status'].unique(),
                                     default=df['Status'].unique())
    
    with col2:
        subject_filter = st.multiselect("Filter by Subject", 
                                      options=df['Subject Violations'].unique())
    
    with col3:
        min_relief = st.number_input("Min Relief Minutes", value=0, min_value=0)
    
    # Apply filters
    filtered_df = df[df['Status'].isin(status_filter)]
    if subject_filter:
        filtered_df = filtered_df[filtered_df['Subject Violations'].isin(subject_filter)]
    filtered_df = filtered_df[filtered_df['Relief Minutes'] >= min_relief]
    
    # Display filtered data
    st.write(f"Showing {len(filtered_df)} of {len(df)} claims")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Export options
    if st.button("📥 Export Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"filtered_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_realtime_status_tab():
    st.header("🔄 Real-time Status")
    
    # Show current scraping status
    if 'scraping_status' in st.session_state:
        status = st.session_state.scraping_status
        
        if status['running']:
            st.warning("🔄 Data collection in progress...")
            
            # Progress indicators
            if 'progress' in status:
                st.progress(status['progress'])
                st.write(f"Processed: {status.get('processed', 0)} / {status.get('total', 0)}")
            
            # Live log output
            if 'log_output' in status:
                st.subheader("Live Log Output")
                st.text_area("Log", status['log_output'], height=300)
                
            # Auto-refresh every 5 seconds
            time.sleep(5)
            st.rerun()
        else:
            st.success("✅ Data collection completed!")
            if 'completion_time' in status:
                st.write(f"Completed at: {status['completion_time']}")
    else:
        st.info("No active data collection process.")

def run_data_collection(relief_rate, export_path, username, password, headless_mode, max_pages):
    """Run the data collection script with UI parameters"""
    
    st.session_state.scraping_status = {
        'running': True,
        'start_time': datetime.now(),
        'progress': 0,
        'processed': 0,
        'total': 0,
        'log_output': ""
    }
    
    # Create export directory if it doesn't exist
    os.makedirs(export_path, exist_ok=True)
    
    # Create a modified version of the original script with the UI parameters
    script_content = create_modified_script(relief_rate, export_path, username, password, headless_mode, max_pages)
    
    # Save the script temporarily in the current directory
    temp_script_path = os.path.join(os.path.dirname(__file__), "temp_sts_script.py")
    with open(temp_script_path, 'w') as f:
        f.write(script_content)
    
    # Run in a separate thread
    def run_script():
        try:
            # Get the Python executable path
            import sys
            python_exe = sys.executable
            
            process = subprocess.Popen([python_exe, temp_script_path], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True,
                                     cwd=os.path.dirname(__file__))
            
            output, error = process.communicate()
            
            st.session_state.scraping_status.update({
                'running': False,
                'completion_time': datetime.now(),
                'output': output,
                'error': error,
                'success': process.returncode == 0
            })
            
            # Clean up temp script
            try:
                os.remove(temp_script_path)
            except:
                pass
            
            # Load the data automatically after completion
            if process.returncode == 0:
                load_latest_data(export_path)
                
        except Exception as e:
            st.session_state.scraping_status.update({
                'running': False,
                'error': str(e),
                'success': False
            })
    
    # Start the script in a thread
    threading.Thread(target=run_script, daemon=True).start()
    st.success("🚀 Data collection started! Check the Real-time Status tab for progress.")

def create_modified_script(relief_rate, export_path, username, password, headless_mode, max_pages):
    """Create a modified version of the original script with UI parameters"""
    return f'''
import sys
import os
sys.path.append(r"{os.path.dirname(os.path.abspath(__file__))}")

from sts_processor import STSClaimsProcessor

if __name__ == "__main__":
    processor = STSClaimsProcessor(
        relief_rate={relief_rate},
        export_path=r"{export_path}",
        headless={headless_mode}
    )
    result = processor.run_full_process("{username}", "{password}", max_pages={max_pages})
    print("Process completed:", result)
'''

def load_latest_data(export_path):
    """Load the most recent data file"""
    try:
        # Create export directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)
        
        # Find the most recent analytics file
        files = [f for f in os.listdir(export_path) if f.startswith('sts_claims_analytics_') and f.endswith('.csv')]
        if not files:
            st.warning("No data files found. Please run data collection first.")
            return
        
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(export_path, x)))
        file_path = os.path.join(export_path, latest_file)
        
        # Load the data - this is the analytics summary, not raw claims data
        df = pd.read_csv(file_path)
        st.session_state.analytics_data = df
        
        # Try to also load raw claims data if available
        raw_files = [f for f in os.listdir(export_path) if 'claims_raw' in f and f.endswith('.csv')]
        if raw_files:
            latest_raw = max(raw_files, key=lambda x: os.path.getctime(os.path.join(export_path, x)))
            raw_df = pd.read_csv(os.path.join(export_path, latest_raw))
            st.session_state.claims_data = raw_df
        else:
            # Create sample data for demo purposes
            sample_data = {
                'Ticket #': [f'12345{i}' for i in range(10)],
                'Status': ['approved', 'denied', 'open', 'in review'] * 3 + ['approved', 'denied'],
                'Relief Minutes': [120, 90, 180, 60, 240, 150, 75, 300, 45, 200],
                'Subject Violations': ['Rest', '11.F', 'Yellow Slip / 12.T', 'Green Slip / 23.Q'] * 3 + ['Rest', 'Short Call'],
                'Emp #': [f'N{100000 + i}' for i in range(10)],
                'Last Interaction': ['2025-01-15', '2025-01-14', '2025-01-13'] * 4,
            }
            st.session_state.claims_data = pd.DataFrame(sample_data)
        
        st.success(f"✅ Loaded data from {latest_file}")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Create sample data for demo
        sample_data = {
            'Ticket #': ['123456', '123457', '123458'],
            'Status': ['approved', 'denied', 'open'],
            'Relief Minutes': [120, 90, 180],
            'Subject Violations': ['Rest', '11.F', 'Yellow Slip / 12.T'],
            'Emp #': ['N100000', 'N100001', 'N100002'],
            'Last Interaction': ['2025-01-15', '2025-01-14', '2025-01-13'],
        }
        st.session_state.claims_data = pd.DataFrame(sample_data)
        st.info("Demo data loaded for testing purposes.")

# === UTILITY FUNCTIONS ===

@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def export_data():
    """Export current data in various formats"""
    if 'claims_data' not in st.session_state:
        st.warning("No data to export")
        return
    
    df = st.session_state.claims_data
    
    # Add enhanced columns for export
    relief_rate = 320.47
    enhanced_df = df.copy()
    enhanced_df['Relief_Dollars'] = enhanced_df['Relief Minutes'].apply(relief_dollars)
    enhanced_df['Subject_Grouped'] = enhanced_df['Subject Violations'].apply(group_subject_key)
    enhanced_df['Status_Canonical'] = enhanced_df['Status'].apply(status_canonical)
    
    # CSV export
    csv = convert_df_to_csv(enhanced_df)
    st.download_button(
        label="📥 Download Enhanced Data as CSV",
        data=csv,
        file_name=f'sts_claims_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )
    
    # Summary report
    subject_stats, all_statuses, _ = calculate_comprehensive_analytics(df, 320.47)
    
    summary_data = []
    for subject, stats in subject_stats.items():
        summary_data.append({
            'Subject': subject,
            'Total_Cases': stats['count'],
            'Total_Hours': stats['total_hours'],
            'Total_Dollars': stats['total_dollars'],
            'Approved_Count': stats.get('approved_count', 0),
            'Denied_Count': stats.get('denied_count', 0),
            'Pending_Count': stats.get('open_count', 0) + stats.get('in review_count', 0),
            'Approval_Rate': stats.get('approved_count', 0) / max(stats.get('approved_count', 0) + stats.get('denied_count', 0), 1)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = convert_df_to_csv(summary_df)
    st.download_button(
        label="📊 Download Analytics Summary as CSV",
        data=summary_csv,
        file_name=f'sts_analytics_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
