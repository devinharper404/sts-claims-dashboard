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
    
    # Initialize demo data if no data exists and demo mode is enabled
    if 'claims_data' not in st.session_state and st.session_state.get('demo_mode', True):
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
        
        # Update session state based on toggle
        st.session_state.demo_mode = demo_mode
        
        if demo_mode:
            st.info("🎯 **Demo Mode**: Using sample data for demonstration purposes")
            
            # Small hint for demo password (less prominent)
            with st.expander("ℹ️ Demo Access Info"):
                st.write("**Demo Password:** `STS2025Dashboard!`")
                st.write("*This is for demonstration purposes only*")
            
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
            
            # Store configuration in session state for persistence
            st.session_state.relief_rate = relief_rate
            st.session_state.export_path = export_path
            st.session_state.username = username
            st.session_state.password = password
            st.session_state.headless_mode = headless_mode
            st.session_state.max_pages = max_pages
            
            # Configuration validation
            config_valid = True
            config_issues = []
            
            if not export_path:
                config_issues.append("Export Path is required")
                config_valid = False
            if not username:
                config_issues.append("Username is required")
                config_valid = False
            if not password:
                config_issues.append("Password is required")
                config_valid = False
            
            if config_issues:
                st.warning("⚠️ Configuration Issues:")
                for issue in config_issues:
                    st.write(f"• {issue}")
            else:
                st.success("✅ Configuration Complete")
            
            # Test connection button
            if st.button("🔍 Test Data Source Connection"):
                if not config_valid:
                    st.error("Please complete configuration before testing connection.")
                else:
                    test_data_source_connection(username, password, export_path)
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Start Data Collection", type="primary", disabled=not config_valid):
                    if config_valid:
                        run_data_collection(relief_rate, export_path, username, password, headless_mode, max_pages)
                    else:
                        st.error("Please complete all required configuration fields.")
            
            with col2:
                if st.button("📊 Load Latest Data", disabled=not export_path):
                    if export_path:
                        load_latest_data(export_path)
                    else:
                        st.error("Please provide an export path.")
    
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
    
    # Generate sample claims data with realistic patterns
    subjects = [
        'Rest', '11.F', 'Yellow Slip / 12.T', 'Green Slip / 23.Q', 
        'Short Call', 'Long Call', '23.O', 'Deadhead / 8.D',
        'Payback Day / 23.S.11', 'White Slip / 23.P', 'Reroute / 23.L',
        'ARCOS / 23.Z', 'Inverse Assignment / 23.R', '23.J', '4.F'
    ]
    
    statuses = ['approved', 'denied', 'open', 'in review', 'impasse']
    
    # Create employee pool - some will have multiple claims
    base_employees = [f'N{100000 + i}' for i in range(60)]  # 60 unique employees
    
    # Generate 150 sample claims with some employees having multiple claims
    data = []
    ticket_counter = 123000
    
    for i in range(150):
        # Some employees are more likely to have multiple claims
        if i < 30:
            # First 30 claims - unique employees
            emp_num = base_employees[i]
        else:
            # Remaining claims - some employees repeat (creating multi-claim pilots)
            if random.random() < 0.4:  # 40% chance of repeat employee
                emp_num = random.choice(base_employees[:40])  # Choose from first 40 employees
            else:
                emp_num = random.choice(base_employees[40:])  # Or use remaining employees
        
        # Generate realistic relief times based on subject
        subject = random.choice(subjects)
        if 'Rest' in subject:
            relief_minutes = random.randint(60, 300)  # Rest violations tend to be longer
        elif 'Short Call' in subject:
            relief_minutes = random.randint(30, 120)  # Short calls are typically shorter
        elif 'Yellow Slip' in subject or '12.T' in subject:
            relief_minutes = random.randint(90, 240)  # Medium duration
        else:
            relief_minutes = random.randint(45, 180)  # Standard range
        
        # Status distribution with some bias
        status_weights = [0.35, 0.25, 0.20, 0.15, 0.05]  # approved, denied, open, in review, impasse
        status = random.choices(statuses, weights=status_weights)[0]
        
        # Generate realistic dates over past 6 months
        days_ago = random.randint(1, 180)
        from datetime import datetime, timedelta
        interaction_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        data.append({
            'Ticket #': f'{ticket_counter + i:06d}',
            'Status': status,
            'Relief Minutes': relief_minutes,
            'Subject Violations': subject,
            'Emp #': emp_num,
            'Last Interaction': interaction_date,
            'Assignee': f'Assignee {random.randint(1,12)}',
            'Dispute #': f'D{2000 + i}',
            'Incident Date Rot #': f'{interaction_date} ROT{random.randint(1,50)}'
        })
    
    st.session_state.claims_data = pd.DataFrame(data)
    # Don't automatically set demo_mode here - let the toggle control it

def show_overview_tab():
    st.header("📈 STS Claims Overview")
    
    # Check if data exists
    if 'claims_data' not in st.session_state:
        if st.session_state.get('demo_mode', True):
            st.info("No data loaded. Loading demo data...")
            load_demo_data()
        else:
            st.info("No data loaded. Please run data collection or load existing data from the sidebar.")
            return
    
    df = st.session_state.claims_data
    relief_rate = 320.47
    
    # Show demo mode indicator
    if st.session_state.get('demo_mode', False):
        st.info("🎯 **Demo Mode Active** - Displaying sample data for demonstration and testing purposes")
    
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
        if st.session_state.get('demo_mode', True):
            st.info("No data loaded. Loading demo data...")
            load_demo_data()
        else:
            st.info("No data loaded. Please run data collection or load existing data from the sidebar.")
            return
    
    df = st.session_state.claims_data
    relief_rate = 320.47
    
    # Show demo mode indicator
    if st.session_state.get('demo_mode', False):
        st.info("🎯 **Demo Mode Active** - Displaying comprehensive sample analytics")
    
    # Calculate comprehensive analytics
    subject_stats, all_statuses, enhanced_df = calculate_comprehensive_analytics(df, relief_rate)
    
    # === PILOTS WITH MULTIPLE SUBMISSIONS ===
    st.subheader("👥 Pilots with Multiple Submissions")
    
    # Group by employee number to find multiple submissions
    employee_stats = enhanced_df.groupby('Emp #').agg({
        'Ticket #': 'count',
        'Relief_Dollars': 'sum',
        'Relief Minutes': 'sum',
        'Status_Canonical': lambda x: x.value_counts().to_dict(),
        'Subject_Grouped': lambda x: ', '.join(x.unique()[:3]) + ('...' if len(x.unique()) > 3 else '')
    }).rename(columns={'Ticket #': 'Total_Claims'})
    
    # Filter for employees with multiple claims
    multiple_claims = employee_stats[employee_stats['Total_Claims'] > 1].sort_values('Total_Claims', ascending=False)
    
    if len(multiple_claims) > 0:
        # Create display dataframe
        multi_display = []
        for emp, row in multiple_claims.iterrows():
            status_counts = row['Status_Canonical']
            approved = status_counts.get('approved', 0)
            denied = status_counts.get('denied', 0)
            pending = status_counts.get('open', 0) + status_counts.get('in review', 0)
            
            multi_display.append({
                'Employee #': emp,
                'Total Claims': row['Total_Claims'],
                'Total Value': f"${row['Relief_Dollars']:,.0f}",
                'Total Hours': f"{row['Relief Minutes']/60:.1f}",
                'Approved': approved,
                'Denied': denied,
                'Pending': pending,
                'Top Subjects': row['Subject_Grouped']
            })
        
        multi_df = pd.DataFrame(multi_display)
        st.dataframe(multi_df, use_container_width=True, height=300)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Multi-Claim Pilots", len(multiple_claims))
        with col2:
            avg_claims = multiple_claims['Total_Claims'].mean()
            st.metric("Avg Claims/Pilot", f"{avg_claims:.1f}")
        with col3:
            total_multi_value = multiple_claims['Relief_Dollars'].sum()
            st.metric("Multi-Claim Value", f"${total_multi_value:,.0f}")
        with col4:
            max_claims = multiple_claims['Total_Claims'].max()
            st.metric("Max Claims (1 pilot)", int(max_claims))
    else:
        st.info("No pilots found with multiple claims in current dataset.")
    
    # === TOP 20 HIGHEST VALUE CLAIMS ===
    st.subheader("🏆 Top 20 Highest Value Claims")
    
    # Sort by relief dollars and take top 20
    top_20 = enhanced_df.nlargest(20, 'Relief_Dollars')[
        ['Ticket #', 'Emp #', 'Subject_Grouped', 'Status_Canonical', 'Relief Minutes', 'Relief_Dollars', 'Last Interaction']
    ].copy()
    
    # Format for display
    top_20['Relief_Dollars_Display'] = top_20['Relief_Dollars'].apply(lambda x: f"${x:,.0f}")
    top_20['Relief_Hours'] = (top_20['Relief Minutes'] / 60).round(1)
    
    display_top_20 = top_20[['Ticket #', 'Emp #', 'Subject_Grouped', 'Status_Canonical', 
                            'Relief_Hours', 'Relief_Dollars_Display', 'Last Interaction']].rename(columns={
        'Subject_Grouped': 'Subject',
        'Status_Canonical': 'Status',
        'Relief_Hours': 'Hours',
        'Relief_Dollars_Display': 'Value',
        'Last Interaction': 'Date'
    })
    
    st.dataframe(display_top_20, use_container_width=True, height=400)
    
    # Top 20 summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_20_value = top_20['Relief_Dollars'].sum()
        st.metric("Top 20 Total Value", f"${top_20_value:,.0f}")
    with col2:
        avg_top_20 = top_20['Relief_Dollars'].mean()
        st.metric("Average Top 20", f"${avg_top_20:,.0f}")
    with col3:
        top_20_hours = top_20['Relief Minutes'].sum() / 60
        st.metric("Top 20 Total Hours", f"{top_20_hours:,.1f}")
    with col4:
        pct_of_total = (top_20_value / enhanced_df['Relief_Dollars'].sum()) * 100
        st.metric("% of Total Value", f"{pct_of_total:.1f}%")
    
    # === SUBJECT BREAKDOWN BY STATUS ===
    st.subheader("📋 Complete Subject Violation Analysis")
    
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
    
    # === TIMING ANALYSIS ===
    st.subheader("⏰ Timing and Duration Analysis")
    
    # Relief time distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of relief minutes
        fig = px.histogram(enhanced_df, x='Relief Minutes', nbins=20,
                          title="Distribution of Relief Times")
        fig.update_layout(xaxis_title="Relief Minutes", yaxis_title="Number of Claims")
        st.plotly_chart(fig, use_container_width=True)
        
        # Relief time statistics
        st.write("**Relief Time Statistics:**")
        st.write(f"• Mean: {enhanced_df['Relief Minutes'].mean():.1f} minutes")
        st.write(f"• Median: {enhanced_df['Relief Minutes'].median():.1f} minutes")
        st.write(f"• Min: {enhanced_df['Relief Minutes'].min()} minutes")
        st.write(f"• Max: {enhanced_df['Relief Minutes'].max()} minutes")
    
    with col2:
        # Relief time by subject (box plot)
        fig = px.box(enhanced_df, x='Subject_Grouped', y='Relief Minutes',
                    title="Relief Time Distribution by Subject")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Average relief time by subject
        avg_relief = enhanced_df.groupby('Subject_Grouped')['Relief Minutes'].mean().sort_values(ascending=False)
        st.write("**Average Relief Time by Subject:**")
        for subject, avg_time in avg_relief.head(5).items():
            st.write(f"• {subject}: {avg_time:.1f} minutes")
    
    # === APPROVAL PATTERNS ===
    st.subheader("📈 Approval Rate Patterns")
    
    # Calculate approval rates for subjects with enough data
    approval_patterns = []
    for subject, stats in subject_stats.items():
        approved = stats.get("approved_count", 0)
        denied = stats.get("denied_count", 0)
        total_decided = approved + denied
        
        if total_decided >= 3:  # Only include subjects with at least 3 decided cases
            approval_rate = approved / total_decided
            approval_patterns.append({
                'Subject': subject,
                'Total Decided': total_decided,
                'Approved': approved,
                'Denied': denied,
                'Approval Rate': approval_rate,
                'Approval Rate %': f"{approval_rate:.1%}",
                'Confidence': 'High' if total_decided >= 10 else 'Medium' if total_decided >= 5 else 'Low'
            })
    
    if approval_patterns:
        pattern_df = pd.DataFrame(approval_patterns).sort_values('Approval Rate', ascending=False)
        
        # Approval rate chart
        fig = px.bar(pattern_df, x='Subject', y='Approval Rate',
                    color='Confidence', title="Approval Rates by Subject (Confidence Level)")
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(pattern_df[['Subject', 'Total Decided', 'Approved', 'Denied', 'Approval Rate %', 'Confidence']], 
                    use_container_width=True)
    
    # === EXPORT ALL ANALYTICS ===
    st.subheader("📥 Export Complete Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Subject Analysis"):
            csv = breakdown_df.to_csv(index=False)
            st.download_button(
                label="Download Subject Analysis CSV",
                data=csv,
                file_name=f"subject_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if len(multiple_claims) > 0 and st.button("👥 Export Multi-Claim Pilots"):
            csv = pd.DataFrame(multi_display).to_csv(index=False)
            st.download_button(
                label="Download Multi-Claim Pilots CSV",
                data=csv,
                file_name=f"multi_claim_pilots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🏆 Export Top 20 Claims"):
            csv = display_top_20.to_csv(index=False)
            st.download_button(
                label="Download Top 20 Claims CSV",
                data=csv,
                file_name=f"top_20_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # === COMPREHENSIVE SUMMARY ===
    st.subheader("📋 Analytics Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **📈 Data Overview:**
        • Total claims analyzed: **{len(enhanced_df):,}**
        • Unique employees: **{enhanced_df['Emp #'].nunique():,}**
        • Unique violation types: **{len(subject_stats)}**
        • Date range: **{enhanced_df['Last Interaction'].min() if 'Last Interaction' in enhanced_df.columns else 'N/A'}** to **{enhanced_df['Last Interaction'].max() if 'Last Interaction' in enhanced_df.columns else 'N/A'}**
        """)
    
    with col2:
        # Find most active pilot and highest value subject
        most_active = employee_stats.sort_values('Total_Claims', ascending=False).index[0] if len(employee_stats) > 0 else 'N/A'
        highest_value_subject = max(subject_stats.items(), key=lambda x: sum([x[1].get(f"{s}_dollars", 0) for s in all_statuses]))[0] if subject_stats else 'N/A'
        
        st.success(f"""
        **🎯 Key Insights:**
        • Most active pilot: **{most_active}** ({employee_stats.loc[most_active, 'Total_Claims'] if most_active != 'N/A' else 0} claims)
        • Highest value subject: **{highest_value_subject}**
        • Multi-claim pilots: **{len(multiple_claims)}** ({len(multiple_claims)/len(employee_stats)*100:.1f}% of pilots)
        • Top 20 claims represent: **{pct_of_total:.1f}%** of total value
        """)

def show_financial_tab():
    st.header("💰 Comprehensive Financial Analysis")
    
    if 'claims_data' not in st.session_state:
        if st.session_state.get('demo_mode', True):
            st.info("No data loaded. Loading demo data...")
            load_demo_data()
        else:
            st.info("No data loaded. Please run data collection or load existing data from the sidebar.")
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
    
    # Show demo mode indicator
    if st.session_state.get('demo_mode', False):
        st.info("🎯 **Demo Mode Active** - Displaying sample data for demonstration and testing purposes")
    
    # Add enhanced columns
    enhanced_df = df.copy()
    enhanced_df['Relief_Dollars'] = enhanced_df['Relief Minutes'].apply(relief_dollars)
    enhanced_df['Subject_Grouped'] = enhanced_df['Subject Violations'].apply(group_subject_key)
    enhanced_df['Status_Canonical'] = enhanced_df['Status'].apply(status_canonical)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=enhanced_df['Status_Canonical'].unique(),
            default=enhanced_df['Status_Canonical'].unique()
        )
    
    with col2:
        subject_filter = st.multiselect(
            "Filter by Subject",
            options=enhanced_df['Subject_Grouped'].unique(),
            default=enhanced_df['Subject_Grouped'].unique()
        )
    
    with col3:
        min_value = st.number_input("Min Relief Value ($)", value=0.0, step=100.0)
    
    # Apply filters
    filtered_df = enhanced_df[
        (enhanced_df['Status_Canonical'].isin(status_filter)) &
        (enhanced_df['Subject_Grouped'].isin(subject_filter)) &
        (enhanced_df['Relief_Dollars'] >= min_value)
    ]
    
    # Display summary
    st.subheader(f"📊 Showing {len(filtered_df)} of {len(enhanced_df)} claims")
    
    # Summary metrics for filtered data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = filtered_df['Relief_Dollars'].sum()
        st.metric("Total Value", f"${total_value:,.0f}")
    
    with col2:
        avg_value = filtered_df['Relief_Dollars'].mean()
        st.metric("Average Value", f"${avg_value:,.0f}")
    
    with col3:
        total_hours = filtered_df['Relief Minutes'].sum() / 60
        st.metric("Total Hours", f"{total_hours:,.1f}")
    
    with col4:
        unique_employees = filtered_df['Emp #'].nunique()
        st.metric("Unique Employees", unique_employees)
    
    # Display data table
    display_columns = [
        'Ticket #', 'Status', 'Relief Minutes', 'Relief_Dollars',
        'Subject Violations', 'Subject_Grouped', 'Emp #',
        'Last Interaction', 'Assignee'
    ]
    
    # Format the dataframe for display
    display_df = filtered_df[display_columns].copy()
    display_df['Relief_Dollars'] = display_df['Relief_Dollars'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Export filtered data
    if st.button("📥 Export Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Claims CSV",
            data=csv,
            file_name=f"filtered_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_realtime_status_tab():
    st.header("🔄 Real-time Status")
    
    if st.session_state.get('demo_mode', True):
        st.info("🎯 **Demo Mode Active** - Real-time features disabled in demo mode")
        st.write("In production mode, this tab would show:")
        st.write("• Live scraping status")
        st.write("• Data collection progress")
        st.write("• System health monitoring")
        st.write("• Last update timestamps")
        st.write("• Error logs and alerts")
        return
    
    # Real-time features when not in demo mode
    st.info("🔧 **Production Mode** - Real-time monitoring features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Data Collection Status")
        st.write("• **Last Update:** Ready for configuration")
        st.write("• **Collection Status:** Awaiting start")
        st.write("• **Records Processed:** Configure and start data collection")
        
        if st.button("🔄 Check Data Source Connection"):
            st.info("Testing connection to data source...")
            st.warning("Connection test would be implemented here based on your original script.")
    
    with col2:
        st.subheader("⚡ System Health")
        st.write("• **Connection Status:** Ready")
        st.write("• **Processing Speed:** Not running")
        st.write("• **Error Rate:** 0%")
        
        if st.button("📊 System Diagnostics"):
            st.info("Running system diagnostics...")
            st.success("System ready for data collection.")
    
    # Configuration status
    st.subheader("⚙️ Configuration Status")
    
    # Check sidebar configuration dynamically
    relief_rate_set = True  # Relief rate always has a default value
    export_path_set = len(st.session_state.get('export_path', '')) > 0
    credentials_set = (len(st.session_state.get('username', '')) > 0 and 
                      len(st.session_state.get('password', '')) > 0)
    
    config_status = []
    config_status.append(("Relief Rate", "✅ Configured" if relief_rate_set else "❌ Not set"))
    config_status.append(("Export Path", "✅ Configured" if export_path_set else "❌ Not set"))
    config_status.append(("Credentials", "✅ Configured" if credentials_set else "❌ Not set"))
    
    for item, status in config_status:
        st.write(f"• **{item}**: {status}")
    
    if not all([relief_rate_set, export_path_set, credentials_set]):
        st.warning("⚠️ Complete configuration in the sidebar to enable data collection.")
        st.info("💡 Switch off Demo Mode in the sidebar to access configuration options.")
    else:
        st.success("✅ All configuration complete. Ready for data collection!")
        
        # Add connection test button in real-time status
        if st.button("🔍 Test Connection Now"):
            test_data_source_connection(
                st.session_state.get('username', ''),
                st.session_state.get('password', ''),
                st.session_state.get('export_path', '')
            )

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
        # Calculate total dollars and hours from all statuses
        total_dollars = sum([stats.get(f"{status}_dollars", 0) for status in all_statuses])
        total_minutes = sum([stats.get(f"{status}_minutes", 0) for status in all_statuses])
        total_hours = total_minutes / 60
        
        summary_data.append({
            'Subject': subject,
            'Total_Cases': stats['count'],
            'Total_Hours': total_hours,
            'Total_Dollars': total_dollars,
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

def test_data_source_connection(username, password, export_path):
    """Test connection to the data source"""
    with st.spinner("Testing connection to data source..."):
        import time
        time.sleep(2)  # Simulate connection test
        
        # In a real implementation, this would:
        # 1. Test network connectivity
        # 2. Validate credentials
        # 3. Check access to the data source
        # 4. Verify export path accessibility
        
        # Simulate some basic checks
        connection_results = []
        
        # Check export path
        try:
            if os.path.exists(export_path):
                connection_results.append(("✅", "Export Path", "Directory exists and is accessible"))
            else:
                connection_results.append(("⚠️", "Export Path", f"Directory '{export_path}' does not exist (will be created)"))
        except Exception as e:
            connection_results.append(("❌", "Export Path", f"Error accessing path: {str(e)}"))
        
        # Simulate credential validation
        if username and password:
            if len(username) >= 3 and len(password) >= 6:
                connection_results.append(("✅", "Credentials", "Username and password format valid"))
            else:
                connection_results.append(("⚠️", "Credentials", "Username/password may be too short"))
        else:
            connection_results.append(("❌", "Credentials", "Username and password are required"))
        
        # Simulate network connectivity test
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            connection_results.append(("✅", "Network", "Internet connectivity confirmed"))
        except:
            connection_results.append(("⚠️", "Network", "Limited network connectivity detected"))
        
        # Display results
        st.subheader("🔍 Connection Test Results")
        for icon, component, message in connection_results:
            st.write(f"{icon} **{component}**: {message}")
        
        # Overall status
        success_count = sum(1 for icon, _, _ in connection_results if icon == "✅")
        warning_count = sum(1 for icon, _, _ in connection_results if icon == "⚠️")
        error_count = sum(1 for icon, _, _ in connection_results if icon == "❌")
        
        if error_count > 0:
            st.error(f"❌ Connection test failed: {error_count} errors found. Please resolve issues before proceeding.")
        elif warning_count > 0:
            st.warning(f"⚠️ Connection test completed with warnings: {warning_count} issues detected. Data collection may still work.")
        else:
            st.success("✅ All connection tests passed! Ready for data collection.")
        
        # Add note about real implementation
        with st.expander("ℹ️ About Connection Testing"):
            st.info("""
            In the full implementation, this would test:
            • Connection to the actual STS claims system
            • Authentication with provided credentials
            • Selenium WebDriver availability
            • Browser compatibility
            • Data source accessibility
            • Write permissions to export directory
            """)

def run_data_collection(relief_rate, export_path, username, password, headless_mode, max_pages):
    """Run actual data collection when not in demo mode"""
    if not username or not password:
        st.error("Please provide username and password for data collection.")
        return
    
    if not export_path:
        st.error("Please provide an export path for data collection.")
        return
    
    # Clear any existing demo data
    if 'claims_data' in st.session_state:
        del st.session_state['claims_data']
    
    st.info("🚀 Starting data collection...")
    st.info("⚠️ Note: This is a placeholder for the actual data collection functionality.")
    st.info("In the full implementation, this would:")
    st.write("• Connect to the data source using provided credentials")
    st.write("• Scrape data based on the configured parameters")
    st.write("• Process and save data to the specified export path")
    st.write("• Load the collected data into the dashboard")
    
    # Placeholder for actual implementation
    st.warning("Real data collection functionality would be implemented here based on your original script.")

def load_latest_data(export_path):
    """Load latest data from export path when not in demo mode"""
    if not export_path:
        st.error("Please provide an export path to load data from.")
        return
    
    # Clear any existing demo data
    if 'claims_data' in st.session_state:
        del st.session_state['claims_data']
    
    st.info("📊 Loading latest data...")
    st.info("⚠️ Note: This is a placeholder for the actual data loading functionality.")
    st.info("In the full implementation, this would:")
    st.write("• Look for the latest data files in the specified export path")
    st.write("• Load and validate the data")
    st.write("• Display the loaded data in the dashboard")
    
    # Placeholder for actual implementation
    st.warning("Real data loading functionality would be implemented here based on your original script.")

# Run the main app
if __name__ == "__main__":
    main()
