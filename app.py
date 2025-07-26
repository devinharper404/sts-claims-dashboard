import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import statistics
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="STS Claims Analytics Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .status-approved { background-color: #28a745; }
    .status-denied { background-color: #dc3545; }
    .status-pending { background-color: #ffc107; color: #000; }
    .status-review { background-color: #17a2b8; }
    .status-open { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True
if 'data_collected' not in st.session_state:
    st.session_state.data_collected = False
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = pd.DataFrame()
if 'collection_status' not in st.session_state:
    st.session_state.collection_status = {
        'running': False,
        'current_step': '',
        'claims_found': 0,
        'pages_processed': 0,
        'start_time': None,
        'export_file': None
    }

# --- ANALYTICS FUNCTIONS FROM ORIGINAL SCRIPT ---

# --- ANALYTICS FUNCTIONS FROM ORIGINAL SCRIPT ---

def hhmm_to_minutes(hhmm):
    """Convert HH:MM format to minutes"""
    try:
        if not hhmm:
            return 0
        hhmm = hhmm.replace(';', ':')
        hours, minutes = map(int, hhmm.strip().split(':'))
        return hours * 60 + minutes
    except:
        return 0

def extract_emp_number(empcat):
    """Extract employee number from emp category string"""
    m = re.search(r'\b(\d{6})\b', empcat)
    if m:
        return m.group(1)
    return empcat.strip()

def relief_dollars(relief_minutes, relief_rate=320.47):
    """Convert relief minutes to dollar amount"""
    try:
        return float(relief_minutes) / 60 * relief_rate
    except (ValueError, TypeError):
        return 0

def group_subject_key(subject):
    """Group subjects by key patterns from original script"""
    if pd.isna(subject) or subject == '':
        return 'Other'
    
    subject_str = str(subject).strip()
    
    # Define grouping patterns based on original script
    if any(x in subject_str for x in ['Rest', 'rest']):
        return 'Rest'
    elif any(x in subject_str for x in ['11.F', '11F']):
        return '11.F'
    elif any(x in subject_str for x in ['Yellow Slip', '12.T', '12T']):
        return 'Yellow Slip / 12.T'
    elif any(x in subject_str for x in ['Green Slip', '23.Q', '23Q']):
        return 'Green Slip / 23.Q'
    elif any(x in subject_str for x in ['Short Call', 'Short-Call']):
        return 'Short Call'
    elif any(x in subject_str for x in ['Long Call', 'Long-Call']):
        return 'Long Call'
    elif any(x in subject_str for x in ['23.O', '23O']):
        return '23.O'
    elif any(x in subject_str for x in ['Deadhead', '8.D', '8D']):
        return 'Deadhead / 8.D'
    elif any(x in subject_str for x in ['Payback', '23.S.11', '23S11']):
        return 'Payback Day / 23.S.11'
    elif any(x in subject_str for x in ['White Slip', '23.P', '23P']):
        return 'White Slip / 23.P'
    elif any(x in subject_str for x in ['Reroute', '23.L', '23L']):
        return 'Reroute / 23.L'
    elif any(x in subject_str for x in ['ARCOS', '23.Z', '23Z']):
        return 'ARCOS / 23.Z'
    elif any(x in subject_str for x in ['Inverse', '23.R', '23R']):
        return 'Inverse Assignment / 23.R'
    elif any(x in subject_str for x in ['23.J', '23J']):
        return '23.J'
    elif any(x in subject_str for x in ['4.F', '4F']):
        return '4.F'
    else:
        return 'Other'

def status_canonical(status):
    """Canonicalize status values with partial matching"""
    if pd.isna(status) or status == '':
        return 'unknown'
    
    status_str = str(status).lower().strip()
    
    # Use partial matching for common status values
    if 'paid' in status_str or 'approved' in status_str:
        return 'approved'
    elif 'denied' in status_str or 'reject' in status_str:
        return 'denied'
    elif 'review' in status_str:
        return 'in review'
    elif 'open' in status_str:
        return 'open'
    elif 'pending' in status_str:
        return 'pending'
    elif 'impasse' in status_str:
        return 'impasse'
    else:
        return status_str

def calculate_comprehensive_analytics(df, relief_rate=320.47):
    """Calculate all analytics from original script including cost analytics"""
    # Add grouped subject and canonical status
    df = df.copy()
    df['Subject_Grouped'] = df['subject'].apply(group_subject_key)
    df['Status_Canonical'] = df['status'].apply(status_canonical)
    
    # Ensure we have relief_dollars - either from existing column or calculate from relief_minutes
    if 'relief_minutes' in df.columns and 'relief_dollars' not in df.columns:
        df['Relief_Dollars'] = df['relief_minutes'].apply(lambda x: relief_dollars(x, relief_rate))
    elif 'relief_dollars' in df.columns:
        df['Relief_Dollars'] = df['relief_dollars']
    else:
        df['Relief_Dollars'] = 0
    
    # Get all statuses
    all_statuses = sorted(df['Status_Canonical'].unique())
    
    # Subject grouped stats with dollars per status
    subject_stats = {}
    
    for subject in df['Subject_Grouped'].unique():
        subject_data = df[df['Subject_Grouped'] == subject]
        stats = {"count": len(subject_data), "minutes": subject_data['Relief_Dollars'].sum()}
        
        for status in all_statuses:
            status_data = subject_data[subject_data['Status_Canonical'] == status]
            # Use safe key format - replace spaces with underscores
            safe_status = status.replace(' ', '_').replace('-', '_')
            stats[f"{safe_status}_count"] = len(status_data)
            stats[f"{safe_status}_minutes"] = status_data['Relief_Dollars'].sum()
            stats[f"{safe_status}_dollars"] = status_data['Relief_Dollars'].sum()
            stats[f"{safe_status}_pct"] = (len(status_data) / len(subject_data) * 100) if len(subject_data) > 0 else 0
            
        subject_stats[subject] = stats
    
    # Calculate probability of payment by subject
    probability_by_subject = {}
    for subject, stats in subject_stats.items():
        approved = stats.get("approved_count", 0)
        denied = stats.get("denied_count", 0)
        total_decided = approved + denied
        probability_by_subject[subject] = approved / total_decided if total_decided > 0 else 0
        probability_by_subject[subject] = approved / total_decided if total_decided > 0 else 0
    
    # Calculate average relief minutes per subject
    avg_relief_minutes_per_subject = {}
    for subject, stats in subject_stats.items():
        total_cases = stats["count"]
        avg_relief_minutes_per_subject[subject] = stats["minutes"] / total_cases if total_cases > 0 else 0
    
    # Calculate comprehensive cost analytics
    cost_analytics = calculate_cost_analytics(df, subject_stats, probability_by_subject, avg_relief_minutes_per_subject, relief_rate)
    
    # Calculate aging forecast
    aging_data = aging_forecast(df)
    
    # Calculate monthly trends
    monthly_trends = monthly_trends_analysis(df)
    
    # Calculate outlier analysis
    relief_q1 = df['Relief_Dollars'].quantile(0.25)
    relief_q3 = df['Relief_Dollars'].quantile(0.75)
    iqr = relief_q3 - relief_q1
    outlier_threshold_high = relief_q3 + 1.5 * iqr
    outlier_threshold_low = relief_q1 - 1.5 * iqr
    
    outlier_analysis = {
        'high_cost_outliers': df[df['Relief_Dollars'] > outlier_threshold_high].to_dict('records'),
        'low_cost_outliers': df[df['Relief_Dollars'] < outlier_threshold_low].to_dict('records'),
        'outlier_threshold_high': outlier_threshold_high,
        'outlier_threshold_low': outlier_threshold_low
    }
    
    # Calculate pilots with multiple submissions
    pilot_counts = df['pilot'].value_counts()
    pilots_multiple_submissions = pilot_counts[pilot_counts > 1].to_dict()
    
    # Top 20 highest value claims
    top_20_claims = df.nlargest(20, 'Relief_Dollars')[['case_number', 'pilot', 'subject', 'Relief_Dollars', 'status']].to_dict('records')
    
    return {
        'subject_stats': subject_stats,
        'all_statuses': all_statuses,
        'processed_df': df,
        'probability_by_subject': probability_by_subject,
        'avg_relief_minutes_per_subject': avg_relief_minutes_per_subject,
        'cost_analytics': cost_analytics,
        'aging_forecast': aging_data,
        'monthly_trends': monthly_trends,
        'outlier_analysis': outlier_analysis,
        'pilots_multiple_submissions': pilots_multiple_submissions,
        'top_20_claims': top_20_claims,
        'total_claims': len(df),
        'total_relief': df['Relief_Dollars'].sum(),
        'avg_relief': df['Relief_Dollars'].mean(),
        'median_relief': df['Relief_Dollars'].median()
    }

def calculate_cost_analytics(df, subject_stats, probability_by_subject, avg_relief_minutes_per_subject, relief_rate):
    """Calculate comprehensive cost analytics from original script"""
    
    # Filter claims by status
    approved_claims = df[df['Status_Canonical'] == 'approved'].copy()
    open_claims = df[df['Status_Canonical'] == 'open'].copy()
    in_review_claims = df[df['Status_Canonical'] == 'in review'].copy()
    
    # Total actual paid (approved) - use Relief_Dollars column
    total_actual_paid_cost = approved_claims['Relief_Dollars'].sum()
    num_approved_cases = len(approved_claims)
    avg_paid_per_case = total_actual_paid_cost / num_approved_cases if num_approved_cases > 0 else 0
    
    # Actual paid by pilot
    actual_paid_by_pilot = {}
    if len(approved_claims) > 0:
        actual_paid_by_pilot = approved_claims.groupby('pilot')['Relief_Dollars'].sum().to_dict()
    top20_actual_paid_by_pilot = sorted(actual_paid_by_pilot.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Actual paid by subject group
    actual_paid_by_subject = {}
    if len(approved_claims) > 0:
        actual_paid_by_subject = approved_claims.groupby('Subject_Grouped')['Relief_Dollars'].sum().to_dict()
    
    # Forecasted cost by subject - fix the calculation
    forecasted_cost_by_subject = {}
    for subject in df['Subject_Grouped'].unique():
        subject_data = df[df['Subject_Grouped'] == subject]
        open_and_review = subject_data[subject_data['Status_Canonical'].isin(['open', 'in review'])]
        
        if len(open_and_review) > 0:
            prob = probability_by_subject.get(subject, 0)
            # Use average relief dollars for this subject from historical data
            avg_relief = subject_data['Relief_Dollars'].mean() if len(subject_data) > 0 else 0
            forecasted_cost_by_subject[subject] = len(open_and_review) * prob * avg_relief
        else:
            forecasted_cost_by_subject[subject] = 0
    
    # Top 20 forecasted by pilot
    pilot_forecasts = {}
    for pilot in df['pilot'].unique():
        pilot_data = df[df['pilot'] == pilot]
        pilot_forecasts[pilot] = _calculate_pilot_forecast(pilot_data, probability_by_subject, avg_relief_minutes_per_subject, relief_rate)
    
    top20_forecasted_by_pilot = sorted(pilot_forecasts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Status costs
    status_costs = df.groupby('Status_Canonical')['Relief_Dollars'].sum().to_dict()
    status_avg_costs = df.groupby('Status_Canonical')['Relief_Dollars'].mean().to_dict()
    
    # Outlier cases (high cost)
    relief_q3 = df['Relief_Dollars'].quantile(0.75)
    relief_q1 = df['Relief_Dollars'].quantile(0.25)
    iqr = relief_q3 - relief_q1
    outlier_threshold = relief_q3 + 1.5 * iqr
    outlier_cases = df[df['Relief_Dollars'] > outlier_threshold].to_dict('records')
    
    # Cost statistics
    paid_costs = approved_claims['Relief_Dollars'].tolist() if len(approved_claims) > 0 else [0]
    median_paid_cost = statistics.median(paid_costs)
    mean_paid_cost = statistics.mean(paid_costs)
    
    forecasted_costs_list = list(forecasted_cost_by_subject.values())
    median_forecasted_cost = statistics.median(forecasted_costs_list) if forecasted_costs_list else 0
    mean_forecasted_cost = statistics.mean(forecasted_costs_list) if forecasted_costs_list else 0
    
    total_forecasted_cost = sum(forecasted_costs_list)
    cost_variance = total_actual_paid_cost - total_forecasted_cost
    variance_percentage = (cost_variance / total_forecasted_cost * 100) if total_forecasted_cost > 0 else 0
    
    # Top pilots by cost
    top_pilots_by_cost = [{'pilot': pilot, 'total_cost': cost} for pilot, cost in top20_actual_paid_by_pilot]
    
    return {
        'total_actual_cost': total_actual_paid_cost,
        'total_forecasted_cost': total_forecasted_cost,
        'cost_variance': cost_variance,
        'variance_percentage': variance_percentage,
        'top_pilots_by_cost': top_pilots_by_cost,
        'cost_by_status': status_costs,
        'forecasted_cost_by_subject': forecasted_cost_by_subject,
        'cost_statistics': {
            'mean_cost': mean_paid_cost,
            'median_cost': median_paid_cost,
            'min_cost': df['relief_dollars'].min(),
            'max_cost': df['relief_dollars'].max(),
            'std_cost': df['relief_dollars'].std(),
            'cost_variance_stat': df['relief_dollars'].var()
        },
        'total_actual_paid_cost': total_actual_paid_cost,
        'num_approved_cases': num_approved_cases,
        'avg_paid_per_case': avg_paid_per_case,
        'top20_actual_paid_by_pilot': top20_actual_paid_by_pilot,
        'actual_paid_by_subject': actual_paid_by_subject,
        'top20_forecasted_by_pilot': top20_forecasted_by_pilot,
        'status_costs': status_costs,
        'status_avg_costs': status_avg_costs,
        'outlier_cases': outlier_cases,
        'median_paid_cost': median_paid_cost,
        'mean_paid_cost': mean_paid_cost,
        'median_forecasted_cost': median_forecasted_cost,
        'mean_forecasted_cost': mean_forecasted_cost
    }

def _calculate_pilot_forecast(pilot_data, probability_by_subject, avg_relief_minutes_per_subject, relief_rate):
    """Helper function to calculate forecasted cost for a pilot"""
    total_forecast = 0
    for _, row in pilot_data.iterrows():
        if row['Status_Canonical'] in ['open', 'in review']:
            subject = row['Subject_Grouped']
            prob = probability_by_subject.get(subject, 0)
            # Use the average relief for this subject from all data
            avg_relief = pilot_data[pilot_data['Subject_Grouped'] == subject]['Relief_Dollars'].mean()
            if pd.isna(avg_relief):
                avg_relief = 0
            total_forecast += prob * avg_relief
    return total_forecast

def aging_forecast(df):
    """Generate aging forecast analysis"""
    try:
        current_date = datetime.now()
        aging_data = []
        
        for _, row in df.iterrows():
            if pd.isna(row.get('decision_date')) or row.get('decision_date') == '':
                # Case is still open, calculate aging
                submission_date = pd.to_datetime(row['submission_date'])
                days_old = (current_date - submission_date).days
                
                # Predict likely decision timeframe based on subject and status
                subject_avg_days = {
                    'Rest': 45, '11.F': 60, 'Yellow Slip / 12.T': 30, 'Green Slip / 23.Q': 35,
                    'Short Call': 25, 'Long Call': 35, '23.O': 40, 'Deadhead / 8.D': 30,
                    'Payback Day / 23.S.11': 50, 'White Slip / 23.P': 35, 'Reroute / 23.L': 40,
                    'ARCOS / 23.Z': 45, 'Inverse Assignment / 23.R': 55, '23.J': 40, '4.F': 35,
                    'Other': 60
                }
                
                subject_grouped = group_subject_key(row.get('subject', ''))
                expected_days = subject_avg_days.get(subject_grouped, 60)
                estimated_decision_date = submission_date + timedelta(days=expected_days)
                
                aging_data.append({
                    'case_number': row.get('case_number', ''),
                    'pilot': row.get('pilot', ''),
                    'subject': row.get('subject', ''),
                    'days_old': days_old,
                    'estimated_decision_date': estimated_decision_date,
                    'status': row.get('status', ''),
                    'relief_dollars': row.get('relief_dollars', 0)
                })
        
        return pd.DataFrame(aging_data)
    
    except Exception as e:
        print(f"Error in aging_forecast: {e}")
        return pd.DataFrame()

def monthly_trends_analysis(df):
    """Analyze monthly trends in claims data"""
    try:
        # Convert submission_date to datetime
        df['submission_date'] = pd.to_datetime(df['submission_date'])
        df['month_year'] = df['submission_date'].dt.to_period('M')
        
        # Monthly submission trends
        monthly_submissions = df.groupby('month_year').size()
        
        # Monthly cost trends
        monthly_costs = df.groupby('month_year')['relief_dollars'].sum()
        
        # Monthly approval rates
        approved_cases = df[df['status'].str.contains('approved|paid', case=False, na=False)]
        monthly_approvals = approved_cases.groupby('month_year').size()
        monthly_approval_rates = (monthly_approvals / monthly_submissions * 100).fillna(0)
        
        return {
            'submissions_by_month': {str(k): v for k, v in monthly_submissions.to_dict().items()},
            'cost_by_month': {str(k): v for k, v in monthly_costs.to_dict().items()},
            'approval_rates_by_month': {str(k): v for k, v in monthly_approval_rates.to_dict().items()}
        }
    
    except Exception as e:
        print(f"Error in monthly_trends_analysis: {e}")
        return {}

def generate_demo_data():
    """Generate comprehensive demo data for testing"""
    np.random.seed(42)  # For consistent demo data
    
    # Create demo data with realistic patterns
    n_claims = 150
    pilots = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliet']
    subjects = ['Rest', '11.F', 'Yellow Slip / 12.T', 'Green Slip / 23.Q', 'Short Call', 'Long Call', '23.O', 'Deadhead / 8.D', 'Payback Day / 23.S.11', 'White Slip / 23.P']
    
    # Realistic status distribution - ensure we have decided cases for probability calculations
    statuses = ['approved', 'denied', 'in review', 'open', 'pending']
    status_weights = [0.35, 0.25, 0.15, 0.15, 0.10]  # 35% approved, 25% denied, etc.
    
    data = []
    for i in range(n_claims):
        # Create some pilots with multiple cases
        if i < 30:  # First 30 claims use repeating pilots for multi-case scenarios
            pilot = pilots[i % 5]  # Use first 5 pilots repeatedly
        else:
            pilot = np.random.choice(pilots)
        
        # Generate realistic financial data
        relief_amount = np.random.lognormal(6, 1.2) * 100  # Log-normal for realistic money distribution
        
        # Weighted status selection for realistic distributions
        status = np.random.choice(statuses, p=status_weights)
        
        # Generate dates
        submission_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        
        # Only generate decision date for approved/denied cases
        if status in ['approved', 'denied']:
            decision_date = submission_date + timedelta(days=np.random.randint(30, 180))
        else:
            decision_date = None
        
        # Generate subject with some bias toward certain types having higher approval rates
        subject = np.random.choice(subjects)
        
        # Adjust approval probability based on subject type (realistic patterns)
        if status in ['approved', 'denied'] and subject in ['Rest', 'Short Call', 'Yellow Slip / 12.T']:
            # These subjects have higher approval rates in real data
            if np.random.random() < 0.7:  # 70% chance to be approved
                status = 'approved'
            else:
                status = 'denied'
        
        data.append({
            'case_number': f'STS-{2024}-{i+1:04d}',
            'pilot': pilot,
            'subject': subject,
            'status': status,
            'relief_dollars': relief_amount,
            'submission_date': submission_date.strftime('%Y-%m-%d'),
            'decision_date': decision_date.strftime('%Y-%m-%d') if decision_date else '',
            'processing_days': (decision_date - submission_date).days if decision_date else None,
            'violation_type': np.random.choice(['Type A', 'Type B', 'Type C', 'Type D']),
            'probability_of_payment': np.random.uniform(0.1, 0.9)
        })
    
    return pd.DataFrame(data)

def get_data():
    """Get the current data from session state or demo data"""
    if st.session_state.get('demo_mode', True):
        return generate_demo_data()
    else:
        return st.session_state.get('collected_data', pd.DataFrame())

# --- DASHBOARD FUNCTIONS ---

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

def show_overview_tab():
    """Display overview analytics"""
    st.subheader("📈 Overview Analytics")
    
    # Get data
    df = get_data()
    if df.empty:
        st.warning("No data available. Please collect data or enable demo mode.")
        return
    
    try:
        analytics = calculate_comprehensive_analytics(df)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Claims", analytics['total_claims'])
        with col2:
            st.metric("Total Relief Value", f"${analytics['total_relief']:,.2f}")
        with col3:
            st.metric("Average Relief", f"${analytics['avg_relief']:,.2f}")
        with col4:
            st.metric("Median Relief", f"${analytics['median_relief']:,.2f}")
        
        # Status distribution
        st.subheader("📊 Claims Status Distribution")
        if analytics['processed_df'] is not None:
            status_counts = analytics['processed_df']['Status_Canonical'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, 
                        title="Claims by Status")
            st.plotly_chart(fig, use_container_width=True)
        
        # Pilots with multiple submissions
        if analytics['pilots_multiple_submissions']:
            st.subheader("👥 Pilots with Multiple Submissions")
            pilots_multi_df = pd.DataFrame(list(analytics['pilots_multiple_submissions'].items()), 
                                         columns=['Pilot', 'Number of Claims'])
            pilots_multi_df = pilots_multi_df.sort_values('Number of Claims', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(pilots_multi_df, use_container_width=True)
            with col2:
                fig = px.bar(pilots_multi_df.head(10), x='Pilot', y='Number of Claims',
                           title="Top 10 Pilots by Number of Claims")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying overview: {str(e)}")

def show_analytics_tab():
    """Display detailed analytics"""
    st.subheader("📊 Detailed Analytics")
    
    df = get_data()
    if df.empty:
        st.warning("No data available for analytics.")
        return
    
    try:
        analytics = calculate_comprehensive_analytics(df)
        
        # Top 20 highest value claims
        st.subheader("🏆 Top 20 Highest Value Claims")
        if analytics['top_20_claims']:
            top_claims_df = pd.DataFrame(analytics['top_20_claims'])
            st.dataframe(top_claims_df, use_container_width=True)
            
            # Chart for top 10
            fig = px.bar(top_claims_df.head(10), x='case_number', y='relief_dollars',
                        title="Top 10 Claims by Relief Value", hover_data=['pilot', 'subject'])
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Subject analysis
        st.subheader("📋 Subject Violation Analysis")
        if analytics['subject_stats']:
            subject_summary = []
            for subject, stats in analytics['subject_stats'].items():
                subject_summary.append({
                    'Subject': subject,
                    'Total Cases': stats['count'],
                    'Total Relief ($)': stats['minutes'],
                    'Approved Cases': stats.get('approved_count', 0),
                    'Denied Cases': stats.get('denied_count', 0),
                    'Approval Rate (%)': round(stats.get('approved_pct', 0), 1)
                })
            
            subject_df = pd.DataFrame(subject_summary)
            subject_df = subject_df.sort_values('Total Relief ($)', ascending=False)
            st.dataframe(subject_df, use_container_width=True)
            
            # Subject distribution chart
            fig = px.bar(subject_df.head(10), x='Subject', y='Total Relief ($)',
                        title="Top 10 Subjects by Total Relief Value")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Outlier analysis
        if analytics.get('outlier_analysis') and analytics['outlier_analysis'].get('high_cost_outliers'):
            st.subheader("🔍 High Cost Outlier Analysis")
            outliers = analytics['outlier_analysis']['high_cost_outliers']
            if outliers:
                outliers_df = pd.DataFrame(outliers)
                st.dataframe(outliers_df[['case_number', 'pilot', 'subject', 'relief_dollars', 'status']], 
                           use_container_width=True)
        
        # Monthly trends
        if analytics.get('monthly_trends'):
            st.subheader("📈 Monthly Trends")
            monthly_data = analytics['monthly_trends']
            
            if monthly_data.get('cost_by_month'):
                months = list(monthly_data['cost_by_month'].keys())
                costs = list(monthly_data['cost_by_month'].values())
                
                fig = px.line(x=months, y=costs, title="Monthly Cost Trends",
                             labels={'x': 'Month', 'y': 'Total Cost ($)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying analytics: {str(e)}")

def show_financial_tab():
    """Display financial analytics"""
    st.subheader("💰 Financial Analytics")
    
    df = get_data()
    if df.empty:
        st.warning("No data available for financial analytics.")
        return
    
    try:
        analytics = calculate_comprehensive_analytics(df)
        cost_data = analytics.get('cost_analytics', {})
        
        if not cost_data:
            st.error("Cost analytics data not available.")
            return
        
        # Debug information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Data Summary:**")
            st.write(f"- Total rows in data: {len(df)}")
            st.write(f"- Status distribution: {df['status'].value_counts().to_dict()}")
            if 'Status_Canonical' in df.columns:
                st.write(f"- Canonical status distribution: {df['Status_Canonical'].value_counts().to_dict()}")
            
            st.write("**Probability Calculations:**")
            prob_data = analytics.get('probability_by_subject', {})
            for subject, prob in prob_data.items():
                st.write(f"- {subject}: {prob:.2%}")
            
            st.write("**Forecasting Data:**")
            forecast_data = cost_data.get('forecasted_cost_by_subject', {})
            for subject, forecast in forecast_data.items():
                st.write(f"- {subject}: ${forecast:,.2f}")
        
        # Financial overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Actual Cost", f"${cost_data.get('total_actual_cost', 0):,.2f}")
        with col2:
            st.metric("Total Forecasted Cost", f"${cost_data.get('total_forecasted_cost', 0):,.2f}")
        with col3:
            variance = cost_data.get('cost_variance', 0)
            st.metric("Cost Variance", f"${variance:,.2f}")
        with col4:
            variance_pct = cost_data.get('variance_percentage', 0)
            st.metric("Variance %", f"{variance_pct:.1f}%")
        
        # Top pilots by cost
        if cost_data.get('top_pilots_by_cost'):
            st.subheader("🏆 Top Pilots by Cost")
            pilots_df = pd.DataFrame(cost_data['top_pilots_by_cost'])
            st.dataframe(pilots_df.head(20), use_container_width=True)
            
            # Chart
            if len(pilots_df) >= 10:
                fig = px.bar(pilots_df.head(10), x='pilot', y='total_cost',
                            title="Top 10 Pilots by Total Cost")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Forecasted cost by subject
        if cost_data.get('forecasted_cost_by_subject'):
            st.subheader("🎯 Forecasted Cost by Subject")
            forecast_data = cost_data['forecasted_cost_by_subject']
            if any(v > 0 for v in forecast_data.values()):
                forecast_df = pd.DataFrame(list(forecast_data.items()), 
                                         columns=['Subject', 'Forecasted Cost'])
                forecast_df = forecast_df[forecast_df['Forecasted Cost'] > 0]
                forecast_df = forecast_df.sort_values('Forecasted Cost', ascending=False)
                
                st.dataframe(forecast_df, use_container_width=True)
                
                # Chart
                if len(forecast_df) >= 5:
                    fig = px.bar(forecast_df.head(10), x='Subject', y='Forecasted Cost',
                                title="Forecasted Cost by Subject")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No forecasted costs available (no open/review cases or zero probabilities)")
        
        # Cost by status
        if cost_data.get('cost_by_status'):
            st.subheader("📊 Cost Distribution by Status")
            status_costs = cost_data['cost_by_status']
            
            fig = px.pie(values=list(status_costs.values()), names=list(status_costs.keys()),
                        title="Cost Distribution by Status")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying financial analytics: {str(e)}")

def show_claims_details_tab():
    """Display detailed claims data"""
    st.subheader("📋 Claims Details")
    
    df = get_data()
    if df.empty:
        st.warning("No data available.")
        return
    
    # Display raw data with filters
    st.subheader("🔍 Filter Claims")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Filter by Status", 
                                   options=['All'] + list(df['status'].unique()))
    
    with col2:
        pilot_filter = st.selectbox("Filter by Pilot", 
                                  options=['All'] + list(df['pilot'].unique()))
    
    with col3:
        min_relief = st.number_input("Minimum Relief Amount", value=0.0, step=100.0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    
    if pilot_filter != 'All':
        filtered_df = filtered_df[filtered_df['pilot'] == pilot_filter]
    
    filtered_df = filtered_df[filtered_df['relief_dollars'] >= min_relief]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} claims")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Export functionality
    if st.button("Export Filtered Data to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sts_claims_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def scrape_sts_data():
    """Scrape STS data with comprehensive monitoring"""
    if st.session_state.collection_status['running']:
        st.warning("Data collection is already in progress.")
        return False
    
    # Initialize collection status
    st.session_state.collection_status = {
        'running': True,
        'current_step': 'Initializing...',
        'claims_found': 0,
        'pages_processed': 0,
        'start_time': datetime.now(),
        'export_file': None
    }
    
    # UI for monitoring
    st.subheader("📊 Data Collection Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        claims_metric = st.empty()
    with col2:
        pages_metric = st.empty()
    with col3:
        time_metric = st.empty()
    
    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        st.session_state.collection_status['current_step'] = 'Starting browser...'
        status_text.text("Starting browser...")
        
        driver = webdriver.Chrome(options=chrome_options)
        # Updated URL to match working script
        driver.get("https://sts2.alpa.org/adfs/ls/?wa=wsignin1.0&wtrealm=https%3a%2f%2fdal.alpa.org&wctx=rm%3d0%26id%3d65788f3a-90df-4c13-b375-f2e8ad524a11%26ru%3d%252fsts-admin&wct=2025-07-11T11%3a52%3a42Z&whr=https%3a%2f%2fdal.alpa.org&cfg=6")
        time.sleep(3)
        
        # Wait for login elements - updated IDs to match working script
        st.session_state.collection_status['current_step'] = 'Waiting for login page...'
        status_text.text("Waiting for login page...")
        
        wait = WebDriverWait(driver, 20)
        username_input = wait.until(EC.presence_of_element_located((By.ID, "userNameInput")))
        password_input = driver.find_element(By.ID, "passwordInput")
        login_button = driver.find_element(By.ID, "submitButton")
        
        # Login credentials - updated to match working script
        username_input.send_keys("N0000937")
        password_input.send_keys("STSD@L!42AlPa14")
        
        st.session_state.collection_status['current_step'] = 'Logging in...'
        status_text.text("Logging in...")
        
        login_button.click()
        time.sleep(5)
        
        # Navigate to claims page after login
        driver.get("https://dal.alpa.org/sts-admin/claims")
        
        # Wait for page to load and set up filters like in working script
        st.session_state.collection_status['current_step'] = 'Setting up filters...'
        status_text.text("Setting up filters...")
        
        # Click Sort & Filter button
        sort_filter_button = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/form/div[3]/div/div/div/div/div/div/main/div[1]/div[1]/div[2]/div[1]/div/div[2]/button[2]")))
        driver.execute_script("arguments[0].scrollIntoView(true);", sort_filter_button)
        sort_filter_button.click()
        time.sleep(2)
        
        # Click Add All button
        try:
            add_all_button = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/form/div[3]/div/div/div/div/div/div/main/div[1]/div[1]/div[2]/div[6]/div[1]/div[1]/div/div[3]/div/div/div[1]/fieldset/div[4]/button[1]")))
            driver.execute_script("arguments[0].scrollIntoView(true);", add_all_button)
            add_all_button.click()
            time.sleep(1)
        except Exception as e:
            st.warning(f"Could not click 'Add All': {e}")
        
        # Click Apply button
        try:
            apply_button = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/form/div[3]/div/div/div/div/div/div/main/div[1]/div[1]/div[2]/div[6]/div[1]/div[2]/div[2]/button[2]")))
            driver.execute_script("arguments[0].scrollIntoView(true);", apply_button)
            apply_button.click()
            time.sleep(2)
        except Exception as e:
            st.warning(f"Could not click 'Apply': {e}")
        
        time.sleep(5)
        
        # Set results per page to 50
        try:
            per_page_dropdown = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select")))
            driver.execute_script("arguments[0].scrollIntoView(true);", per_page_dropdown)
            if per_page_dropdown.get_attribute("value") != "50":
                per_page_dropdown.click()
                option_50 = per_page_dropdown.find_element(By.XPATH, ".//option[@value='50']")
                option_50.click()
                time.sleep(3)
        except Exception as e:
            st.warning(f"Could not set results per page to 50: {e}")
        
        # Wait for dashboard - updated to match new structure
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
        
        # Start collecting ticket URLs like in working script
        st.session_state.collection_status['current_step'] = 'Collecting ticket URLs...'
        status_text.text("Collecting ticket URLs...")
        
        all_tickets = []
        processed_tickets = set()
        
        # Collect data from all pages
        claims_data = []
        page_num = 1
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while True:
            try:
                st.session_state.collection_status['current_step'] = f'Processing page {page_num}...'
                st.session_state.collection_status['pages_processed'] = page_num
                status_text.text(f"Processing page {page_num}...")
                pages_metric.metric("Pages Processed", page_num)
                
                # Get table data with retry logic
                table = wait.until(EC.presence_of_element_located((By.ID, "claimsTable")))
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
                
                page_claims = 0
                for row in rows:
                    try:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 6:  # Minimum required columns
                            # More flexible column extraction
                            case_number = cells[0].text.strip() if len(cells) > 0 else ''
                            pilot = cells[1].text.strip() if len(cells) > 1 else ''
                            subject = cells[2].text.strip() if len(cells) > 2 else ''
                            status = cells[3].text.strip() if len(cells) > 3 else ''
                            submission_date = cells[4].text.strip() if len(cells) > 4 else ''
                            decision_date = cells[5].text.strip() if len(cells) > 5 else ''
                            
                            # Try to get relief minutes - more robust parsing
                            relief_minutes = 0
                            if len(cells) > 6:
                                try:
                                    relief_text = cells[6].text.strip()
                                    if relief_text and relief_text not in ['', '-', 'N/A']:
                                        relief_minutes = float(relief_text.replace(',', ''))
                                except (ValueError, AttributeError):
                                    relief_minutes = 0
                            
                            processing_days = cells[7].text.strip() if len(cells) > 7 else ''
                            
                            # Only include claims with positive relief
                            if relief_minutes > 0 and case_number:
                                claims_data.append({
                                    'case_number': case_number,
                                    'pilot': pilot,
                                    'subject': subject,
                                    'status': status,
                                    'submission_date': submission_date,
                                    'decision_date': decision_date,
                                    'relief_minutes': relief_minutes,
                                    'relief_dollars': relief_dollars(relief_minutes),
                                    'processing_days': processing_days
                                })
                                page_claims += 1
                    except Exception as e:
                        # Log individual row errors but continue
                        continue
                
                st.session_state.collection_status['claims_found'] = len(claims_data)
                claims_metric.metric("Claims Found", len(claims_data))
                
                # Dynamic progress calculation
                if page_num <= 5:
                    estimated_total_pages = 65  # Initial estimate
                else:
                    # Better estimate based on claims per page
                    avg_claims_per_page = len(claims_data) / page_num
                    if avg_claims_per_page > 0:
                        estimated_total_pages = min(max(50, page_num + 10), 100)  # Reasonable bounds
                    else:
                        estimated_total_pages = page_num + 5
                
                progress = min(page_num / estimated_total_pages, 0.95)  # Never show 100% until done
                progress_bar.progress(progress)
                
                # Reset error counter on successful page
                consecutive_errors = 0
                
                # Check for next page
                try:
                    next_button = driver.find_element(By.XPATH, "//a[contains(@class, 'ui-paginator-next')]")
                    if "ui-state-disabled" in next_button.get_attribute("class"):
                        break
                    next_button.click()
                    time.sleep(2)  # Increased wait time for page load
                    page_num += 1
                except Exception as nav_error:
                    # No more pages available
                    break
                    
            except Exception as page_error:
                consecutive_errors += 1
                st.warning(f"⚠️ Error on page {page_num}: {str(page_error)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    st.error(f"❌ Too many consecutive errors ({consecutive_errors}). Stopping collection.")
                    break
                else:
                    # Try to continue to next page
                    try:
                        page_num += 1
                        time.sleep(3)  # Wait longer before retry
                        continue
                    except:
                        break
        
        driver.quit()
        
        # Save data with validation
        if claims_data:
            df = pd.DataFrame(claims_data)
            
            # Data validation
            st.info(f"📊 **Collection Summary:**")
            st.write(f"- Total pages processed: {page_num}")
            st.write(f"- Total claims found: {len(claims_data)}")
            st.write(f"- Claims with relief > 0: {len(df[df['relief_minutes'] > 0])}")
            st.write(f"- Date range: {df['submission_date'].min()} to {df['submission_date'].max()}")
            st.write(f"- Status distribution: {df['status'].value_counts().head().to_dict()}")
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'sts_claims_data_{timestamp}.csv'
            df.to_csv(filename, index=False)
            
            # Store in session state
            st.session_state.collected_data = df
            st.session_state.data_collected = True
            st.session_state.collection_status['export_file'] = filename
            
            st.success(f"✅ Data collection completed successfully!")
            st.info(f"📁 Data saved to: {filename}")
        else:
            st.warning("⚠️ No claims data collected. This might indicate:")
            st.write("- All claims have 0 relief minutes")
            st.write("- Website structure has changed") 
            st.write("- Network or login issues")
            return False
        
        st.session_state.collection_status['running'] = False
        progress_bar.progress(1.0)
        
        # Final metrics
        elapsed_total = datetime.now() - st.session_state.collection_status['start_time']
        time_metric.metric("Total Time", f"{elapsed_total.seconds}s")
        
        return True
        
    except Exception as e:
        if 'driver' in locals():
            driver.quit()
        
        st.session_state.collection_status['running'] = False
        st.session_state.collection_status['current_step'] = f'Error: {str(e)}'
        status_text.text(f"Error: {str(e)}")
        st.error(f"❌ Data collection failed: {e}")
        
        return False

def main():
    """Main dashboard application"""
    
    # Check password first
    if not check_password():
        return
    
    # Header
    st.markdown('<h1 class="main-header">✈️ STS Claims Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Demo mode toggle
        demo_mode = st.toggle("Demo Mode", value=st.session_state.get('demo_mode', True))
        st.session_state.demo_mode = demo_mode
        
        if demo_mode:
            st.info("📊 **Demo Mode Active**\n\nUsing sample data for demonstration.")
        else:
            st.info("🔴 **Production Mode**\n\nReady for live data collection.")
        
        st.divider()
        
        # Data collection section (only in production mode)
        if not demo_mode:
            st.header("📥 Data Collection")
            
            if not st.session_state.get('data_collected', False):
                st.warning("⚠️ No data collected yet")
                if st.button("🚀 Start Data Collection", type="primary"):
                    scrape_sts_data()
            else:
                st.success("✅ Data collected successfully")
                df = st.session_state.get('collected_data', pd.DataFrame())
                st.metric("Claims Collected", len(df))
                
                if st.button("🔄 Refresh Data"):
                    scrape_sts_data()
        
        st.divider()
        
        # Data info
        df = get_data()
        if not df.empty:
            st.header("📊 Current Data")
            st.metric("Total Claims", len(df))
            # Use consistent column name
            relief_col = 'relief_dollars' if 'relief_dollars' in df.columns else 'Relief_Dollars'
            if relief_col in df.columns:
                st.metric("Total Relief Value", f"${df[relief_col].sum():,.2f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Analytics", "💰 Financial", "📋 Claims Details"])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_analytics_tab()
    
    with tab3:
        show_financial_tab()
    
    with tab4:
        show_claims_details_tab()

if __name__ == "__main__":
    main()
