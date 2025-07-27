import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import statistics
import re
import os
import platform
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

def minutes_to_hhmm(minutes):
    """Convert minutes to HH:MM format"""
    try:
        if pd.isna(minutes) or minutes == 0:
            return "00:00"
        minutes = int(float(minutes))
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    except:
        return "00:00"

def process_relief_data(df, relief_rate=320.47):
    """Process relief data to ensure both minutes/dollars and HH:MM format are available"""
    df = df.copy()
    
    # Handle various relief column names and formats
    relief_cols = ['relief_requested', 'relief_minutes', 'relief_hours', 'relief_time']
    relief_found = False
    
    for col in relief_cols:
        if col in df.columns:
            relief_found = True
            if col == 'relief_requested':
                # If relief_requested is in HH:MM format, convert to minutes
                df['relief_minutes'] = df[col].apply(hhmm_to_minutes)
            elif col == 'relief_hours':
                # Convert hours to minutes
                df['relief_minutes'] = df[col] * 60
            elif col == 'relief_time':
                # Assume it's in HH:MM format
                df['relief_minutes'] = df[col].apply(hhmm_to_minutes)
            break
    
    # If no relief column found, set default
    if not relief_found and 'relief_minutes' not in df.columns:
        df['relief_minutes'] = 60  # Default 1 hour
    
    # Ensure relief_minutes exists and is numeric
    if 'relief_minutes' not in df.columns:
        df['relief_minutes'] = 0
    
    # Convert relief_minutes to numeric, handling any non-numeric values
    df['relief_minutes'] = pd.to_numeric(df['relief_minutes'], errors='coerce').fillna(0)
    
    # Calculate relief_dollars if not present
    if 'relief_dollars' not in df.columns:
        df['relief_dollars'] = df['relief_minutes'].apply(lambda x: relief_dollars(x, relief_rate))
    
    # Add HH:MM format column
    df['relief_hhmm'] = df['relief_minutes'].apply(minutes_to_hhmm)
    
    return df

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

def extract_month(date_str):
    """Extract month from date string in YYYY-MM format"""
    if not date_str or pd.isna(date_str):
        return "Unknown"
    
    date_str = str(date_str).split()[0]  # Get just the date part
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m")
    except:
        try:
            dt = datetime.strptime(date_str, "%m/%d/%Y")
            return dt.strftime("%Y-%m")
        except:
            return "Unknown"

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
    
    # Top 20 pilots by number of cases submitted
    top_20_pilots_by_cases = pilot_counts.head(20).to_dict()
    
    # ===== TOP 20 PILOTS BY RELIEF REQUESTED (FROM ORIGINAL SCRIPT) =====
    # Overall top 20 by relief amount
    emp_relief_totals = df.groupby('pilot')['Relief_Dollars'].sum()
    top20_pilots_overall = emp_relief_totals.nlargest(20).to_dict()
    
    # Top 20 by relief amount per status
    top20_pilots_by_status = {}
    for status in all_statuses:
        status_data = df[df['Status_Canonical'] == status]
        if len(status_data) > 0:
            emp_relief_by_status = status_data.groupby('pilot')['Relief_Dollars'].sum()
            top20_pilots_by_status[status] = emp_relief_by_status.nlargest(20).to_dict()
        else:
            top20_pilots_by_status[status] = {}
    
    # Top 20 highest value claims
    top_20_claims = df.nlargest(20, 'Relief_Dollars')[['case_number', 'pilot', 'subject', 'Relief_Dollars', 'status']].to_dict('records')
    
    # Multiple case employees (pilots with more than one case)
    multiple_case_employees = {}
    for pilot, count in pilot_counts.items():
        if count > 1:
            pilot_cases = df[df['pilot'] == pilot]
            multiple_case_employees[pilot] = {
                'case_count': count,
                'total_relief': pilot_cases['Relief_Dollars'].sum(),
                'avg_relief': pilot_cases['Relief_Dollars'].mean(),
                'subjects': pilot_cases['Subject_Grouped'].unique().tolist(),
                'statuses': pilot_cases['Status_Canonical'].unique().tolist()
            }
    
    # ===== TOTAL RELIEF REQUESTED BY SUBJECT (FROM ORIGINAL SCRIPT) =====
    # Calculate total relief by subject group in hours and percentages
    subject_relief_totals = df.groupby('Subject_Grouped')['Relief_Dollars'].sum().to_dict()
    total_relief_all = df['Relief_Dollars'].sum()
    subject_relief_hours = {k: v / relief_rate for k, v in subject_relief_totals.items()}  # Convert to hours
    subject_relief_percentages = {k: (v / total_relief_all * 100) if total_relief_all > 0 else 0 for k, v in subject_relief_totals.items()}
    
    # Sort by relief amount
    subject_relief_sorted = sorted(subject_relief_hours.items(), key=lambda x: x[1], reverse=True)
    
    # ===== VIOLATION TYPE ANALYSIS (FROM ORIGINAL SCRIPT) =====
    violation_counter = df['subject'].value_counts().to_dict()
    total_cases = len(df)
    violation_percentages = {v: (count / total_cases) * 100 for v, count in violation_counter.items()}
    
    # ===== RECENT CASES ANALYSIS =====
    recent_cases = 0
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    for _, row in df.iterrows():
        last_interaction = row.get('submission_date', '')  # Use submission_date or last_interaction
        try:
            interaction_date = datetime.strptime(str(last_interaction).split()[0], "%Y-%m-%d")
            if interaction_date >= seven_days_ago:
                recent_cases += 1
        except:
            pass
    
    # ===== OLDEST INCIDENT CASES =====
    incident_cases = []
    for _, row in df.iterrows():
        try:
            date_str = str(row.get('submission_date', ''))
            if date_str and date_str != 'nan':
                date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
                incident_cases.append({
                    'date': date_obj,
                    'case_number': row.get('case_number', ''),
                    'pilot': row.get('pilot', '')
                })
        except:
            continue
    incident_cases.sort(key=lambda x: x['date'])
    oldest_5_cases = incident_cases[:5]
    
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
        'top_20_pilots_by_cases': top_20_pilots_by_cases,
        'multiple_case_employees': multiple_case_employees,
        'top_20_claims': top_20_claims,
        # New comprehensive analytics from original script
        'top20_pilots_overall': top20_pilots_overall,
        'top20_pilots_by_status': top20_pilots_by_status,
        'violation_counter': violation_counter,
        'violation_percentages': violation_percentages,
        'recent_cases': recent_cases,
        'oldest_5_cases': oldest_5_cases,
        'unique_violation_types': len(violation_counter),
        'top_10_pilots_by_cases': dict(pilot_counts.head(10)),
        # TOTAL RELIEF REQUESTED BY SUBJECT ANALYTICS (from original script)
        'subject_relief_totals': subject_relief_totals,
        'subject_relief_hours': subject_relief_hours,
        'subject_relief_percentages': subject_relief_percentages,
        'subject_relief_sorted': subject_relief_sorted,
        'total_relief_all_subjects': total_relief_all,
        # Summary statistics
        'total_claims': len(df),
        'total_relief': df['Relief_Dollars'].sum(),
        'avg_relief': df['Relief_Dollars'].mean(),
        'median_relief': df['Relief_Dollars'].median(),
        'unique_employees': len(pilot_counts),
        'employees_with_multiple_cases': len(multiple_case_employees),
        'percentage_multiple_cases': (sum(multiple_case_employees[emp]['case_count'] for emp in multiple_case_employees) / total_cases * 100) if total_cases > 0 else 0
    }

def calculate_cost_analytics(df, subject_stats, probability_by_subject, avg_relief_minutes_per_subject, relief_rate):
    """Calculate comprehensive cost analytics from original script with correct forecasting"""
    
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
    
    # Actual paid by violation type (raw)
    actual_paid_by_violation = {}
    if len(approved_claims) > 0:
        actual_paid_by_violation = approved_claims.groupby('subject')['Relief_Dollars'].sum().to_dict()
    
    # ===== CORRECTED FORECASTING LOGIC FROM ORIGINAL SCRIPT =====
    
    # Forecasted cost by subject - using correct calculation
    forecasted_cost_by_subject = {}
    for subject, stats in subject_stats.items():
        prob = probability_by_subject.get(subject, 0)
        unresolved = stats.get("open_count", 0) + stats.get("in_review_count", 0)
        avg_relief_dollars = avg_relief_minutes_per_subject.get(subject, 0)  # This is already in dollars
        forecasted_cost_by_subject[subject] = prob * unresolved * avg_relief_dollars
    
    # Forecasted by pilot - matching original script logic
    pilot_open_counts = {}
    pilot_in_review_counts = {}
    pilot_avg_relief = {}
    pilot_prob = {}
    pilot_subjects = {}
    
    # Group by pilot to get their unresolved cases and subjects
    for pilot in df['pilot'].unique():
        pilot_data = df[df['pilot'] == pilot]
        pilot_open_counts[pilot] = len(pilot_data[pilot_data['Status_Canonical'] == 'open'])
        pilot_in_review_counts[pilot] = len(pilot_data[pilot_data['Status_Canonical'] == 'in review'])
        
        # Get all subjects for this pilot
        subjects = pilot_data['Subject_Grouped'].tolist()
        pilot_subjects[pilot] = subjects
        
        # Calculate average probability and relief for this pilot based on their subjects
        if subjects:
            pilot_prob[pilot] = sum(probability_by_subject.get(s, 0) for s in subjects) / len(subjects)
            pilot_avg_relief[pilot] = sum(avg_relief_minutes_per_subject.get(s, 0) for s in subjects) / len(subjects)
        else:
            pilot_prob[pilot] = 0
            pilot_avg_relief[pilot] = 0
    
    # Calculate forecasted cost by pilot
    forecasted_cost_by_pilot = {}
    for pilot in pilot_subjects.keys():
        unresolved = pilot_open_counts.get(pilot, 0) + pilot_in_review_counts.get(pilot, 0)
        forecasted_cost_by_pilot[pilot] = pilot_prob[pilot] * unresolved * pilot_avg_relief[pilot]
    
    top20_forecasted_by_pilot = sorted(forecasted_cost_by_pilot.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Forecasted by violation type (raw)
    violation_prob = {}
    violation_avg_relief = {}
    violation_open = {}
    violation_in_review = {}
    
    for violation in df['subject'].unique():
        # Map violation to subject group to get probability
        subject_group = group_subject_key(violation)
        violation_prob[violation] = probability_by_subject.get(subject_group, 0)
        violation_avg_relief[violation] = avg_relief_minutes_per_subject.get(subject_group, 0)
        
        violation_data = df[df['subject'] == violation]
        violation_open[violation] = len(violation_data[violation_data['Status_Canonical'] == 'open'])
        violation_in_review[violation] = len(violation_data[violation_data['Status_Canonical'] == 'in review'])
    
    forecasted_cost_by_violation = {}
    for violation in violation_prob:
        unresolved = violation_open[violation] + violation_in_review[violation]
        forecasted_cost_by_violation[violation] = violation_prob[violation] * unresolved * violation_avg_relief[violation]
    
    # Forecasted by month - using incident date
    month_prob = {}
    month_avg_relief = {}
    month_open = {}
    month_in_review = {}
    
    for _, row in df.iterrows():
        # Extract month from incident date or submission date
        month = extract_month(row.get('submission_date', ''))
        if month == "Unknown":
            continue
            
        subject = row['Subject_Grouped']
        if month not in month_prob:
            month_prob[month] = probability_by_subject.get(subject, 0)
            month_avg_relief[month] = avg_relief_minutes_per_subject.get(subject, 0)
            month_open[month] = 0
            month_in_review[month] = 0
        
        if row['Status_Canonical'] == 'open':
            month_open[month] += 1
        elif row['Status_Canonical'] == 'in review':
            month_in_review[month] += 1
    
    forecasted_cost_by_month = {}
    for month in month_prob:
        unresolved = month_open[month] + month_in_review[month]
        forecasted_cost_by_month[month] = month_prob[month] * unresolved * month_avg_relief[month]
    
    # ===== AGING FORECAST =====
    now = datetime.now()
    aging_buckets = ["0-30", "31-60", "61-90", "91+"]
    aging_forecast = {bucket: 0.0 for bucket in aging_buckets}
    
    for _, row in df.iterrows():
        if row['Status_Canonical'] not in ['open', 'in review']:
            continue
        
        try:
            # Use submission date for aging calculation
            created = datetime.strptime(str(row.get('submission_date', '')).split()[0], "%Y-%m-%d")
            age = (now - created).days
        except:
            age = 0
        
        # Determine age bucket
        if age <= 30:
            bucket = "0-30"
        elif age <= 60:
            bucket = "31-60"
        elif age <= 90:
            bucket = "61-90"
        else:
            bucket = "91+"
        
        # Calculate forecasted cost for this case
        subject = row['Subject_Grouped']
        prob = probability_by_subject.get(subject, 0)
        avg_relief = avg_relief_minutes_per_subject.get(subject, 0)
        cost = prob * avg_relief
        aging_forecast[bucket] += cost
    
    # Status costs
    status_costs = df.groupby('Status_Canonical')['Relief_Dollars'].sum().to_dict()
    status_avg_costs = df.groupby('Status_Canonical')['Relief_Dollars'].mean().to_dict()
    
    # Outlier cases (high cost) - using statistical outlier detection
    reliefs = df[df['Relief_Dollars'] > 0]['Relief_Dollars'].tolist()
    if reliefs:
        mean_relief = sum(reliefs) / len(reliefs)
        std_relief = (sum((x - mean_relief) ** 2 for x in reliefs) / len(reliefs)) ** 0.5
        outlier_threshold = mean_relief + 2 * std_relief
        outlier_cases = df[df['Relief_Dollars'] > outlier_threshold].to_dict('records')
    else:
        outlier_cases = []
    
    # Cost statistics
    paid_costs = approved_claims['Relief_Dollars'].tolist() if len(approved_claims) > 0 else [0]
    median_paid_cost = statistics.median(paid_costs) if paid_costs else 0
    mean_paid_cost = statistics.mean(paid_costs) if paid_costs else 0
    
    forecasted_costs_list = [v for v in forecasted_cost_by_subject.values() if v > 0]
    median_forecasted_cost = statistics.median(forecasted_costs_list) if forecasted_costs_list else 0
    mean_forecasted_cost = statistics.mean(forecasted_costs_list) if forecasted_costs_list else 0
    
    total_forecasted_cost = sum(forecasted_cost_by_subject.values())
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
        'status_costs': status_costs,
        'status_avg_costs': status_avg_costs,
        'forecasted_cost_by_subject': forecasted_cost_by_subject,
        'forecasted_cost_by_pilot': forecasted_cost_by_pilot,
        'forecasted_cost_by_violation': forecasted_cost_by_violation,
        'forecasted_cost_by_month': forecasted_cost_by_month,
        'aging_forecast': aging_forecast,
        'actual_paid_by_pilot': actual_paid_by_pilot,
        'actual_paid_by_subject': actual_paid_by_subject,
        'actual_paid_by_violation': actual_paid_by_violation,
        'total_actual_paid_cost': total_actual_paid_cost,
        'num_approved_cases': num_approved_cases,
        'avg_paid_per_case': avg_paid_per_case,
        'top20_actual_paid_by_pilot': top20_actual_paid_by_pilot,
        'top20_forecasted_by_pilot': top20_forecasted_by_pilot,
        'outlier_cases': outlier_cases,
        'median_paid_cost': median_paid_cost,
        'mean_paid_cost': mean_paid_cost,
        'median_forecasted_cost': median_forecasted_cost,
        'mean_forecasted_cost': mean_forecasted_cost,
        'cost_statistics': {
            'mean_cost': mean_paid_cost,
            'median_cost': median_paid_cost,
            'min_cost': df['Relief_Dollars'].min() if len(df) > 0 else 0,
            'max_cost': df['Relief_Dollars'].max() if len(df) > 0 else 0,
            'std_cost': df['Relief_Dollars'].std() if len(df) > 0 else 0,
            'cost_variance_stat': df['Relief_Dollars'].var() if len(df) > 0 else 0
        }
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
        relief_rate = 320.47  # Use default rate for demo data
        relief_minutes = (relief_amount / relief_rate) * 60  # Convert back to minutes
        
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
            'relief_minutes': relief_minutes,
            'relief_dollars': relief_amount,
            'relief_hhmm': minutes_to_hhmm(relief_minutes),
            'submission_date': submission_date.strftime('%Y-%m-%d'),
            'decision_date': decision_date.strftime('%Y-%m-%d') if decision_date else '',
            'processing_days': (decision_date - submission_date).days if decision_date else None,
            'violation_type': np.random.choice(['Type A', 'Type B', 'Type C', 'Type D']),
            'probability_of_payment': np.random.uniform(0.1, 0.9)
        })
    
    return pd.DataFrame(data)

def get_data():
    """Get the current data from session state, uploaded data, or demo data"""
    relief_rate = st.session_state.get('relief_rate', 320.47)
    
    if st.session_state.get('demo_mode', True):
        return generate_demo_data()
    else:
        # Check for uploaded data first
        if st.session_state.get('data_source') == 'uploaded' and 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data'].copy()
            return process_relief_data(df, relief_rate)
        
        # Check for manual data
        elif st.session_state.get('data_source') == 'manual' and 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data'].copy()
            return process_relief_data(df, relief_rate)
        
        # Check for collected data (from web scraping)
        elif st.session_state.get('data_collected', False) and 'collected_data' in st.session_state:
            df = st.session_state['collected_data'].copy()
            return process_relief_data(df, relief_rate)
        
        # Try to load from saved files
        elif os.path.exists('uploaded_claims_data.csv'):
            try:
                df = pd.read_csv('uploaded_claims_data.csv')
                df = process_relief_data(df, relief_rate)
                st.session_state['uploaded_data'] = df
                st.session_state['data_source'] = 'uploaded'
                return df
            except:
                pass
        
        elif os.path.exists('manual_claims_data.csv'):
            try:
                df = pd.read_csv('manual_claims_data.csv')
                df = process_relief_data(df, relief_rate)
                st.session_state['uploaded_data'] = df
                st.session_state['data_source'] = 'manual'
                return df
            except:
                pass
        
        # Return empty DataFrame if no data available
        return pd.DataFrame()

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
        # Get relief rate from session state
        relief_rate = st.session_state.get('relief_rate', 320.47)
        analytics = calculate_comprehensive_analytics(df, relief_rate)
        
        # Configuration info
        st.info(f"💰 **Current Relief Rate:** ${relief_rate:.2f}/hour")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate total relief hours for display
        total_relief_hours = analytics['total_relief'] / relief_rate
        avg_relief_hours = analytics['avg_relief'] / relief_rate
        median_relief_hours = analytics['median_relief'] / relief_rate
        
        with col1:
            st.metric("Total Claims", analytics['total_claims'])
        with col2:
            st.metric("Total Relief Value", f"${analytics['total_relief']:,.2f}", f"{total_relief_hours:.1f} hours")
        with col3:
            st.metric("Average Relief", f"${analytics['avg_relief']:,.2f}", f"{minutes_to_hhmm(avg_relief_hours * 60)}")
        with col4:
            st.metric("Median Relief", f"${analytics['median_relief']:,.2f}", f"{minutes_to_hhmm(median_relief_hours * 60)}")
        
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
        
        # Top 20 Pilots by Cases Submitted
        if analytics['top_20_pilots_by_cases']:
            st.subheader("🏆 Top 20 Pilots by Cases Submitted")
            top_pilots_df = pd.DataFrame(list(analytics['top_20_pilots_by_cases'].items()), 
                                       columns=['Pilot', 'Cases Submitted'])
            top_pilots_df = top_pilots_df.sort_values('Cases Submitted', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(top_pilots_df, use_container_width=True, height=400)
            with col2:
                fig = px.bar(top_pilots_df.head(15), x='Cases Submitted', y='Pilot',
                           title="Top 15 Pilots by Cases Submitted",
                           orientation='h')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Multiple Case Employees (Enhanced)
        if analytics['multiple_case_employees']:
            st.subheader("👥 Multiple Case Employees (Detailed)")
            
            # Create detailed dataframe
            multi_employees_data = []
            for pilot, details in analytics['multiple_case_employees'].items():
                multi_employees_data.append({
                    'Pilot': pilot,
                    'Cases': details['case_count'],
                    'Total Relief ($)': f"${details['total_relief']:,.2f}",
                    'Avg Relief ($)': f"${details['avg_relief']:,.2f}",
                    'Subjects': ', '.join(details['subjects'][:3]) + ('...' if len(details['subjects']) > 3 else ''),
                    'Statuses': ', '.join(details['statuses'])
                })
            
            multi_df = pd.DataFrame(multi_employees_data)
            multi_df = multi_df.sort_values('Cases', ascending=False)
            
            # Display table and chart
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(multi_df, use_container_width=True, height=300)
            with col2:
                # Chart showing case distribution
                case_dist = pd.DataFrame(list(analytics['multiple_case_employees'].items()))
                case_dist.columns = ['Pilot', 'Details']
                case_dist['Cases'] = case_dist['Details'].apply(lambda x: x['case_count'])
                case_dist['Relief'] = case_dist['Details'].apply(lambda x: x['total_relief'])
                
                fig = px.scatter(case_dist, x='Cases', y='Relief', 
                               hover_data=['Pilot'],
                               title="Cases vs Relief Amount",
                               labels={'Cases': 'Number of Cases', 'Relief': 'Total Relief ($)'})
                st.plotly_chart(fig, use_container_width=True)
        
        # ===== TOTAL RELIEF REQUESTED BY SUBJECT OVERVIEW =====
        st.subheader("💰 Total Relief Requested by Subject - Overview")
        if analytics.get('subject_relief_sorted'):
            # Show top 10 subjects in overview
            top_subjects_data = []
            for subject, hours in analytics['subject_relief_sorted'][:10]:  # Top 10
                percentage = analytics['subject_relief_percentages'].get(subject, 0)
                dollar_value = analytics['subject_relief_totals'].get(subject, 0)
                minutes = hours * 60  # Convert hours back to minutes for HH:MM display
                top_subjects_data.append({
                    'Subject': subject,
                    'Hours': round(hours, 2),
                    'HH:MM': minutes_to_hhmm(minutes),
                    'Dollar Value': dollar_value,
                    '% of Total': round(percentage, 2)
                })
            
            subjects_df = pd.DataFrame(top_subjects_data)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                # Format for display
                display_df = subjects_df.copy()
                display_df['Dollar Value'] = display_df['Dollar Value'].apply(lambda x: f"${x:,.2f}")
                display_df['% of Total'] = display_df['% of Total'].apply(lambda x: f"{x}%")
                st.dataframe(display_df, use_container_width=True)
            with col2:
                # Pie chart of top 8 subjects
                fig = px.pie(subjects_df.head(8), values='Hours', names='Subject',
                           title="Relief Hours Distribution (Top 8 Subjects)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Total relief summary
            total_relief_hours = sum(analytics['subject_relief_hours'].values())
            total_relief_value = analytics.get('total_relief_all_subjects', 0)
            total_relief_minutes = total_relief_hours * 60
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🕒 **Total Relief Hours**", f"{total_relief_hours:,.1f} hours")
            with col2:
                st.metric("⏱️ **Total Relief Time**", f"{minutes_to_hhmm(total_relief_minutes)}")
            with col3:
                st.metric("💰 **Total Relief Value**", f"${total_relief_value:,.2f}")
        
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
        # Get relief rate from session state
        relief_rate = st.session_state.get('relief_rate', 320.47)
        analytics = calculate_comprehensive_analytics(df, relief_rate)
        
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
        
        # ===== NEW: TOP 20 PILOTS BY RELIEF AMOUNT (FROM ORIGINAL SCRIPT) =====
        st.subheader("🥇 Top 20 Pilots by Relief Amount - Overall")
        if analytics.get('top20_pilots_overall'):
            top_pilots_relief_df = pd.DataFrame(list(analytics['top20_pilots_overall'].items()), 
                                               columns=['Pilot', 'Total Relief ($)'])
            top_pilots_relief_df = top_pilots_relief_df.sort_values('Total Relief ($)', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(top_pilots_relief_df, use_container_width=True)
            with col2:
                fig = px.bar(top_pilots_relief_df.head(10), x='Total Relief ($)', y='Pilot',
                           title="Top 10 Pilots by Relief Amount", orientation='h')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Top 20 Pilots by Relief Amount per Status
        if analytics.get('top20_pilots_by_status'):
            st.subheader("🏅 Top 20 Pilots by Relief Amount - By Status")
            
            # Create tabs for each status
            status_tabs = st.tabs([status.title() for status in analytics['top20_pilots_by_status'].keys()])
            
            for i, (status, status_data) in enumerate(analytics['top20_pilots_by_status'].items()):
                with status_tabs[i]:
                    if status_data:
                        status_df = pd.DataFrame(list(status_data.items()), 
                                               columns=['Pilot', f'Relief ({status.title()})'])
                        status_df = status_df.sort_values(f'Relief ({status.title()})', ascending=False)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.dataframe(status_df, use_container_width=True)
                        with col2:
                            if len(status_df) > 0:
                                fig = px.bar(status_df.head(10), 
                                           x=f'Relief ({status.title()})', y='Pilot',
                                           title=f"Top 10 Pilots - {status.title()}", 
                                           orientation='h')
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {status} status")
        
        # ===== TOTAL RELIEF REQUESTED BY SUBJECT (FROM ORIGINAL SCRIPT) =====
        st.subheader("💰 Total Relief Requested by Subject (Hours, HH:MM & Percentages)")
        if analytics.get('subject_relief_sorted'):
            relief_subject_data = []
            for subject, hours in analytics['subject_relief_sorted']:
                percentage = analytics['subject_relief_percentages'].get(subject, 0)
                minutes = hours * 60  # Convert hours to minutes for HH:MM display
                relief_subject_data.append({
                    'Subject': subject,
                    'Hours': round(hours, 2),
                    'HH:MM': minutes_to_hhmm(minutes),
                    'Percentage of Total': f"{percentage:.2f}%",
                    'Dollar Value': f"${analytics['subject_relief_totals'].get(subject, 0):,.2f}"
                })
            
            relief_subject_df = pd.DataFrame(relief_subject_data)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(relief_subject_df, use_container_width=True)
            with col2:
                # Chart showing top subjects by relief hours
                top_subjects = relief_subject_df.head(8)
                fig = px.pie(top_subjects, values='Hours', names='Subject',
                           title="Relief Distribution by Subject (Top 8)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Total relief metric
            total_relief_hours = sum(analytics['subject_relief_hours'].values())
            st.metric("**Total Relief Requested (All Subjects)**", f"{total_relief_hours:,.1f} hours", 
                     f"${analytics.get('total_relief_all_subjects', 0):,.2f}")
        
        # ===== VIOLATION TYPE ANALYSIS =====
        st.subheader("⚖️ Violation Type Analysis")
        if analytics.get('violation_counter'):
            violation_df = pd.DataFrame([
                {'Violation': violation, 'Count': count, 'Percentage': f"{analytics['violation_percentages'][violation]:.2f}%"}
                for violation, count in analytics['violation_counter'].items()
            ])
            violation_df = violation_df.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(violation_df, use_container_width=True, height=400)
            with col2:
                st.metric("Total Unique Violation Types", analytics.get('unique_violation_types', 0))
                st.metric("Recent Cases (Last 7 Days)", analytics.get('recent_cases', 0))
                
                # Top violations chart
                top_violations = violation_df.head(8)
                fig = px.bar(top_violations, x='Count', y='Violation',
                           title="Top 8 Violation Types", orientation='h')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # ===== OLDEST CASES =====
        if analytics.get('oldest_5_cases'):
            st.subheader("📅 5 Oldest Cases")
            oldest_df = pd.DataFrame(analytics['oldest_5_cases'])
            oldest_df['date'] = oldest_df['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(oldest_df, use_container_width=True)
        
        # ===== EMPLOYEE STATISTICS =====
        st.subheader("👨‍✈️ Employee Statistics Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Unique Employees", analytics.get('unique_employees', 0))
        with col2:
            st.metric("Employees with Multiple Cases", analytics.get('employees_with_multiple_cases', 0))
        with col3:
            st.metric("% Cases from Multi-Case Employees", f"{analytics.get('percentage_multiple_cases', 0):.1f}%")
        with col4:
            st.metric("Total Cases", analytics.get('total_claims', 0))
        
        # ===== TOP 10 PILOTS BY CASE COUNT =====
        if analytics.get('top_10_pilots_by_cases'):
            st.subheader("🔟 Top 10 Pilots by Number of Cases Submitted")
            top10_pilots_df = pd.DataFrame([
                {'Rank': i+1, 'Pilot Employee #': pilot, 'Case Count': count}
                for i, (pilot, count) in enumerate(analytics['top_10_pilots_by_cases'].items())
            ])
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(top10_pilots_df, use_container_width=True)
            with col2:
                fig = px.bar(top10_pilots_df, x='Pilot Employee #', y='Case Count',
                           title="Top 10 Pilots by Case Count")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Subject analysis - Enhanced to match original script
        st.subheader("📋 Complete Subject Violation Breakdown (All Statuses)")
        if analytics['subject_stats']:
            # Get all statuses for column headers
            all_statuses = analytics.get('all_statuses', [])
            
            # Create comprehensive subject breakdown
            subject_summary = []
            for subject, stats in analytics['subject_stats'].items():
                row = {
                    'Subject': subject,
                    'Total Cases': stats['count'],
                    'Total Relief ($)': stats['minutes'],
                    '% of Total Cases': round((stats['count'] / analytics.get('total_claims', 1)) * 100, 2)
                }
                
                # Add counts for each status
                for status in all_statuses:
                    safe_status = status.replace(' ', '_').replace('-', '_')
                    status_count = stats.get(f"{safe_status}_count", 0)
                    row[f"{status.title()} Cases"] = status_count
                
                # Add probability and approval metrics
                approved = stats.get('approved_count', 0)
                denied = stats.get('denied_count', 0)
                total_decided = approved + denied
                approval_rate = (approved / total_decided * 100) if total_decided > 0 else 0
                row['Approval Rate (%)'] = round(approval_rate, 1)
                
                subject_summary.append(row)
            
            subject_df = pd.DataFrame(subject_summary)
            subject_df = subject_df.sort_values('Total Relief ($)', ascending=False)
            
            # Display comprehensive table
            st.dataframe(subject_df, use_container_width=True, height=400)
            
            # Create two visualizations side by side
            col1, col2 = st.columns(2)
            
            with col1:
                # Subject distribution by relief value chart
                fig1 = px.bar(subject_df.head(10), x='Subject', y='Total Relief ($)',
                            title="Top 10 Subjects by Total Relief Value")
                fig1.update_xaxes(tickangle=45)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Subject distribution by case count
                fig2 = px.bar(subject_df.head(10), x='Subject', y='Total Cases',
                            title="Top 10 Subjects by Case Count")
                fig2.update_xaxes(tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)
        
        # ===== PROBABILITY OF PAYMENT BY SUBJECT (FROM ORIGINAL SCRIPT) =====
        st.subheader("📊 Probability of Payment by Subject")
        if analytics.get('probability_by_subject'):
            prob_data = []
            for subject, probability in analytics['probability_by_subject'].items():
                # Get the stats for this subject to show context
                subject_stats = analytics['subject_stats'].get(subject, {})
                approved = subject_stats.get('approved_count', 0)
                denied = subject_stats.get('denied_count', 0)
                total_decided = approved + denied
                
                prob_data.append({
                    'Subject': subject,
                    'Probability of Payment': f"{probability:.2%}",
                    'Probability (Decimal)': probability,
                    'Approved Cases': approved,
                    'Denied Cases': denied,
                    'Total Decided Cases': total_decided,
                    'Estimated Future Approvals': round(probability * (subject_stats.get('open_count', 0) + subject_stats.get('in_review_count', 0)), 2)
                })
            
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values('Probability (Decimal)', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                # Display probability table (exclude decimal column for clean display)
                display_prob_df = prob_df.drop('Probability (Decimal)', axis=1)
                st.dataframe(display_prob_df, use_container_width=True, height=400)
            
            with col2:
                # Probability visualization
                fig = px.bar(prob_df.head(10), x='Probability (Decimal)', y='Subject',
                           title="Probability of Payment by Subject (Top 10)", orientation='h',
                           hover_data=['Approved Cases', 'Denied Cases'])
                fig.update_layout(height=400)
                fig.update_xaxes(title="Probability of Payment", tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics for probabilities
            avg_probability = prob_df['Probability (Decimal)'].mean()
            median_probability = prob_df['Probability (Decimal)'].median()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Probability", f"{avg_probability:.1%}")
            with col2:
                st.metric("Median Probability", f"{median_probability:.1%}")
            with col3:
                highest_prob_subject = prob_df.iloc[0]['Subject'] if len(prob_df) > 0 else "N/A"
                highest_prob_value = prob_df.iloc[0]['Probability (Decimal)'] if len(prob_df) > 0 else 0
                st.metric("Highest Probability Subject", f"{highest_prob_subject}", f"{highest_prob_value:.1%}")
        
        # ===== ESTIMATED APPROVALS FROM OPEN & IN REVIEW CASES =====
        st.subheader("🔮 Estimated Future Approvals (Based on Probability)")
        if analytics.get('probability_by_subject') and analytics.get('subject_stats'):
            future_approvals = []
            total_estimated = 0
            
            for subject, probability in analytics['probability_by_subject'].items():
                subject_stats = analytics['subject_stats'].get(subject, {})
                open_cases = subject_stats.get('open_count', 0)
                in_review_cases = subject_stats.get('in_review_count', 0)
                unresolved = open_cases + in_review_cases
                estimated = probability * unresolved
                total_estimated += estimated
                
                if unresolved > 0:  # Only show subjects with unresolved cases
                    future_approvals.append({
                        'Subject': subject,
                        'Open Cases': open_cases,
                        'In Review Cases': in_review_cases,
                        'Total Unresolved': unresolved,
                        'Probability': f"{probability:.1%}",
                        'Estimated Approvals': round(estimated, 2)
                    })
            
            if future_approvals:
                future_df = pd.DataFrame(future_approvals)
                future_df = future_df.sort_values('Estimated Approvals', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(future_df, use_container_width=True)
                with col2:
                    st.metric("**Total Estimated Future Approvals**", f"{total_estimated:.1f} cases")
                    
                    # Chart of top estimated approvals
                    if len(future_df) > 0:
                        fig = px.bar(future_df.head(8), x='Estimated Approvals', y='Subject',
                                   title="Top 8 Subjects by Estimated Future Approvals", orientation='h')
                        fig.update_layout(height=300)
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
        
        # Top 20 Pilots by Cases Submitted (Detailed View)
        if analytics.get('top_20_pilots_by_cases'):
            st.subheader("🏆 Top 20 Pilots by Cases Submitted (Detailed)")
            
            # Create enhanced pilot statistics
            pilot_detailed = []
            for pilot, case_count in analytics['top_20_pilots_by_cases'].items():
                pilot_data = df[df['pilot'] == pilot]
                pilot_detailed.append({
                    'Pilot': pilot,
                    'Total Cases': case_count,
                    'Total Relief ($)': pilot_data['Relief_Dollars'].sum(),
                    'Avg Relief ($)': pilot_data['Relief_Dollars'].mean(),
                    'Approved Cases': len(pilot_data[pilot_data['Status_Canonical'] == 'approved']),
                    'Denied Cases': len(pilot_data[pilot_data['Status_Canonical'] == 'denied']),
                    'Open Cases': len(pilot_data[pilot_data['Status_Canonical'] == 'open']),
                    'Approval Rate (%)': round(
                        len(pilot_data[pilot_data['Status_Canonical'] == 'approved']) / 
                        max(len(pilot_data[pilot_data['Status_Canonical'].isin(['approved', 'denied'])]), 1) * 100, 1
                    )
                })
            
            pilot_detailed_df = pd.DataFrame(pilot_detailed)
            pilot_detailed_df = pilot_detailed_df.sort_values('Total Cases', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(pilot_detailed_df, use_container_width=True, height=400)
            with col2:
                # Top 10 pilots by cases chart
                fig = px.bar(pilot_detailed_df.head(10), x='Total Cases', y='Pilot',
                           title="Top 10 Pilots by Cases", orientation='h')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Multiple Case Employees Analysis
        if analytics.get('multiple_case_employees'):
            st.subheader("👥 Multiple Case Employees - Deep Dive")
            
            multi_employees = analytics['multiple_case_employees']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pilots with Multiple Cases", len(multi_employees))
            with col2:
                total_multi_cases = sum(details['case_count'] for details in multi_employees.values())
                st.metric("Total Cases (Multi-Case Pilots)", total_multi_cases)
            with col3:
                total_multi_relief = sum(details['total_relief'] for details in multi_employees.values())
                st.metric("Total Relief (Multi-Case)", f"${total_multi_relief:,.0f}")
            with col4:
                avg_cases_per_pilot = total_multi_cases / len(multi_employees) if multi_employees else 0
                st.metric("Avg Cases per Multi-Case Pilot", f"{avg_cases_per_pilot:.1f}")
            
            # Detailed breakdown
            multi_detail = []
            for pilot, details in multi_employees.items():
                multi_detail.append({
                    'Pilot': pilot,
                    'Number of Cases': details['case_count'],
                    'Total Relief ($)': details['total_relief'],
                    'Average Relief ($)': details['avg_relief'],
                    'Subject Categories': len(details['subjects']),
                    'Status Types': len(details['statuses']),
                    'Primary Subjects': ', '.join(details['subjects'][:2]),
                    'Status Distribution': ', '.join(details['statuses'])
                })
            
            multi_detail_df = pd.DataFrame(multi_detail)
            multi_detail_df = multi_detail_df.sort_values('Number of Cases', ascending=False)
            
            # Display detailed table
            st.dataframe(multi_detail_df, use_container_width=True)
            
            # Analysis charts
            col1, col2 = st.columns(2)
            with col1:
                # Distribution of case counts
                case_counts = [details['case_count'] for details in multi_employees.values()]
                fig = px.histogram(x=case_counts, title="Distribution of Case Counts (Multi-Case Pilots)",
                                 labels={'x': 'Number of Cases', 'y': 'Number of Pilots'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Relief vs case count scatter
                relief_data = [(details['case_count'], details['total_relief']) 
                              for details in multi_employees.values()]
                if relief_data:
                    cases, relief = zip(*relief_data)
                    fig = px.scatter(x=cases, y=relief, 
                                   title="Cases vs Total Relief (Multi-Case Pilots)",
                                   labels={'x': 'Number of Cases', 'y': 'Total Relief ($)'})
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
        # Get relief rate from session state
        relief_rate = st.session_state.get('relief_rate', 320.47)
        analytics = calculate_comprehensive_analytics(df, relief_rate)
        cost_data = analytics.get('cost_analytics', {})
        
        # Display current relief rate being used
        st.info(f"💰 **Using Relief Rate:** ${relief_rate:.2f}/hour for all cost calculations")
        
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
        
        # ===== TOTAL RELIEF REQUESTED BY SUBJECT - FINANCIAL VIEW =====
        st.subheader("💰 Total Relief Requested by Subject (Financial Breakdown)")
        if analytics.get('subject_relief_sorted'):
            relief_financial_data = []
            for subject, hours in analytics['subject_relief_sorted']:
                dollar_value = analytics['subject_relief_totals'].get(subject, 0)
                percentage = analytics['subject_relief_percentages'].get(subject, 0)
                minutes = hours * 60  # Convert hours to minutes for HH:MM display
                relief_financial_data.append({
                    'Subject': subject,
                    'Relief Hours': round(hours, 2),
                    'Relief HH:MM': minutes_to_hhmm(minutes),
                    'Dollar Value': dollar_value,
                    'Percentage of Total Relief': percentage
                })
            
            relief_financial_df = pd.DataFrame(relief_financial_data)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Format the DataFrame for better display
                display_df = relief_financial_df.copy()
                display_df['Dollar Value'] = display_df['Dollar Value'].apply(lambda x: f"${x:,.2f}")
                display_df['Percentage of Total Relief'] = display_df['Percentage of Total Relief'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(display_df, use_container_width=True)
            with col2:
                # Top 8 subjects by dollar value chart
                top_subjects = relief_financial_df.head(8)
                fig = px.bar(top_subjects, x='Dollar Value', y='Subject',
                           title="Top 8 Subjects by Relief Value", orientation='h')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics for relief by subject
            total_relief_value = analytics.get('total_relief_all_subjects', 0)
            total_relief_hours = sum(analytics['subject_relief_hours'].values())
            total_relief_minutes = total_relief_hours * 60
            st.metrics_container = st.container()
            with st.metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Relief Value", f"${total_relief_value:,.2f}")
                with col2:
                    st.metric("Total Relief Hours", f"{total_relief_hours:,.1f} hrs")
                with col3:
                    st.metric("Total Relief Time", f"{minutes_to_hhmm(total_relief_minutes)}")
                with col4:
                    st.metric("Relief Rate", f"${relief_rate:.2f}/hr")
        
        # Top pilots by cost
        if cost_data.get('top_pilots_by_cost'):
            st.subheader("🏆 Top Pilots by Cost")
            pilots_df = pd.DataFrame(cost_data['top_pilots_by_cost'])
            
            # Format the total_cost column for display
            if 'total_cost' in pilots_df.columns:
                pilots_df_display = pilots_df.copy()
                pilots_df_display['total_cost'] = pilots_df_display['total_cost'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
                st.dataframe(pilots_df_display.head(20), use_container_width=True)
            else:
                st.dataframe(pilots_df.head(20), use_container_width=True)
            
            # Chart (use original numeric values)
            if len(pilots_df) >= 10:
                fig = px.bar(pilots_df.head(10), x='pilot', y='total_cost',
                            title="Top 10 Pilots by Total Cost")
                fig.update_xaxes(tickangle=45)
                # Format y-axis to show currency
                fig.update_yaxes(tickformat="$,.0f")
                st.plotly_chart(fig, use_container_width=True)
        
        # Forecasted cost by subject
        if cost_data.get('forecasted_cost_by_subject'):
            st.subheader("🎯 Forecasted Cost by Subject")
            forecast_data = cost_data['forecasted_cost_by_subject']
            if any(v > 0 for v in forecast_data.values()):
                forecast_df = pd.DataFrame(list(forecast_data.items()), 
                                         columns=['Subject', 'Forecasted Cost'])
                forecast_df = forecast_df[forecast_df['Forecasted Cost'] > 0].sort_values('Forecasted Cost', ascending=False)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(forecast_df, use_container_width=True)
                with col2:
                    if len(forecast_df) > 0:
                        fig = px.bar(forecast_df.head(8), x='Forecasted Cost', y='Subject',
                                   title="Top Forecasted Costs by Subject", orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No positive forecasted costs to display")
        
        # ===== COMPREHENSIVE COST ANALYTICS FROM ORIGINAL SCRIPT =====
        
        # Top 20 Forecasted by Pilot
        if cost_data.get('top20_forecasted_by_pilot'):
            st.subheader("🚀 Top 20 Forecasted Cost by Pilot")
            forecast_pilots = cost_data['top20_forecasted_by_pilot']
            if forecast_pilots:
                forecast_pilots_df = pd.DataFrame(forecast_pilots, columns=['Pilot', 'Forecasted Cost'])
                forecast_pilots_df = forecast_pilots_df[forecast_pilots_df['Forecasted Cost'] > 0]
                
                if len(forecast_pilots_df) > 0:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(forecast_pilots_df, use_container_width=True)
                    with col2:
                        fig = px.bar(forecast_pilots_df.head(10), x='Forecasted Cost', y='Pilot',
                                   title="Top 10 Forecasted by Pilot", orientation='h')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No positive pilot forecasts to display")
        
        # Forecasted Cost by Violation Type
        if cost_data.get('forecasted_cost_by_violation'):
            st.subheader("⚖️ Forecasted Cost by Violation Type")
            violation_forecast = cost_data['forecasted_cost_by_violation']
            positive_forecasts = {k: v for k, v in violation_forecast.items() if v > 0}
            
            if positive_forecasts:
                violation_df = pd.DataFrame(list(positive_forecasts.items()), 
                                          columns=['Violation', 'Forecasted Cost'])
                violation_df = violation_df.sort_values('Forecasted Cost', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(violation_df, use_container_width=True, height=300)
                with col2:
                    fig = px.bar(violation_df.head(8), x='Forecasted Cost', y='Violation',
                               title="Top Violations by Forecast", orientation='h')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Forecasted Cost by Month
        if cost_data.get('forecasted_cost_by_month'):
            st.subheader("📅 Forecasted Cost by Month")
            monthly_forecast = cost_data['forecasted_cost_by_month']
            positive_monthly = {k: v for k, v in monthly_forecast.items() if v > 0}
            
            if positive_monthly:
                monthly_df = pd.DataFrame(list(positive_monthly.items()), 
                                        columns=['Month', 'Forecasted Cost'])
                monthly_df = monthly_df.sort_values('Month')
                
                # Format the Forecasted Cost column for display
                monthly_df_display = monthly_df.copy()
                monthly_df_display['Forecasted Cost'] = monthly_df_display['Forecasted Cost'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(monthly_df_display, use_container_width=True)
                with col2:
                    # Use original numeric values for the chart
                    fig = px.line(monthly_df, x='Month', y='Forecasted Cost',
                                title="Monthly Forecast Trend")
                    fig.update_xaxes(tickangle=45)
                    # Format y-axis to show currency
                    fig.update_yaxes(tickformat="$,.0f")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Aging Forecast
        if cost_data.get('aging_forecast'):
            st.subheader("📊 Cost Aging Forecast")
            aging_data = cost_data['aging_forecast']
            
            aging_df = pd.DataFrame(list(aging_data.items()), 
                                  columns=['Age Bucket (Days)', 'Forecasted Cost'])
            aging_df = aging_df[aging_df['Forecasted Cost'] > 0]
            
            if len(aging_df) > 0:
                # Format the Forecasted Cost column for display
                aging_df_display = aging_df.copy()
                aging_df_display['Forecasted Cost'] = aging_df_display['Forecasted Cost'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(aging_df_display, use_container_width=True)
                with col2:
                    # Use original numeric values for the chart
                    fig = px.bar(aging_df, x='Age Bucket (Days)', y='Forecasted Cost',
                               title="Forecasted Cost by Case Age")
                    # Format y-axis to show currency
                    fig.update_yaxes(tickformat="$,.0f")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Forecasted Cost Comparison
        st.subheader("📈 Actual vs Forecasted Cost Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Actual Cost", f"${cost_data.get('mean_paid_cost', 0):,.2f}")
            st.metric("Median Actual Cost", f"${cost_data.get('median_paid_cost', 0):,.2f}")
        
        with col2:
            st.metric("Mean Forecasted Cost", f"${cost_data.get('mean_forecasted_cost', 0):,.2f}")
            st.metric("Median Forecasted Cost", f"${cost_data.get('median_forecasted_cost', 0):,.2f}")
        
        with col3:
            st.metric("Number of Approved Cases", cost_data.get('num_approved_cases', 0))
            st.metric("Average Cost per Approved Case", f"${cost_data.get('avg_paid_per_case', 0):,.2f}")
        
        # Outlier Analysis
        if cost_data.get('outlier_cases'):
            st.subheader("🔍 High Cost Outlier Cases")
            outliers = cost_data['outlier_cases']
            if outliers:
                outlier_df = pd.DataFrame(outliers)
                # Format Relief_Dollars column for proper currency display
                if 'Relief_Dollars' in outlier_df.columns:
                    outlier_df['Relief_Dollars'] = outlier_df['Relief_Dollars'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
                
                # Select relevant columns for display
                display_cols = ['case_number', 'pilot', 'subject', 'Relief_Dollars', 'Status_Canonical']
                available_cols = [col for col in display_cols if col in outlier_df.columns]
                if available_cols:
                    st.dataframe(outlier_df[available_cols], use_container_width=True)
                else:
                    st.dataframe(outlier_df, use_container_width=True)
                
                forecast_df = pd.DataFrame(list(forecast_data.items()), 
                                         columns=['Subject', 'Forecasted Cost'])
                forecast_df = forecast_df[forecast_df['Forecasted Cost'] > 0]
                # Format Forecasted Cost column for proper currency display
                forecast_df['Forecasted Cost'] = forecast_df['Forecasted Cost'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00")
                forecast_df = forecast_df.sort_values('Forecasted Cost', ascending=False, key=lambda x: x.str.replace('$', '').str.replace(',', '').astype(float))
                
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
    
    # Add HH:MM format to display if not already present
    if 'relief_hhmm' not in filtered_df.columns and 'relief_minutes' in filtered_df.columns:
        filtered_df['relief_hhmm'] = filtered_df['relief_minutes'].apply(minutes_to_hhmm)
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} claims")
    
    # Reorder columns to show HH:MM prominently
    display_columns = ['case_number', 'pilot', 'subject', 'status']
    if 'relief_hhmm' in filtered_df.columns:
        display_columns.append('relief_hhmm')
    if 'relief_dollars' in filtered_df.columns:
        display_columns.append('relief_dollars')
    display_columns.extend([col for col in filtered_df.columns if col not in display_columns])
    
    st.dataframe(filtered_df[display_columns], use_container_width=True)
    
    # Quick Pilot Analytics
    st.subheader("👨‍✈️ Quick Pilot Analytics")
    
    # Get analytics for current filtered data
    if not filtered_df.empty:
        # Calculate relief hours and minutes for pilots
        pilot_relief_data = []
        for pilot in filtered_df['pilot'].unique():
            pilot_data = filtered_df[filtered_df['pilot'] == pilot]
            total_relief_dollars = pilot_data['relief_dollars'].sum()
            relief_rate = st.session_state.get('relief_rate', 320.47)
            total_relief_hours = total_relief_dollars / relief_rate
            total_relief_minutes = total_relief_hours * 60
            
            pilot_relief_data.append({
                'Pilot': pilot,
                'Cases': len(pilot_data),
                'Total Relief ($)': f"${total_relief_dollars:,.2f}",
                'Total Relief (HH:MM)': minutes_to_hhmm(total_relief_minutes),
                'Avg Relief ($)': f"${pilot_data['relief_dollars'].mean():,.2f}",
                'Approved Cases': (pilot_data['status'] == 'approved').sum()
            })
        
        pilot_summary_df = pd.DataFrame(pilot_relief_data)
        pilot_summary_df = pilot_summary_df.sort_values('Cases', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Pilot Summary (Current Filters):**")
            st.dataframe(pilot_summary_df, use_container_width=True)
        
        with col2:
            # Quick stats
            multi_case_pilots = pilot_summary_df[pilot_summary_df['Cases'] > 1]
            st.metric("Pilots with Multiple Cases", len(multi_case_pilots))
            if not multi_case_pilots.empty:
                st.metric("Max Cases by One Pilot", int(pilot_summary_df['Cases'].max()))
                st.metric("Avg Cases (Multi-Case Pilots)", f"{multi_case_pilots['Cases'].mean():.1f}")
    
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
    # Check if we're in a cloud environment where Selenium won't work
    is_cloud_env = (
        os.environ.get('STREAMLIT_SHARING') or 
        os.environ.get('HEROKU') or 
        os.environ.get('RENDER') or
        'streamlit.app' in os.environ.get('HOSTNAME', '') or
        platform.system() == 'Linux' and '/home/appuser' in os.environ.get('HOME', '')
    )
    
    if is_cloud_env:
        st.error("🚫 **Data Collection Not Available in Cloud Environment**")
        st.warning("""
        **Browser automation (Selenium) cannot run in hosted environments like Streamlit Cloud.**
        
        **To use data collection:**
        1. **Run locally**: Download and run this dashboard on your local machine
        2. **Use demo mode**: Toggle "Demo Mode" to see the dashboard with sample data
        3. **Manual upload**: Export data from your local script and upload it
        
        **For now, please enable Demo Mode to explore the dashboard features.**
        """)
        return False
    
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
        
        # Relief Rate Configuration
        st.header("⚙️ Configuration")
        
        # Initialize relief rate in session state if not exists
        if 'relief_rate' not in st.session_state:
            st.session_state.relief_rate = 320.47
        
        relief_rate = st.number_input(
            "💰 Relief Rate ($/hour)",
            min_value=100.0,
            max_value=1000.0,
            value=st.session_state.relief_rate,
            step=0.01,
            format="%.2f",
            help="Average composite rate of pay per hour used for cost calculations"
        )
        
        # Preset relief rate buttons
        st.write("**Quick Presets:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("Standard\n$320.47", use_container_width=True):
                st.session_state.relief_rate = 320.47
                st.rerun()
        
        with preset_col2:
            if st.button("High Rate\n$400.00", use_container_width=True):
                st.session_state.relief_rate = 400.00
                st.rerun()
        
        with preset_col3:
            if st.button("Conservative\n$275.00", use_container_width=True):
                st.session_state.relief_rate = 275.00
                st.rerun()
        
        # Update session state when value changes
        if relief_rate != st.session_state.relief_rate:
            st.session_state.relief_rate = relief_rate
            st.success(f"Relief rate updated to ${relief_rate:.2f}/hour")
        
        # Reset button for relief rate
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset to Default", help="Reset to $320.47/hour"):
                st.session_state.relief_rate = 320.47
                st.rerun()
        
        with col2:
            # Display current configuration
            st.markdown(f"**Current:** ${relief_rate:.2f}/hr")
        
        st.divider()
        
        # FILE UPLOAD SECTION (Available in both demo and production mode)
        st.header("📁 Data Upload & Management")
        
        upload_tab1, upload_tab2, upload_tab3 = st.tabs(["📤 Upload CSV", "📝 Manual Entry", "🔧 Instructions"])
        
        with upload_tab1:
            st.subheader("Upload Your STS Claims Data")
            st.write("Upload a CSV file with your claims data to use in the dashboard.")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type=['csv'],
                help="CSV should have columns: case_number, pilot, subject, status, relief_minutes (or relief_dollars), submission_date"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Show preview
                    st.write("**File Preview:**")
                    st.dataframe(df_upload.head(), use_container_width=True)
                    
                    # Validate columns
                    required_cols = ['case_number', 'pilot', 'subject', 'status']
                    optional_cols = ['relief_minutes', 'relief_dollars', 'relief_requested', 'submission_date']
                    
                    missing_required = [col for col in required_cols if col not in df_upload.columns]
                    available_optional = [col for col in optional_cols if col in df_upload.columns]
                    
                    if missing_required:
                        st.error(f"❌ Missing required columns: {missing_required}")
                        st.write("**Required columns:** case_number, pilot, subject, status")
                    else:
                        st.success("✅ All required columns found!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Records", len(df_upload))
                        with col2:
                            st.metric("Available Optional Columns", len(available_optional))
                        
                        # Handle relief data - check for relief_requested first
                        if 'relief_requested' in df_upload.columns and 'relief_minutes' not in df_upload.columns:
                            # Convert relief_requested (HH:MM format) to relief_minutes
                            df_upload['relief_minutes'] = df_upload['relief_requested'].apply(hhmm_to_minutes)
                            st.info("✅ Converted relief_requested (HH:MM) to relief_minutes")
                        elif 'relief_minutes' not in df_upload.columns and 'relief_dollars' not in df_upload.columns:
                            st.warning("⚠️ No relief data found. Adding default relief values.")
                            df_upload['relief_minutes'] = 60  # Default 1 hour
                        
                        if 'relief_dollars' not in df_upload.columns and 'relief_minutes' in df_upload.columns:
                            relief_rate = st.session_state.get('relief_rate', 320.47)
                            df_upload['relief_dollars'] = df_upload['relief_minutes'] * (relief_rate / 60)
                        
                        if 'submission_date' not in df_upload.columns:
                            st.warning("⚠️ No submission date found. Using today's date.")
                            df_upload['submission_date'] = datetime.now().strftime('%Y-%m-%d')
                        
                        if st.button("📊 Use This Data", type="primary"):
                            # Save to session state
                            st.session_state['uploaded_data'] = df_upload
                            st.session_state['data_source'] = 'uploaded'
                            st.session_state['data_collected'] = True
                            
                            # Save to file for persistence
                            df_upload.to_csv('uploaded_claims_data.csv', index=False)
                            
                            st.success(f"🎉 Data uploaded successfully! {len(df_upload)} records loaded.")
                            st.balloons()
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    st.write("Please ensure your CSV file is properly formatted.")
        
        with upload_tab2:
            st.subheader("Manual Data Entry")
            st.write("Add individual records manually (useful for testing):")
            
            with st.form("manual_entry_form"):
                col1, col2 = st.columns(2)
                with col1:
                    case_num = st.text_input("Case Number*", placeholder="e.g., 12345")
                    pilot_emp = st.text_input("Pilot Employee #*", placeholder="e.g., N123456")
                with col2:
                    violation = st.text_input("Subject/Violation*", placeholder="e.g., Rest Violation")
                    status = st.selectbox("Status*", ["open", "approved", "denied", "in review", "impasse", "contested"])
                
                col3, col4 = st.columns(2)
                with col3:
                    relief_input_type = st.radio("Relief Input Format", ["Hours (decimal)", "HH:MM"], horizontal=True)
                    if relief_input_type == "Hours (decimal)":
                        relief_hrs = st.number_input("Relief Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.25)
                        relief_minutes = relief_hrs * 60
                    else:
                        relief_hhmm = st.text_input("Relief HH:MM", value="01:00", placeholder="e.g., 02:30")
                        relief_minutes = hhmm_to_minutes(relief_hhmm)
                        st.write(f"= {relief_minutes/60:.2f} hours")
                with col4:
                    sub_date = st.date_input("Submission Date", value=datetime.now().date())
                
                submitted = st.form_submit_button("➕ Add Record", type="primary")
                
                if submitted:
                    if case_num and pilot_emp and violation:
                        relief_rate = st.session_state.get('relief_rate', 320.47)
                        new_record = {
                            'case_number': case_num,
                            'pilot': pilot_emp,
                            'subject': violation,
                            'status': status,
                            'relief_minutes': relief_minutes,
                            'relief_dollars': relief_hrs * relief_rate,
                            'submission_date': sub_date.strftime('%Y-%m-%d')
                        }
                        
                        # Initialize or append to manual data
                        if 'manual_records' not in st.session_state:
                            st.session_state['manual_records'] = []
                        
                        st.session_state['manual_records'].append(new_record)
                        
                        # Update data source
                        if st.session_state.get('data_source') != 'uploaded':
                            manual_df = pd.DataFrame(st.session_state['manual_records'])
                            st.session_state['uploaded_data'] = manual_df
                            st.session_state['data_source'] = 'manual'
                            st.session_state['data_collected'] = True
                            manual_df.to_csv('manual_claims_data.csv', index=False)
                        
                        st.success(f"✅ Record added! Total records: {len(st.session_state['manual_records'])}")
                        st.rerun()
                    else:
                        st.error("❌ Please fill in all required fields (marked with *)")
            
            # Show current manual records
            if 'manual_records' in st.session_state and st.session_state['manual_records']:
                st.write(f"**Current Manual Records: {len(st.session_state['manual_records'])}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 View All Records"):
                        st.dataframe(pd.DataFrame(st.session_state['manual_records']), use_container_width=True)
                with col2:
                    if st.button("🗑️ Clear All Manual Records"):
                        st.session_state['manual_records'] = []
                        st.rerun()
        
        with upload_tab3:
            st.subheader("🔧 Data Collection Instructions")
            
            st.write("**Option 1: Run your existing script locally**")
            st.code("""
# Run your comprehensive script:
python sts_totalpackage_v2_Version5_Version2.py

# This generates CSV files like:
# - sts_claims_analytics_TIMESTAMP.csv
# - sts_claims_monthly_trends_TIMESTAMP.csv

# Upload the main analytics CSV file using the Upload tab above
            """)
            
            st.write("**Option 1b: Use Standalone Data Collector**")
            st.write("Download a simplified data collection script that you can run locally:")
            
            # Read the standalone script content
            try:
                with open('sts_data_collector.py', 'r') as f:
                    script_content = f.read()
                
                st.download_button(
                    label="📥 Download Standalone Collector Script",
                    data=script_content,
                    file_name="sts_data_collector.py",
                    mime="text/x-python",
                    help="Download a standalone Python script for collecting STS data locally"
                )
                
                st.info("""
                **How to use the standalone collector:**
                1. Download the script above
                2. Install requirements: `pip install selenium pandas webdriver-manager`
                3. Run: `python sts_data_collector.py`
                4. Upload the generated CSV file using the Upload tab above
                """)
            except:
                st.warning("Standalone collector script not available in this environment.")
            
            st.write("**Option 2: Export from other tools**")
            st.write("Your CSV should have these columns:")
            st.json({
                "required": ["case_number", "pilot", "subject", "status"],
                "optional": ["relief_minutes", "relief_dollars", "submission_date"],
                "example_data": {
                    "case_number": "12345",
                    "pilot": "N123456", 
                    "subject": "Rest Violation",
                    "status": "approved",
                    "relief_minutes": 120,
                    "relief_dollars": 640.94,
                    "submission_date": "2025-01-15"
                }
            })
            
            st.write("**Option 3: Use Demo Mode**")
            st.write("Toggle 'Demo Mode' above to explore the dashboard with sample data.")
            
            st.write("**Option 4: Download CSV Template**")
            # Create a sample CSV template showing both HH:MM and minutes/dollars
            template_data = {
                'case_number': ['12345', '12346', '12347'],
                'pilot': ['N123456', 'N123457', 'N123458'],
                'subject': ['Rest Violation', '11.F', 'Yellow Slip / 12.T'],
                'status': ['approved', 'denied', 'open'],
                'relief_requested': ['02:00', '01:00', '03:00'],  # HH:MM format
                'relief_minutes': [120, 60, 180],  # Optional - will be calculated if missing
                'relief_dollars': [640.94, 320.47, 961.41],  # Optional - will be calculated if missing
                'submission_date': ['2025-01-15', '2025-01-16', '2025-01-17']
            }
            template_df = pd.DataFrame(template_data)
            
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Template",
                data=csv_template,
                file_name="sts_claims_template.csv",
                mime="text/csv",
                help="Template shows relief_requested in HH:MM format. Dashboard will auto-calculate minutes/dollars if missing."
            )
        
        st.divider()
        
        # Data collection section (only in production mode)
        if not demo_mode:
            st.header("📥 Data Collection")
            
            # Check if we're in a cloud environment
            is_cloud_env = (
                os.environ.get('STREAMLIT_SHARING') or 
                os.environ.get('HEROKU') or 
                os.environ.get('RENDER') or
                'streamlit.app' in os.environ.get('HOSTNAME', '') or
                platform.system() == 'Linux' and '/home/appuser' in os.environ.get('HOME', '')
            )
            
            if is_cloud_env:
                st.error("🚫 **Data Collection Not Available in Cloud Environment**")
                st.warning("""
                **Browser automation (Selenium) cannot run in hosted environments like Streamlit Cloud.**
                
                **To use data collection:**
                - 💻 **Run locally**: Download and run this dashboard on your local machine
                - 🎮 **Use demo mode**: Toggle "Demo Mode" above to see sample data
                - 📤 **Manual upload**: Export data from your local script and upload it
                """)
                st.info("**💡 Tip:** Enable Demo Mode above to explore the dashboard with sample data!")
            else:
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📊 Analytics", "💰 Financial", "📋 Claims Details", "🔍 Comprehensive Analytics"])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_analytics_tab()
    
    with tab3:
        show_financial_tab()
    
    with tab4:
        show_claims_details_tab()
    
    with tab5:
        show_comprehensive_analytics_tab()

def show_comprehensive_analytics_tab():
    """Show comprehensive analytics matching original script detail"""
    st.header("🔍 Comprehensive Analytics")
    st.markdown("*Complete analytics matching original script with full detail and breakdown*")
    
    df = get_data()
    if df.empty:
        st.warning("No data available. Please upload data or enable demo mode.")
        return
    
    relief_rate = st.session_state.get('relief_rate', 320.47)
    analytics = calculate_comprehensive_analytics(df, relief_rate)
    
    # Create multiple sections with expanders
    
    # === STATUS BREAKDOWN WITH COMPREHENSIVE DETAILS ===
    with st.expander("📊 Case Status Breakdown (Comprehensive)", expanded=True):
        st.subheader("Status Analysis with Relief & Percentages")
        
        status_data = []
        total_cases = len(df)
        total_relief_minutes = df['relief_minutes'].sum() if 'relief_minutes' in df.columns else 0
        
        for status in analytics['all_statuses']:
            status_df = df[df['status'].apply(status_canonical) == status]
            count = len(status_df)
            relief_mins = status_df['relief_minutes'].sum() if 'relief_minutes' in status_df.columns else 0
            relief_hhmm = minutes_to_hhmm(relief_mins)
            relief_cost = relief_dollars(relief_mins, relief_rate)
            pct_cases = (count / total_cases * 100) if total_cases > 0 else 0
            pct_relief = (relief_mins / total_relief_minutes * 100) if total_relief_minutes > 0 else 0
            
            status_data.append({
                'Status': status.title(),
                'Case Count': count,
                '% of Total Cases': f"{pct_cases:.2f}%",
                'Relief (HH:MM)': relief_hhmm,
                'Relief Cost': f"${relief_cost:,.2f}",
                '% of Total Relief': f"{pct_relief:.2f}%"
            })
        
        status_df_display = pd.DataFrame(status_data)
        st.dataframe(status_df_display, use_container_width=True)
    
    # === 5 OLDEST INCIDENT CASES ===
    with st.expander("📅 5 Oldest Incident Cases", expanded=True):
        if 'oldest_5_cases' in analytics and analytics['oldest_5_cases']:
            st.subheader("Oldest Cases by Incident Date")
            oldest_data = []
            for case in analytics['oldest_5_cases']:
                oldest_data.append({
                    'Date': case['date'].strftime("%Y-%m-%d") if hasattr(case['date'], 'strftime') else str(case['date']),
                    'Case Number': case['case_number'],
                    'Pilot Employee #': case['pilot']
                })
            oldest_df = pd.DataFrame(oldest_data)
            st.dataframe(oldest_df, use_container_width=True)
        else:
            st.info("No incident date data available for oldest cases analysis.")
    
    # === VIOLATION BREAKDOWN BY SUBJECT INCLUDING STATUS SEGMENTATION ===
    with st.expander("🎯 Violation Breakdown by Subject with Status Segmentation", expanded=True):
        st.subheader("Subject Analysis with Relief by Status")
        
        # Create comprehensive subject breakdown
        subject_breakdown_data = []
        for subject, stats in analytics['subject_stats'].items():
            # Base data
            total_cases = stats['count']
            total_relief_mins = stats.get('minutes', 0)
            total_relief_hhmm = minutes_to_hhmm(total_relief_mins)
            total_relief_cost = relief_dollars(total_relief_mins, relief_rate)
            
            # Status breakdown
            status_details = {}
            for status in analytics['all_statuses']:
                safe_status = status.replace(' ', '_').replace('-', '_')
                count_key = f"{safe_status}_count"
                dollars_key = f"{safe_status}_dollars"
                status_details[f"{status.title()} Count"] = stats.get(count_key, 0)
                status_details[f"{status.title()} Relief"] = f"${stats.get(dollars_key, 0):,.2f}"
            
            row_data = {
                'Subject': subject,
                'Total Cases': total_cases,
                'Total Relief (HH:MM)': total_relief_hhmm,
                'Total Relief Cost': f"${total_relief_cost:,.2f}",
                **status_details
            }
            subject_breakdown_data.append(row_data)
        
        # Sort by total relief cost
        subject_breakdown_data.sort(key=lambda x: float(x['Total Relief Cost'].replace('$', '').replace(',', '')), reverse=True)
        subject_breakdown_df = pd.DataFrame(subject_breakdown_data)
        st.dataframe(subject_breakdown_df, use_container_width=True)
    
    # === EMPLOYEE MULTIPLE CASES ANALYTICS ===
    with st.expander("👥 Employee Multiple Cases Analytics", expanded=True):
        st.subheader("Pilots with Multiple Case Submissions")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            unique_pilots = len(df['pilot'].unique())
            st.metric("Unique Pilots", unique_pilots)
        
        with col2:
            pilots_with_multiple = len([p for p, count in df['pilot'].value_counts().items() if count > 1])
            st.metric("Pilots with Multiple Cases", pilots_with_multiple)
        
        with col3:
            multiple_case_total = sum([count for count in df['pilot'].value_counts() if count > 1])
            total_cases = len(df)
            multiple_percentage = (multiple_case_total / total_cases * 100) if total_cases > 0 else 0
            st.metric("% of Total Caseload from Multiple Submitters", f"{multiple_percentage:.2f}%")
        
        with col4:
            st.metric("Total Cases from Multiple Submitters", multiple_case_total)
        
        # Detailed breakdown
        st.subheader("Detailed Multiple Case Analysis")
        multiple_cases_data = []
        pilot_counts = df['pilot'].value_counts()
        
        for pilot, count in pilot_counts.items():
            if count > 1:
                pilot_df = df[df['pilot'] == pilot]
                total_relief_mins = pilot_df['relief_minutes'].sum() if 'relief_minutes' in pilot_df.columns else 0
                total_relief_hhmm = minutes_to_hhmm(total_relief_mins)
                total_relief_cost = relief_dollars(total_relief_mins, relief_rate)
                subjects = ', '.join(pilot_df['subject'].apply(group_subject_key).unique())
                statuses = ', '.join(pilot_df['status'].apply(status_canonical).unique())
                
                multiple_cases_data.append({
                    'Pilot Employee #': pilot,
                    'Number of Cases': count,
                    'Total Relief (HH:MM)': total_relief_hhmm,
                    'Total Relief Cost': f"${total_relief_cost:,.2f}",
                    'Subject Types': subjects,
                    'Case Statuses': statuses
                })
        
        # Sort by number of cases
        multiple_cases_data.sort(key=lambda x: x['Number of Cases'], reverse=True)
        multiple_cases_df = pd.DataFrame(multiple_cases_data)
        st.dataframe(multiple_cases_df, use_container_width=True)
    
    # === TOP 20 PILOTS BY RELIEF (OVERALL AND BY STATUS) ===
    with st.expander("🏆 Top 20 Pilots by Relief Requested", expanded=True):
        st.subheader("Top Pilots by Relief Amount")
        
        # Overall top 20
        st.markdown("**Top 20 Pilots (Overall Relief)**")
        top20_overall_data = []
        for i, (pilot, relief_cost) in enumerate(analytics['top20_pilots_overall'].items(), 1):
            relief_hours = relief_cost / relief_rate
            relief_mins = relief_hours * 60
            relief_hhmm = minutes_to_hhmm(relief_mins)
            top20_overall_data.append({
                'Rank': i,
                'Pilot Employee #': pilot,
                'Relief (HH:MM)': relief_hhmm,
                'Relief Cost': f"${relief_cost:,.2f}"
            })
        
        top20_overall_df = pd.DataFrame(top20_overall_data)
        st.dataframe(top20_overall_df, use_container_width=True)
        
        # By status
        for status, pilot_data in analytics['top20_pilots_by_status'].items():
            if pilot_data:  # Only show if there's data
                st.markdown(f"**Top 20 Pilots - {status.title()} Status**")
                status_data = []
                for i, (pilot, relief_cost) in enumerate(pilot_data.items(), 1):
                    relief_hours = relief_cost / relief_rate
                    relief_mins = relief_hours * 60
                    relief_hhmm = minutes_to_hhmm(relief_mins)
                    status_data.append({
                        'Rank': i,
                        'Pilot Employee #': pilot,
                        'Relief (HH:MM)': relief_hhmm,
                        'Relief Cost': f"${relief_cost:,.2f}"
                    })
                
                status_df = pd.DataFrame(status_data)
                st.dataframe(status_df, use_container_width=True)
    
    # === VIOLATION TYPE COUNTS AND PERCENTAGES ===
    with st.expander("📋 Violation Type Analysis", expanded=True):
        st.subheader("Raw Violation Types with Counts and Percentages")
        
        violation_data = []
        total_cases = len(df)
        for violation, count in analytics['violation_counter'].items():
            percentage = (count / total_cases) * 100 if total_cases > 0 else 0
            violation_data.append({
                'Violation Type': violation,
                'Case Count': count,
                'Percentage of Total': f"{percentage:.2f}%"
            })
        
        # Sort by count
        violation_data.sort(key=lambda x: x['Case Count'], reverse=True)
        violation_df = pd.DataFrame(violation_data)
        st.dataframe(violation_df, use_container_width=True)
        
        st.metric("Total Unique Violation Types", analytics['unique_violation_types'])
    
    # === RECENT CASES AND TOP PILOTS BY CASES ===
    with st.expander("📈 Recent Activity & Top Submitters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Activity")
            st.metric("Cases with Activity in Past 7 Days", analytics['recent_cases'])
        
        with col2:
            st.subheader("Top 10 Pilots by Number of Cases")
            top10_data = []
            for i, (pilot, count) in enumerate(analytics['top_10_pilots_by_cases'].items(), 1):
                top10_data.append({
                    'Rank': i,
                    'Pilot Employee #': pilot,
                    'Number of Cases': count
                })
            
            top10_df = pd.DataFrame(top10_data)
            st.dataframe(top10_df, use_container_width=True)
    
    # === PROBABILITY OF PAYMENT BY SUBJECT ===
    with st.expander("🎲 Probability Analysis", expanded=True):
        st.subheader("Probability of Payment by Subject")
        st.markdown("*Based on Approved/(Approved+Denied) ratio*")
        
        prob_data = []
        for subject, prob in analytics['probability_by_subject'].items():
            prob_data.append({
                'Subject': subject,
                'Probability of Payment': f"{prob*100:.2f}%"
            })
        
        # Sort by probability
        prob_data.sort(key=lambda x: float(x['Probability of Payment'].replace('%', '')), reverse=True)
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, use_container_width=True)
    
    # === PROJECTED COSTS FOR OPEN/IN REVIEW CASES ===
    with st.expander("💰 Cost Projections", expanded=True):
        st.subheader("Projected Costs for Open and In Review Cases")
        st.markdown("*Based on probability of payment and average relief amounts*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Forecasted Costs by Subject**")
            forecast_data = []
            total_forecasted = 0
            for subject, cost in analytics['cost_analytics']['forecasted_cost_by_subject'].items():
                forecast_data.append({
                    'Subject': subject,
                    'Forecasted Cost': f"${cost:,.2f}"
                })
                total_forecasted += cost
            
            forecast_data.sort(key=lambda x: float(x['Forecasted Cost'].replace('$', '').replace(',', '')), reverse=True)
            forecast_df = pd.DataFrame(forecast_data)
            st.dataframe(forecast_df, use_container_width=True)
            st.metric("Total Forecasted Cost", f"${total_forecasted:,.2f}")
        
        with col2:
            st.markdown("**Actual Costs (Approved Cases)**")
            actual_data = []
            total_actual = 0
            for subject, cost in analytics['cost_analytics']['actual_paid_by_subject'].items():
                actual_data.append({
                    'Subject': subject,
                    'Actual Paid': f"${cost:,.2f}"
                })
                total_actual += cost
            
            actual_data.sort(key=lambda x: float(x['Actual Paid'].replace('$', '').replace(',', '')), reverse=True)
            actual_df = pd.DataFrame(actual_data)
            st.dataframe(actual_df, use_container_width=True)
            st.metric("Total Actual Paid", f"${total_actual:,.2f}")

if __name__ == "__main__":
    main()
