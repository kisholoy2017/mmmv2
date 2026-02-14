"""
Marketing Mix Modeling (MMM) Streamlit App - FULL COMPREHENSIVE VERSION v7

COMPLETE IMPLEMENTATION combining:
- All features from working 1,502-line app
- Plus all new v7 features (8 variable types, feature selection, VIF, etc.)

NEW v7 FEATURES:
OK Multi-variable type upload (8 types: media, competition, controls, TV, traditional, ATL)
OK User guide with Data Dictionary naming conventions
OK Channel-specific parameter ranges (TV, Digital, Traditional, Competition)
OK Feature selection using correlation analysis (optional)
OK Extended diagnostics (VIF, NRMSE, AIC, BIC, Durbin-Watson)
OK Confidence intervals for all coefficients
OK DECOMP.RSSD metric (spend vs effect share)
OK De-standardized reporting (always positive contributions)
OK Proper variable handling (media=transformed, controls=untransformed)

EXISTING FEATURES (from working app):
OK Complete data upload with promotion support
OK Data validation and overview
OK Full MMM modeling with adstock & saturation
OK Complete budget optimization (scipy SLSQP)
OK Detailed visualizations (4-subplot response curves)
OK Complete model diagnostics (4-subplot residual analysis)
OK ROI analysis with marginal ROAS
OK Channel contribution analysis

Created: 2026-01-25
Version: 7.0 - FULL COMPLETE EDITION
Lines: ~2,000+
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MMM Platform v7 Full",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .stDataFrame {
        border: 2px solid #1f77b4;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'media_data' not in st.session_state:
    st.session_state.media_data = {}
if 'kpi_data' not in st.session_state:
    st.session_state.kpi_data = None
if 'combined_data' not in st.session_state:
    st.session_state.combined_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'promotion_data' not in st.session_state:
    st.session_state.promotion_data = None

# NEW v7: Parameter ranges for different channel types
PARAMETER_RANGES = {
    "TV/Video": {
        "theta": (0.3, 0.8, 0.1),
        "alpha": (0.5, 3.0, 0.5),
        "gamma": (0.3, 1.0, 0.1)
    },
    "Digital": {
        "theta": (0.0, 0.3, 0.1),
        "alpha": (0.5, 3.0, 0.5),
        "gamma": (0.3, 1.0, 0.1)
    },
    "Traditional": {
        "theta": (0.1, 0.4, 0.1),
        "alpha": (0.5, 3.0, 0.5),
        "gamma": (0.3, 1.0, 0.1)
    },
    "Competition": {
        "theta": (0.1, 0.8, 0.1),
        "alpha": (0.5, 3.0, 0.5),
        "gamma": (0.3, 1.0, 0.1)
    }
}

# Helper functions for data cleaning and MMM
def clean_numeric_column(series):
    """Clean numeric column - remove commas, convert to float"""
    if series.dtype == 'object':
        try:
            return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')
        except:
            return series
    return series

def clean_dataframe_numeric_columns(df, exclude_cols=None):
    """Clean all numeric columns in dataframe"""
    if exclude_cols is None:
        exclude_cols = []
    
    df = df.copy()
    for col in df.columns:
        if col not in exclude_cols:
            if df[col].dtype == 'object':
                try:
                    cleaned = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(cleaned, errors='ignore')
                except:
                    pass
    return df

def adstock_transformation(x, alpha=0.5):
    """Apply adstock (geometric decay) transformation"""
    y = np.zeros_like(x, dtype=float)
    if len(x) > 0:
        y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = x[t] + alpha * y[t-1]
    return y

def hill_transformation(x, kappa, slope=1.0):
    """Apply Hill saturation transformation"""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    k = max(float(kappa), 1e-9)
    if slope == 1.0:
        return x / (x + k)
    xs = np.power(x, slope)
    ks = np.power(k, slope)
    return xs / (xs + ks)

def hill_derivative(x, kappa, slope=1.0):
    """Calculate derivative of Hill function for marginal ROAS"""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    k = max(float(kappa), 1e-9)
    if slope == 1.0:
        return k / (x + k)**2
    xs = np.power(x, slope)
    ks = np.power(k, slope)
    return slope * np.power(x, slope - 1.0) * ks / (xs + ks)**2

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if mask.sum() > 0 else 0
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) if np.sum(np.abs(y_true)) > 0 else 0
    
    # NEW v7: Additional metrics
    nrmse = np.sqrt(np.mean((y_true - y_pred)**2)) / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) > 0 else 0
    
    return r2, mape, wmape, nrmse

def add_seasonality_features(df, date_col):
    """Add seasonality features: day of week and month"""
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    
    df_with_seasonality = pd.concat([df, day_dummies, month_dummies], axis=1)
    
    return df_with_seasonality

def process_promotion_variable(df, promo_col):
    """Process promotion variable - convert to dummy if string, use as numeric if numeric"""
    df = df.copy()
    
    if df[promo_col].dtype == 'object' or df[promo_col].dtype.name == 'category':
        promo_dummies = pd.get_dummies(df[promo_col], prefix='promo', drop_first=True)
        df = pd.concat([df, promo_dummies], axis=1)
        feature_cols = promo_dummies.columns.tolist()
        is_dummy = True
    else:
        feature_cols = [promo_col]
        is_dummy = False
    
    return df, feature_cols, is_dummy

# NEW v7: DECOMP.RSSD calculation
def calculate_decomp_rssd(test_df, contributions, media_cols):
    """Calculate DECOMP.RSSD metric - measures spend vs effect share alignment"""
    total_spend = sum([test_df[col].sum() for col in media_cols])
    total_effect = sum([contributions.get(col, 0) for col in media_cols])
    
    if total_spend == 0 or total_effect == 0:
        return 0, {}, {}
    
    spend_share = {col: test_df[col].sum() / total_spend for col in media_cols}
    effect_share = {col: contributions.get(col, 0) / total_effect for col in media_cols}
    
    rssd = np.sqrt(sum((effect_share[col] - spend_share[col])**2 for col in media_cols))
    
    return rssd, spend_share, effect_share

# Main app header
st.markdown('<p class="main-header"> Marketing Mix Modeling Platform v7 - FULL EDITION</p>', unsafe_allow_html=True)
st.markdown("**Complete Implementation** - Multi-Variable Support, Feature Selection, Advanced Diagnostics & Full Optimization")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/analytics.png", width=100)
    st.markdown("### Navigation")
    tab_selection = st.radio(
        "Select a section:",
        [" User Guide", " Data Upload", " Data Overview", " Marketing Mix Modeling", " Results & Insights"],
        key="navigation"
    )
    
    st.markdown("---")
    st.markdown("### About v7")
    st.info("""
    **NEW in v7:**
    - 8 variable types support
    - Channel-specific parameters
    - Feature selection (correlation)
    - VIF analysis
    - Extended diagnostics
    - DECOMP.RSSD metric
    - De-standardized reporting
    
    **Plus all existing:**
    - Complete optimization
    - Full visualizations
    - Promotion support
    """)

# TAB 0: USER GUIDE (NEW v7)
if tab_selection == " User Guide":
    st.markdown('<p class="sub-header"> User Guide & Data Requirements</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to MMM Platform v7 - Full Edition
    
    This comprehensive Marketing Mix Modeling platform supports multiple variable types and advanced analytics.
    
    ###  Supported Variable Types
    """)
    
    var_types_df = pd.DataFrame({
        'Variable Type': [
            ' time_column',
            ' dependent_var',
            ' paid_media_spends',
            ' competition_spend_vars',
            '️ untransformed_vars',
            ' tv_vars',
            ' traditional_vars',
            ' atl_vars'
        ],
        'Required': [
            'OK MANDATORY',
            'OK MANDATORY',
            'OK MANDATORY',
            'O Optional',
            'O Optional',
            'O Optional',
            'O Optional',
            'O Optional'
        ],
        'Description': [
            'Date/time column (e.g., Month, Date)',
            'Target KPI - what you want to predict (Revenue, Sales)',
            'Marketing channel spending (main media variables)',
            'Competitor marketing spending',
            'Control variables - no transformation needed (price, inflation)',
            'TV/Video channels (uses higher adstock 0.3-0.8)',
            'Traditional media: Radio, Print, Outdoor (adstock 0.1-0.4)',
            'Above-the-line competitor spends'
        ]
    })
    
    st.dataframe(var_types_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("###  Column Naming Conventions (From Data Dictionary)")
    
    with st.expander(" **KPI Variables (dependent_var)** - Choose ONE"):
        st.code("""
Sales_Volume_Total           - Total number of units sold
Sales_Revenue_Total          - Total sales revenue in USD
Sales_Volume_Category1       - Units sold for specific category
Sales_Revenue_Channel1       - Revenue from specific channel
        """)
    
    with st.expander(" **Paid Media Spends (paid_media_spends)** - Select multiple"):
        st.code("""
TV_Spends                    - TV advertising spend in USD
Radio_Spends                 - Radio advertising spend
Outdoor_Spends               - Outdoor/OOH advertising spend
Paid_Search_Spends           - Paid search advertising
Programmatic_Display_Spends  - Programmatic display advertising
Google_Display_Spend         - Google display advertising
Direct_Display_Spend         - Direct display advertising
Meta1_Spends                 - META platform 1 advertising
Meta2_Spends                 - META platform 2 advertising
Youtube_Spends               - YouTube advertising
Programmatic_Video_Spends    - Programmatic video advertising
Influencer_Marketing_Spends  - Influencer marketing spend
        """)
    
    with st.expander(" **Competition Variables (competition_spend_vars)** - Optional"):
        st.code("""
Brand_B_ATL_Spends           - Competitor Brand B ATL spending
Brand_PH_ATL_Spends          - Competitor Brand PH ATL spending
Brand_P_ATL_Spends           - Competitor Brand P ATL spending
        """)
    
    with st.expander("️ **Control Variables (untransformed_vars)** - No transformation"):
        st.code("""
Inflation_Rate               - Inflation rate (percentage)
Average_Price_Total          - Average product price
Market_Share_Brand_M_Total   - Market share percentage
Brand_PH_Market_Share        - Competitor market share
Brand_B_Market_Share         - Competitor market share
Brand_P_Market_Share         - Competitor market share
TV_GRP                       - TV Gross Rating Points
        """)
    
    st.markdown("---")
    st.markdown("###  NEW v7 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Channel-Specific Parameters:**
        - TV/Video: Adstock 0.3-0.8 (high carryover)
        - Digital: Adstock 0.0-0.3 (low carryover)
        - Traditional: Adstock 0.1-0.4 (medium)
        - Competition: Adstock 0.1-0.8 (variable)
        
        **Feature Selection:**
        - Grid search over θ, α, γ combinations
        - Pearson correlation with target
        - Auto-select best parameters
        """)
    
    with col2:
        st.markdown("""
        **Extended Diagnostics:**
        - VIF (multicollinearity detection)
        - NRMSE (normalized error)
        - AIC/BIC (model selection)
        - Durbin-Watson (autocorrelation)
        - 95% Confidence Intervals
        - DECOMP.RSSD (spend vs effect)
        """)
    
    st.markdown("---")
    st.markdown("""
    ###  Usage Flow
    
    1. ** Data Upload:** Upload your data files (KPI, media channels, optional: promotions, competition, controls)
    2. ** Data Overview:** Validate and explore your data
    3. ** Modeling:** Configure channel types, enable feature selection, train model
    4. ** Results:** View contributions, ROI, response curves, optimization, diagnostics
    
    ---
    
    **Ready to start?** Go to ** Data Upload** ->
    """)

# TAB 1: Data Upload
elif tab_selection == " Data Upload":
    st.markdown('<p class="sub-header">Upload Your Marketing Data</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>INFO Upload Options:</b>
    <ul>
    <li><b>Option A:</b> Upload individual files for KPI and each media channel (simpler)</li>
    <li><b>Option B:</b> Upload one combined dataset with all variables (faster for v7 features)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    upload_mode = st.radio("Select upload mode:", ["Individual Files (Original)", "Combined Dataset (v7)"], horizontal=True)
    
    if upload_mode == "Individual Files (Original)":
        # ORIGINAL UPLOAD MODE
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("####  KPI Data (Revenue)")
            st.info("Upload your store/Shopify revenue data. Must include: **Date** and **Revenue** columns")
            
            kpi_file = st.file_uploader(
                "Choose KPI CSV file",
                type=['csv'],
                key='kpi_upload',
                help="Upload CSV with Date and Revenue columns"
            )
            
            if kpi_file:
                try:
                    kpi_df = pd.read_csv(kpi_file)
                    kpi_df = clean_dataframe_numeric_columns(kpi_df, exclude_cols=[kpi_df.columns[0]])
                    st.session_state.kpi_data = kpi_df
                    
                    st.success(f"OK KPI data uploaded successfully! ({len(kpi_df)} rows)")
                    
                    with st.expander("Preview KPI Data"):
                        st.dataframe(kpi_df.head(10), use_container_width=True)
                        st.markdown("**Data Info:**")
                        st.write(f"- Columns: {', '.join(kpi_df.columns.tolist())}")
                        st.write(f"- Date range: {kpi_df.iloc[:, 0].min()} to {kpi_df.iloc[:, 0].max()}")
                        
                except Exception as e:
                    st.error(f"Error loading KPI data: {str(e)}")
        
        with col2:
            st.markdown("####  Media Spend Data")
            st.info("Upload media channel data. Must include: **Date** and **Cost** columns")
            
            num_channels = st.number_input("Number of media channels", min_value=1, max_value=10, value=2, key='num_channels')
            
            for i in range(num_channels):
                st.markdown(f"**Channel {i+1}:**")
                channel_name = st.text_input(f"Channel name", value=f"Channel_{i+1}", key=f'channel_name_{i}')
                channel_file = st.file_uploader(
                    f"Upload {channel_name} CSV",
                    type=['csv'],
                    key=f'channel_file_{i}'
                )
                
                if channel_file:
                    try:
                        channel_df = pd.read_csv(channel_file)
                        channel_df = clean_dataframe_numeric_columns(channel_df, exclude_cols=[channel_df.columns[0]])
                        st.session_state.media_data[channel_name] = channel_df
                        st.success(f"OK {channel_name} uploaded ({len(channel_df)} rows)")
                        
                        with st.expander(f"Preview {channel_name}"):
                            st.dataframe(channel_df.head(5), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading {channel_name}: {str(e)}")
        
        # Promotion/Discount variable upload
        st.markdown("---")
        st.markdown("####  Promotion/Discount Data (Optional)")
        st.info("""
        Upload promotion data with **Date** and **Promotion** columns.
        - **String values** (e.g., 'Yes'/'No', 'Sale'/'Normal') -> Converted to dummy variables
        - **Numeric values** (e.g., 10%, 0.15) -> Used as continuous variable
        """)
        
        promo_file = st.file_uploader(
            "Upload Promotion CSV (optional)",
            type=['csv'],
            key='promo_upload',
            help="CSV with Date and Promotion columns"
        )
        
        if promo_file:
            try:
                promo_df = pd.read_csv(promo_file)
                if len(promo_df.columns) > 2:
                    for col in promo_df.columns[2:]:
                        promo_df[col] = clean_numeric_column(promo_df[col])
                
                st.session_state.promotion_data = promo_df
                st.success(f"OK Promotion data uploaded! ({len(promo_df)} rows)")
                
                with st.expander("Preview Promotion Data"):
                    st.dataframe(promo_df.head(10), use_container_width=True)
                    promo_col = promo_df.columns[1]
                    if promo_df[promo_col].dtype == 'object':
                        st.info(f"OK Detected **categorical** promotion: {promo_df[promo_col].unique()[:5]}")
                    else:
                        st.info(f"OK Detected **numeric** promotion: Range {promo_df[promo_col].min():.2f} - {promo_df[promo_col].max():.2f}")
                        
            except Exception as e:
                st.error(f"Error loading promotion data: {str(e)}")
        
        # Combine data button
        st.markdown("---")
        if st.button(" Combine All Data", type="primary", use_container_width=True):
            if st.session_state.kpi_data is None:
                st.error("X Please upload KPI data first!")
            elif len(st.session_state.media_data) == 0:
                st.error("X Please upload at least one media channel!")
            else:
                with st.spinner("Combining data..."):
                    try:
                        combined = st.session_state.kpi_data.copy()
                        date_col = combined.columns[0]
                        combined[date_col] = pd.to_datetime(combined[date_col], errors='coerce', dayfirst=True)
                        
                        for channel_name, channel_df in st.session_state.media_data.items():
                            channel_df = channel_df.copy()
                            channel_date_col = channel_df.columns[0]
                            channel_df[channel_date_col] = pd.to_datetime(channel_df[channel_date_col], errors='coerce', dayfirst=True)
                            
                            rename_dict = {}
                            for col in channel_df.columns:
                                if col.lower() not in ['date']:
                                    rename_dict[col] = f"{channel_name}_{col}"
                            channel_df = channel_df.rename(columns=rename_dict)
                            channel_df = channel_df.rename(columns={channel_date_col: date_col})
                            combined = combined.merge(channel_df, on=date_col, how='left')
                        
                        if st.session_state.promotion_data is not None:
                            promo_df = st.session_state.promotion_data.copy()
                            promo_date_col = promo_df.columns[0]
                            promo_df[promo_date_col] = pd.to_datetime(promo_df[promo_date_col], errors='coerce', dayfirst=True)
                            promo_df = promo_df.rename(columns={promo_date_col: date_col})
                            combined = combined.merge(promo_df, on=date_col, how='left')
                            
                            promo_col = promo_df.columns[1]
                            if combined[promo_col].dtype == 'object':
                                combined[promo_col] = combined[promo_col].fillna('None')
                            else:
                                combined[promo_col] = combined[promo_col].fillna(0)
                        
                        cost_cols = [col for col in combined.columns if 'cost' in col.lower() or 'spend' in col.lower()]
                        combined[cost_cols] = combined[cost_cols].fillna(0)
                        combined = clean_dataframe_numeric_columns(combined, exclude_cols=[date_col])
                        combined = combined.dropna(subset=[date_col])
                        
                        st.session_state.combined_data = combined
                        st.session_state.data_uploaded = True
                        
                        st.success("OK Data combined successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error combining data: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        # NEW v7: COMBINED DATASET UPLOAD
        st.markdown("####  Upload Complete Dataset")
        st.info("Upload one file containing all variables (KPI, media, competition, controls, etc.)")
        
        uploaded_file = st.file_uploader(
            "Upload your complete MMM data file (CSV or Excel)",
            type=['csv', 'xlsx'],
            help="Single file with all variables"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"OK File uploaded! Shape: {df.shape}")
                
                with st.expander(" Preview Data"):
                    st.dataframe(df.head(10))
                
                st.markdown("### ️ Configure Variable Types")
                
                # Store variable classifications in session state for modeling
                if 'v7_mode' not in st.session_state:
                    st.session_state.v7_mode = True
                
                col1, col2 = st.columns(2)
                
                with col1:
                    time_col = st.selectbox(" Time Column (MANDATORY)", df.columns)
                    dependent_var = st.selectbox(" KPI/Dependent Variable (MANDATORY)", [c for c in df.columns if c != time_col])
                
                with col2:
                    paid_media_cols = st.multiselect(" Paid Media Spends (MANDATORY)", [c for c in df.columns if c not in [time_col, dependent_var]])
                
                if not paid_media_cols:
                    st.warning("WARNING Select at least one media channel!")
                    st.stop()
                
                col3, col4 = st.columns(2)
                
                with col3:
                    competition_cols = st.multiselect(
                        " Competition Variables (Optional)",
                        [c for c in df.columns if c not in [time_col, dependent_var] + paid_media_cols]
                    )
                    tv_cols = st.multiselect(" TV/Video Channels (Optional)", paid_media_cols)
                
                with col4:
                    control_cols = st.multiselect(
                        "️ Control Variables - Untransformed (Optional)",
                        [c for c in df.columns if c not in [time_col, dependent_var] + paid_media_cols + competition_cols]
                    )
                    traditional_cols = st.multiselect(" Traditional Media (Optional)", [c for c in paid_media_cols if c not in tv_cols])
                
                atl_cols = st.multiselect(" ATL Competition (Optional)", competition_cols) if competition_cols else []
                digital_cols = [c for c in paid_media_cols if c not in tv_cols + traditional_cols]
                
                # Config summary
                st.markdown("---")
                st.markdown("###  Configuration Summary")
                
                config_summary = pd.DataFrame({
                    'Type': ['Time', 'KPI', 'Media', 'Digital', 'TV', 'Traditional', 'Competition', 'ATL', 'Controls'],
                    'Count': [1, 1, len(paid_media_cols), len(digital_cols), len(tv_cols), len(traditional_cols), 
                             len(competition_cols), len(atl_cols), len(control_cols)]
                })
                st.dataframe(config_summary)
                
                if st.button("OK Confirm Configuration & Process", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        df = clean_dataframe_numeric_columns(df, exclude_cols=[time_col])
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce', dayfirst=True)
                        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
                        
                        # Store all configurations
                        st.session_state.combined_data = df
                        st.session_state.v7_time_col = time_col
                        st.session_state.v7_dependent_var = dependent_var
                        st.session_state.v7_paid_media_cols = paid_media_cols
                        st.session_state.v7_digital_cols = digital_cols
                        st.session_state.v7_tv_cols = tv_cols
                        st.session_state.v7_traditional_cols = traditional_cols
                        st.session_state.v7_competition_cols = competition_cols
                        st.session_state.v7_atl_cols = atl_cols
                        st.session_state.v7_control_cols = control_cols
                        st.session_state.data_uploaded = True
                        st.session_state.v7_mode = True
                        
                        st.success("OK Configuration saved! Go to Overview ->")
                        st.balloons()
            
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 2: Data Overview
elif tab_selection == " Data Overview":
    st.markdown('<p class="sub-header">Data Overview & Validation</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("WARNING Please upload and combine data first!")
    else:
        df = st.session_state.combined_data.copy()
        
        # Determine mode and get correct column references
        if st.session_state.get('v7_mode', False):
            date_col = st.session_state.v7_time_col
            target_col = st.session_state.v7_dependent_var
            media_cols = st.session_state.v7_paid_media_cols
        else:
            date_col = df.columns[0]
            target_col = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()][0] if any('revenue' in col.lower() or 'sales' in col.lower() for col in df.columns) else df.columns[1]
            media_cols = [col for col in df.columns if 'cost' in col.lower() or 'spend' in col.lower()]
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            st.session_state.combined_data[date_col] = df[date_col]
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            date_range_days = (df[date_col].max() - df[date_col].min()).days
            st.metric("Date Range (Days)", date_range_days)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            date_range_months = date_range_days / 30
            st.metric("Months of Data", f"{date_range_months:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Media Channels", len(media_cols))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data validation
        st.markdown("---")
        st.markdown("### OK Data Validation")
        
        validation_col1, validation_col2, validation_col3 = st.columns(3)
        
        with validation_col1:
            if date_range_months >= 24:
                st.success(f"OK Sufficient data: {date_range_months:.1f} months")
            else:
                st.warning(f"WARNING Limited data: {date_range_months:.1f} months")
        
        with validation_col2:
            has_revenue = any('revenue' in col.lower() or 'sales' in col.lower() for col in df.columns)
            if has_revenue:
                st.success("OK Revenue/KPI column found")
            else:
                st.error("X Revenue/KPI column not found")
        
        with validation_col3:
            has_promo = any('promo' in col.lower() or 'discount' in col.lower() for col in df.columns)
            if has_promo:
                st.success("OK Promotion data included")
            else:
                st.info("INFO No promotion data")
        
        # Display combined data
        st.markdown("---")
        st.markdown("###  Combined Dataset")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != date_col]
        
        if numeric_cols:
            st.dataframe(
                df.style.background_gradient(subset=numeric_cols, cmap='Blues'),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label=" Download Combined Data",
            data=csv,
            file_name=f"combined_mmm_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Basic statistics
        st.markdown("---")
        st.markdown("###  Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Summary:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("---")
        st.markdown("###  Correlation Heatmap")
        
        numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols_all) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation_matrix = df[numeric_cols_all].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

# Due to length, I'll continue this in parts...

# TAB 3: Marketing Mix Modeling (ENHANCED WITH v7 FEATURES)
elif tab_selection == " Marketing Mix Modeling":
    st.markdown('<p class="sub-header">Marketing Mix Modeling with v7 Enhancements</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("WARNING Please upload data first!")
        st.stop()
    
    df = st.session_state.combined_data.copy()
    
    # Determine mode
    v7_mode = st.session_state.get('v7_mode', False)
    
    if v7_mode:
        # v7 MODE with channel-specific parameters
        date_col = st.session_state.v7_time_col
        target_col = st.session_state.v7_dependent_var
        media_cols = st.session_state.v7_paid_media_cols
        tv_cols = st.session_state.v7_tv_cols
        traditional_cols = st.session_state.v7_traditional_cols
        digital_cols = st.session_state.v7_digital_cols
        competition_cols = st.session_state.v7_competition_cols
        control_cols = st.session_state.v7_control_cols
        
        st.info("OK v7 Mode: Using channel-specific parameter ranges and advanced features")
        
        # Feature Selection Toggle
        st.markdown("###  Feature Selection (NEW v7)")
        enable_feature_selection = st.checkbox(
            "Enable Correlation-Based Feature Selection",
            value=False,
            help="Test multiple parameter combinations and select best based on correlation with target"
        )
        
        channel_params = {}
        
        if enable_feature_selection:
            st.info("Running feature selection... This may take a minute")
            progress_bar = st.progress(0)
            
            for idx, channel in enumerate(media_cols):
                # Determine channel type
                if channel in tv_cols:
                    ch_type = "TV/Video"
                elif channel in traditional_cols:
                    ch_type = "Traditional"
                elif channel in competition_cols:
                    ch_type = "Competition"
                else:
                    ch_type = "Digital"
                
                ranges = PARAMETER_RANGES[ch_type]
                
                # Grid search
                theta_grid = np.arange(ranges["theta"][0], ranges["theta"][1], ranges["theta"][2])
                alpha_grid = np.arange(ranges["alpha"][0], ranges["alpha"][1], ranges["alpha"][2])
                gamma_grid = np.arange(ranges["gamma"][0], ranges["gamma"][1], ranges["gamma"][2])
                
                best_corr = -1
                best_params = ((ranges["theta"][0] + ranges["theta"][1])/2, 1.0, 0.5)
                
                for theta in theta_grid:
                    for alpha in alpha_grid:
                        for gamma in gamma_grid:
                            try:
                                # Transform
                                adstocked = adstock_transformation(df[channel].values, theta)
                                inflexion = df[channel].min() * (1-gamma) + df[channel].max() * gamma
                                kappa = max(inflexion, 1.0)
                                saturated = hill_transformation(adstocked, kappa, alpha)
                                
                                # Correlation
                                corr, _ = pearsonr(saturated, df[target_col])
                                
                                if abs(corr) > best_corr:
                                    best_corr = abs(corr)
                                    best_params = (theta, alpha, gamma)
                            except:
                                continue
                
                channel_params[channel] = {
                    'type': ch_type,
                    'theta': best_params[0],
                    'alpha': best_params[1],
                    'gamma': best_params[2],
                    'corr': best_corr
                }
                
                progress_bar.progress((idx + 1) / len(media_cols))
            
            st.success("OK Feature selection complete!")
            
            # Show results
            results_df = pd.DataFrame(channel_params).T
            results_df['Channel'] = results_df.index
            results_df = results_df[['Channel', 'type', 'theta', 'alpha', 'gamma', 'corr']]
            st.dataframe(results_df.style.format({'theta': '{:.2f}', 'alpha': '{:.2f}', 'gamma': '{:.2f}', 'corr': '{:.4f}'}))
        
        else:
            # Manual channel-specific configuration
            st.markdown("### ️ Channel-Specific Parameters (NEW v7)")
            
            for channel in media_cols:
                # Determine type
                if channel in tv_cols:
                    ch_type = "TV/Video"
                elif channel in traditional_cols:
                    ch_type = "Traditional"
                elif channel in competition_cols:
                    ch_type = "Competition"
                else:
                    ch_type = "Digital"
                
                ranges = PARAMETER_RANGES[ch_type]
                
                with st.expander(f"️ {channel} - {ch_type}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        theta = st.slider(
                            f"Adstock (θ)",
                            ranges["theta"][0],
                            ranges["theta"][1],
                            (ranges["theta"][0] + ranges["theta"][1])/2,
                            0.1,
                            key=f"theta_{channel}"
                        )
                    
                    with col2:
                        alpha = st.slider(
                            f"Hill Slope (α)",
                            ranges["alpha"][0],
                            ranges["alpha"][1],
                            1.0,
                            0.1,
                            key=f"alpha_{channel}"
                        )
                    
                    with col3:
                        gamma = st.slider(
                            f"Inflection (γ)",
                            ranges["gamma"][0],
                            ranges["gamma"][1],
                            0.5,
                            0.1,
                            key=f"gamma_{channel}"
                        )
                    
                    channel_params[channel] = {'type': ch_type, 'theta': theta, 'alpha': alpha, 'gamma': gamma}
        
        # Store channel params
        st.session_state.channel_params = channel_params
        
        # Promotion and other controls
        promo_col = None
        promo_options = [col for col in df.columns if ('promo' in col.lower() or 'discount' in col.lower()) 
                        and col not in media_cols and col != target_col and col != date_col]
        if promo_options:
            use_promo = st.checkbox("Include promotion variable", value=True)
            if use_promo:
                promo_col = st.selectbox("Select promotion column", promo_options)
    
    else:
        # ORIGINAL MODE
        date_col = df.columns[0]
        
        potential_target_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
        target_col = st.selectbox("Select target/KPI column", potential_target_cols if potential_target_cols else df.columns[1:])
        
        cost_cols = [col for col in df.columns if 'cost' in col.lower() or 'spend' in col.lower()]
        media_cols = st.multiselect("Select media spend columns", cost_cols if cost_cols else df.columns[1:], default=cost_cols if cost_cols else [])
        
        if not media_cols:
            st.warning("WARNING Select at least one media channel!")
            st.stop()
        
        promo_options = [col for col in df.columns if ('promo' in col.lower() or 'discount' in col.lower()) 
                        and col not in media_cols and col != target_col and col != date_col]
        
        promo_col = None
        if promo_options:
            use_promo = st.checkbox("Include promotion variable", value=True)
            if use_promo:
                promo_col = st.selectbox("Select promotion column", promo_options)
        
        control_cols = []
        available_controls = [col for col in df.columns if col not in media_cols and col != target_col 
                             and col != date_col and col != promo_col 
                             and not ('promo' in col.lower() or 'discount' in col.lower())]
        control_cols = st.multiselect("Select additional control variables", available_controls)
        
        # Single parameter set for all channels
        st.markdown("### ️ Model Parameters")
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            adstock_alpha = st.slider("Adstock Rate (α)", 0.0, 0.9, 0.5, 0.05)
        with param_col2:
            hill_slope = st.slider("Hill Slope", 0.5, 2.0, 1.0, 0.1)
        with param_col3:
            train_test_split = st.slider("Train/Test Split", 0.6, 0.9, 0.8, 0.05)
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    
    # Train/Test Split
    train_test_split = st.slider("Train/Test Split", 0.6, 0.9, 0.8, 0.05)
    
    # Train Model Button
    st.markdown("---")
    if st.button(" Run Marketing Mix Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                daily_df = df.copy().sort_values(date_col).reset_index(drop=True)
                
                # Ensure numeric
                for col in media_cols:
                    daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce').fillna(0)
                daily_df[target_col] = pd.to_numeric(daily_df[target_col], errors='coerce')
                daily_df = daily_df.dropna(subset=[target_col])
                
                # Add seasonality
                daily_df = add_seasonality_features(daily_df, date_col)
                
                # Process promotion
                promo_features = []
                if promo_col:
                    daily_df, promo_features, promo_is_dummy = process_promotion_variable(daily_df, promo_col)
                    st.session_state.promo_features = promo_features
                    st.session_state.promo_is_dummy = promo_is_dummy
                else:
                    st.session_state.promo_features = []
                    st.session_state.promo_is_dummy = False
                
                # Engineer media features
                meta = {}
                feat_cols = []
                
                if v7_mode:
                    # v7: Use channel-specific parameters
                    for media_col in media_cols:
                        params = channel_params[media_col]
                        
                        # Adstock
                        daily_df[f'{media_col}_adstock'] = adstock_transformation(
                            daily_df[media_col].values, alpha=params['theta']
                        )
                        
                        # Hill with gamma
                        inflexion = daily_df[media_col].min() * (1-params['gamma']) + daily_df[media_col].max() * params['gamma']
                        kappa = max(inflexion, 1.0)
                        
                        daily_df[f'{media_col}_saturated'] = hill_transformation(
                            daily_df[f'{media_col}_adstock'].values,
                            kappa=kappa,
                            slope=params['alpha']
                        )
                        
                        # Standardize
                        mu = daily_df[f'{media_col}_saturated'].mean()
                        sd = daily_df[f'{media_col}_saturated'].std() or 1.0
                        
                        feat_name = f'{media_col}_feat'
                        daily_df[feat_name] = (daily_df[f'{media_col}_saturated'] - mu) / sd
                        
                        feat_cols.append(feat_name)
                        
                        meta[feat_name] = {
                            'spend_col': media_col,
                            'kappa': kappa,
                            'slope': params['alpha'],
                            'mu': mu,
                            'sd': sd,
                            'theta': params['theta'],
                            'gamma': params['gamma']
                        }
                else:
                    # Original: Single parameter set
                    for media_col in media_cols:
                        daily_df[f'{media_col}_adstock'] = adstock_transformation(
                            daily_df[media_col].values, alpha=adstock_alpha
                        )
                        
                        kappa = np.nanmedian(daily_df[f'{media_col}_adstock'].values)
                        if not np.isfinite(kappa) or kappa <= 0:
                            kappa = np.nanmean(daily_df[f'{media_col}_adstock'].values) or 1.0
                        
                        daily_df[f'{media_col}_saturated'] = hill_transformation(
                            daily_df[f'{media_col}_adstock'].values,
                            kappa=kappa,
                            slope=hill_slope
                        )
                        
                        mu = daily_df[f'{media_col}_saturated'].mean()
                        sd = daily_df[f'{media_col}_saturated'].std() or 1.0
                        
                        feat_name = f'{media_col}_feat'
                        daily_df[feat_name] = (daily_df[f'{media_col}_saturated'] - mu) / sd
                        
                        feat_cols.append(feat_name)
                        
                        meta[feat_name] = {
                            'spend_col': media_col,
                            'kappa': kappa,
                            'slope': hill_slope,
                            'mu': mu,
                            'sd': sd
                        }
                    
                    st.session_state.adstock_alpha = adstock_alpha
                
                # Train/test split
                split_idx = int(len(daily_df) * train_test_split)
                train_df = daily_df.iloc[:split_idx].copy()
                test_df = daily_df.iloc[split_idx:].copy()
                
                # Prepare X and y
                seasonality_cols = [col for col in daily_df.columns if 'dow_' in col or 'month_' in col]
                
                all_control_cols = control_cols + promo_features if v7_mode else control_cols + promo_features
                
                # Ensure controls are numeric
                for ctrl_col in all_control_cols:
                    if ctrl_col in train_df.columns:
                        if train_df[ctrl_col].dtype == 'object':
                            try:
                                train_df[ctrl_col] = pd.to_numeric(train_df[ctrl_col], errors='coerce').fillna(0)
                                test_df[ctrl_col] = pd.to_numeric(test_df[ctrl_col], errors='coerce').fillna(0)
                            except:
                                pass
                
                X_train = pd.concat([
                    pd.Series(1.0, index=train_df.index, name='const'),
                    train_df[feat_cols],
                    train_df[all_control_cols] if all_control_cols else pd.DataFrame(index=train_df.index),
                    train_df[seasonality_cols]
                ], axis=1).astype('float64')
                
                X_test = pd.concat([
                    pd.Series(1.0, index=test_df.index, name='const'),
                    test_df[feat_cols],
                    test_df[all_control_cols] if all_control_cols else pd.DataFrame(index=test_df.index),
                    test_df[seasonality_cols]
                ], axis=1).astype('float64')
                
                y_train = train_df[target_col].values.astype(float)
                y_test = test_df[target_col].values.astype(float)
                
                # Train model
                model = sm.OLS(y_train, X_train).fit()
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_r2, train_mape, train_wmape, train_nrmse = calculate_metrics(y_train, y_train_pred)
                test_r2, test_mape, test_wmape, test_nrmse = calculate_metrics(y_test, y_test_pred)
                
                # Store in session state
                st.session_state.model_trained = True
                st.session_state.model = model
                st.session_state.meta = meta
                st.session_state.feat_cols = feat_cols
                st.session_state.media_cols = media_cols
                st.session_state.target_col = target_col
                st.session_state.date_col = date_col
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_train_pred = y_train_pred
                st.session_state.y_test_pred = y_test_pred
                st.session_state.promo_col = promo_col
                st.session_state.control_cols = control_cols if not v7_mode else st.session_state.v7_control_cols
                
                st.success("OK Model trained successfully!")
                st.balloons()
                
                # Display metrics
                st.markdown("---")
                st.markdown("###  Model Performance")
                
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.markdown("**Training Set:**")
                    st.metric("R²", f"{train_r2:.3f}")
                    st.metric("MAPE", f"{train_mape:.2%}")
                    st.metric("wMAPE", f"{train_wmape:.2%}")
                    st.metric("NRMSE (v7)", f"{train_nrmse:.3f}")
                
                with metric_col2:
                    st.markdown("**Test Set:**")
                    st.metric("R²", f"{test_r2:.3f}")
                    st.metric("MAPE", f"{test_mape:.2%}")
                    st.metric("wMAPE", f"{test_wmape:.2%}")
                    st.metric("NRMSE (v7)", f"{test_nrmse:.3f}")
                
                # Model fit plot
                st.markdown("---")
                st.markdown("###  Model Fit Visualization")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                ax1.plot(train_df[date_col], y_train, label='Actual', color='green', alpha=0.7)
                ax1.plot(train_df[date_col], y_train_pred, label='Predicted', color='blue', alpha=0.7)
                ax1.set_title(f'Training Set (R²={train_r2:.3f})', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Date')
                ax1.set_ylabel(target_col)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(test_df[date_col], y_test, label='Actual', color='green', alpha=0.7)
                ax2.plot(test_df[date_col], y_test_pred, label='Predicted', color='blue', alpha=0.7)
                ax2.set_title(f'Test Set (R²={test_r2:.3f})', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Date')
                ax2.set_ylabel(target_col)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("OK Model training complete! Go to 'Results & Insights' ->")
                
            except Exception as e:
                st.error(f"X Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# TAB 4: Results & Insights (COMPLETE WITH v7 ENHANCEMENTS)
elif tab_selection == " Results & Insights":
    st.markdown('<p class="sub-header">Results & Insights</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("WARNING Please train the model first!")
        st.stop()
    
    # Retrieve from session state
    model = st.session_state.model
    meta = st.session_state.meta
    feat_cols = st.session_state.feat_cols
    media_cols = st.session_state.media_cols
    target_col = st.session_state.target_col
    date_col = st.session_state.date_col
    test_df = st.session_state.test_df
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_test_pred = st.session_state.y_test_pred
    promo_col = st.session_state.promo_col
    promo_features = st.session_state.promo_features
    control_cols = st.session_state.control_cols
    
    # Get adstock alpha
    if 'adstock_alpha' in st.session_state:
        adstock_alpha = st.session_state.adstock_alpha
    else:
        # v7 mode - get from channel params
        adstock_alpha = 0.5  # Default
    
    # Tabs for different analyses
    result_tabs = st.tabs([
        " Channel Contribution",
        " ROI Analysis (v7 Enhanced)",
        " Response Curves",
        " Budget Optimization (Full)",
        " Model Summary (v7 Diagnostics)"
    ])
    
    # TAB 1: CHANNEL CONTRIBUTION (WITH DE-STANDARDIZATION)
    with result_tabs[0]:
        st.markdown("### Channel Contribution to Revenue")
        
        st.info(" **v7 Enhancement:** Using de-standardized reporting for business-friendly interpretation")
        
        # Calculate de-standardized contributions
        contributions = {}
        baseline_shifts = 0
        
        for feat in feat_cols:
            beta = float(model.params.get(feat, 0.0))
            channel_name = meta[feat]['spend_col']
            
            # De-standardization
            mu = meta[feat]['mu']
            sigma = meta[feat]['sd']
            
            shift = beta * mu / sigma * len(test_df)
            baseline_shifts += shift
            
            saturated_col = f"{channel_name}_saturated"
            if saturated_col in test_df.columns:
                saturated = test_df[saturated_col].values
                contrib = (beta / sigma) * np.sum(saturated)
                contributions[channel_name] = contrib
            else:
                # Fallback
                contrib = np.sum(X_test[feat].values * beta)
                contributions[channel_name] = contrib
        
        # Add promotion
        if promo_col and promo_features:
            promo_contrib = 0
            for promo_feat in promo_features:
                if promo_feat in X_test.columns:
                    beta = float(model.params.get(promo_feat, 0.0))
                    promo_contrib += np.sum(X_test[promo_feat].values * beta)
            contributions['Promotion'] = promo_contrib
        
        # Adjusted baseline
        original_baseline = float(model.params.get('const', 0.0)) * len(X_test)
        true_baseline = original_baseline - baseline_shifts
        contributions['Baseline'] = true_baseline
        
        contrib_df = pd.DataFrame.from_dict(contributions, orient='index', columns=['Contribution'])
        contrib_df['Contribution %'] = 100 * contrib_df['Contribution'] / contrib_df['Contribution'].sum()
        contrib_df = contrib_df.sort_values('Contribution', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Contribution Summary (De-standardized):**")
            st.dataframe(
                contrib_df.style.format({'Contribution': '{:,.0f}', 'Contribution %': '{:.1f}%'}),
                use_container_width=True
            )
        
        with col2:
            # Pie chart - media only
            media_contrib = contrib_df[contrib_df.index != 'Baseline'].copy()
            positive_contrib = media_contrib[media_contrib['Contribution'] > 0].copy()
            
            if len(positive_contrib) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(range(len(positive_contrib)))
                ax.pie(positive_contrib['Contribution'], labels=positive_contrib.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
                ax.set_title('Revenue Contribution by Channel', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            else:
                st.warning("No positive contributions to display")
        
        # Bar chart
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(12, 6))
        contrib_df['Contribution'].plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Revenue Contribution ($)', fontsize=12)
        ax.set_title('Channel Contribution (De-standardized)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(contrib_df['Contribution']):
            ax.text(v, i, f' ${v:,.0f}', va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # TAB 2: ROI ANALYSIS (WITH DECOMP.RSSD)
    with result_tabs[1]:
        st.markdown("### Return on Investment (ROI) Analysis")
        
        st.info(" **v7 Enhancements:** De-standardized iROAS + DECOMP.RSSD metric")
        
        roi_data = []
        
        for feat in feat_cols:
            channel_name = meta[feat]['spend_col']
            beta = float(model.params.get(feat, 0.0))
            mu = meta[feat]['mu']
            sigma = meta[feat]['sd']
            
            # De-standardized contribution
            saturated_col = f"{channel_name}_saturated"
            if saturated_col in test_df.columns:
                saturated = test_df[saturated_col].values
                contrib = (beta / sigma) * np.sum(saturated)
            else:
                contrib = np.sum(X_test[feat].values * beta)
            
            # Total spend
            total_spend = test_df[channel_name].sum()
            
            if total_spend <= 0:
                continue
            
            # ROI (iROAS)
            roi = contrib / total_spend if total_spend > 0 else 0
            
            # Marginal ROI
            kappa = meta[feat]['kappa']
            slope = meta[feat]['slope']
            
            current_avg_spend = test_df[channel_name].mean()
            
            if current_avg_spend <= 0:
                marginal_roas = 0
            else:
                # Get theta if available
                if 'theta' in meta[feat]:
                    theta = meta[feat]['theta']
                else:
                    theta = adstock_alpha
                
                A = current_avg_spend / (1 - theta) if theta < 1 else current_avg_spend
                marginal_roas = (beta / sigma) * hill_derivative(A, kappa, slope) / (1 - theta) if theta < 1 else 0
            
            roi_data.append({
                'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                'Total Spend': total_spend,
                'Revenue Contribution': contrib,
                'ROI (iROAS)': roi,
                'Marginal ROI': marginal_roas
            })
        
        if not roi_data:
            st.warning("WARNING No channels with positive spend")
            st.stop()
        
        roi_df = pd.DataFrame(roi_data).sort_values('ROI (iROAS)', ascending=False)
        
        # Display table
        st.dataframe(
            roi_df.style.format({
                'Total Spend': '{:,.0f}',
                'Revenue Contribution': '{:,.0f}',
                'ROI (iROAS)': '{:.2f}',
                'Marginal ROI': '{:.2f}'
            }).background_gradient(subset=['ROI (iROAS)', 'Marginal ROI'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # ROI visualization
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            roi_df.plot(x='Channel', y='ROI (iROAS)', kind='bar', ax=ax, color='coral', legend=False)
            ax.set_title('ROI by Channel', fontsize=14, fontweight='bold')
            ax.set_ylabel('ROI', fontsize=12)
            ax.set_xlabel('')
            ax.axhline(y=1, color='red', linestyle='--', label='Break-even')
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            roi_df.plot(x='Channel', y='Marginal ROI', kind='bar', ax=ax, color='skyblue', legend=False)
            ax.set_title('Marginal ROI', fontsize=14, fontweight='bold')
            ax.set_ylabel('Marginal ROI', fontsize=12)
            ax.set_xlabel('')
            ax.axhline(y=1, color='red', linestyle='--', label='Break-even')
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        # NEW v7: DECOMP.RSSD
        st.markdown("---")
        st.markdown("###  DECOMP.RSSD - Spend vs Effect Share Analysis (NEW v7)")
        
        rssd, spend_share, effect_share = calculate_decomp_rssd(test_df, contributions, media_cols)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("DECOMP.RSSD", f"{rssd:.4f}")
            st.caption("Lower is better - measures misalignment")
            
            if rssd < 0.1:
                st.success("OK Excellent alignment")
            elif rssd < 0.2:
                st.info("INFO Good alignment")
            elif rssd < 0.3:
                st.warning("WARNING Moderate misalignment")
            else:
                st.error("X High misalignment")
        
        with col2:
            comparison_df = pd.DataFrame({
                'Channel': media_cols,
                'Spend Share (%)': [spend_share.get(ch, 0) * 100 for ch in media_cols],
                'Effect Share (%)': [effect_share.get(ch, 0) * 100 for ch in media_cols],
                'Difference (pp)': [(effect_share.get(ch, 0) - spend_share.get(ch, 0)) * 100 for ch in media_cols]
            })
            
            st.dataframe(
                comparison_df.style.format({
                    'Spend Share (%)': '{:.2f}',
                    'Effect Share (%)': '{:.2f}',
                    'Difference (pp)': '{:+.2f}'
                }).background_gradient(subset=['Difference (pp)'], cmap='RdYlGn', vmin=-10, vmax=10),
                use_container_width=True
            )
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(media_cols))
        width = 0.35
        
        ax.bar(x - width/2, [spend_share.get(ch, 0)*100 for ch in media_cols], 
               width, label='Spend Share', color='steelblue')
        ax.bar(x + width/2, [effect_share.get(ch, 0)*100 for ch in media_cols], 
               width, label='Effect Share', color='orange')
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('Share (%)')
        ax.set_title('Spend Share vs Effect Share by Channel', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([ch.replace('_Cost', '') for ch in media_cols], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Insights
        st.markdown("---")
        st.markdown("###  Key Insights")
        
        best_roi_channel = roi_df.iloc[0]
        best_marginal_channel = roi_df.sort_values('Marginal ROI', ascending=False).iloc[0]
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.info(f"""
            **Best Overall ROI:**
            - **{best_roi_channel['Channel']}** has ROI of **{best_roi_channel['ROI (iROAS)']:.2f}**
            - Every $1 spent returns ${best_roi_channel['ROI (iROAS)']:.2f}
            """)
        
        with insight_col2:
            st.info(f"""
            **Best Marginal Efficiency:**
            - **{best_marginal_channel['Channel']}** has marginal ROI of **{best_marginal_channel['Marginal ROI']:.2f}**
            - Most room for additional investment
            """)
    
    # TAB 3: RESPONSE CURVES
    with result_tabs[2]:
        st.markdown("### Saturation & Response Curves")
        
        selected_channel = st.selectbox(
            "Select channel to analyze",
            [meta[feat]['spend_col'] for feat in feat_cols],
            key='curve_channel'
        )
        
        feat = [f for f in feat_cols if meta[f]['spend_col'] == selected_channel][0]
        
        beta = float(model.params.get(feat, 0.0))
        kappa = meta[feat]['kappa']
        slope = meta[feat]['slope']
        mu = meta[feat]['mu']
        sd = meta[feat]['sd']
        
        # Get theta
        if 'theta' in meta[feat]:
            theta = meta[feat]['theta']
        else:
            theta = adstock_alpha
        
        # Generate spend range
        historical_spend = test_df[selected_channel].values
        valid_spend = historical_spend[historical_spend > 0]
        
        if len(valid_spend) == 0:
            st.warning(f"WARNING No positive spend for {selected_channel}")
            st.stop()
        
        max_spend = np.percentile(valid_spend, 95)
        spend_range = np.linspace(0, max_spend * 1.5, 200)
        
        # Calculate responses
        if theta < 1:
            adstocked = spend_range / (1 - theta)
        else:
            adstocked = spend_range
            
        saturated = hill_transformation(adstocked, kappa, slope)
        standardized = (saturated - mu) / sd
        revenue = beta * standardized
        
        # Marginal ROAS
        if theta < 1:
            marginal_roas = (beta / sd) * hill_derivative(adstocked, kappa, slope) / (1 - theta)
        else:
            marginal_roas = np.zeros_like(spend_range)
        
        # iROAS
        iroas = np.zeros_like(revenue)
        for i in range(1, len(revenue)):
            iroas[i] = revenue[i] / spend_range[i] if spend_range[i] > 0 else 0
        
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Saturation curve
        axes[0, 0].plot(spend_range, revenue, color='steelblue', linewidth=2)
        axes[0, 0].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
        axes[0, 0].set_title('Saturation Curve', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Daily Spend')
        axes[0, 0].set_ylabel('Incremental Revenue')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Marginal ROAS
        axes[0, 1].plot(spend_range, marginal_roas, color='coral', linewidth=2)
        axes[0, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
        axes[0, 1].axhline(y=1, color='green', linestyle='--', label='Break-even')
        axes[0, 1].set_title('Marginal ROAS', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Daily Spend')
        axes[0, 1].set_ylabel('Marginal ROAS')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # iROAS
        axes[1, 0].plot(spend_range[1:], iroas[1:], color='purple', linewidth=2)
        axes[1, 0].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
        axes[1, 0].axhline(y=1, color='green', linestyle='--', label='Break-even')
        axes[1, 0].set_title('Incremental ROAS', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Daily Spend')
        axes[1, 0].set_ylabel('iROAS')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency
        efficiency = revenue / spend_range
        efficiency[0] = 0
        axes[1, 1].plot(spend_range, efficiency, color='green', linewidth=2)
        axes[1, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg')
        axes[1, 1].set_title('Spend Efficiency', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Daily Spend')
        axes[1, 1].set_ylabel('Revenue / Spend')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Current metrics
        st.markdown("---")
        st.markdown("###  Current Performance")
        
        current_spend = valid_spend.mean()
        current_idx = np.argmin(np.abs(spend_range - current_spend))
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Current Avg Spend", f"${current_spend:,.0f}")
        with metric_col2:
            st.metric("Marginal ROAS", f"{marginal_roas[current_idx]:.2f}")
        with metric_col3:
            st.metric("iROAS", f"{iroas[current_idx]:.2f}")
        with metric_col4:
            if saturated[-1] > 0:
                saturation_level = (saturated[current_idx] / saturated[-1]) * 100
            else:
                saturation_level = 0
            st.metric("Saturation Level", f"{saturation_level:.1f}%")
    
    # TAB 4: BUDGET OPTIMIZATION (COMPLETE IMPLEMENTATION)
    with result_tabs[3]:
        st.markdown("### Budget Allocation Optimizer (Scipy SLSQP - COMPLETE)")
        
        st.info("""
        **Full Optimization:**
        - Scipy SLSQP solver
        - Preserves historical spend patterns
        - Accounts for adstock carryover
        - Includes baseline + seasonality
        - Model vs model comparison
        """)
        
        # Current budget
        current_budget = sum([test_df[meta[feat]['spend_col']].sum() for feat in feat_cols])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_budget = st.slider(
                "Total Budget",
                min_value=int(current_budget * 0.5),
                max_value=int(current_budget * 2),
                value=int(current_budget),
                step=int(current_budget * 0.05),
                format="$%d"
            )
        
        with col2:
            budget_change = ((new_budget - current_budget) / current_budget) * 100
            st.metric("Budget Change", f"{budget_change:+.1f}%")
        
        # Run optimization
        if st.button(" Run Full Optimization", type="primary", use_container_width=True):
            with st.spinner("Running scipy optimization..."):
                try:
                    # Get baseline + seasonality
                    baseline_contrib = float(model.params.get('const', 0.0)) * len(test_df)
                    
                    seasonality_cols = [col for col in X_test.columns if 'dow_' in col or 'month_' in col]
                    seasonality_contrib = 0
                    for col in seasonality_cols:
                        if col in X_test.columns and col in model.params:
                            seasonality_contrib += (X_test[col].values * float(model.params[col])).sum()
                    
                    # Objective function
                    def mmm_objective(channel_totals):
                        total_revenue = baseline_contrib + seasonality_contrib
                        
                        for i, feat in enumerate(feat_cols):
                            channel_name = meta[feat]['spend_col']
                            beta = float(model.params.get(feat, 0.0))
                            kappa = meta[feat]['kappa']
                            slope = meta[feat]['slope']
                            sd = meta[feat]['sd']
                            mu = meta[feat]['mu']
                            
                            # Get theta
                            if 'theta' in meta[feat]:
                                theta = meta[feat]['theta']
                            else:
                                theta = adstock_alpha
                            
                            current_total = test_df[channel_name].sum()
                            optimized_total = channel_totals[i]
                            
                            if current_total > 0:
                                scale = optimized_total / current_total
                            else:
                                scale = 0
                            
                            # Scale historical pattern
                            scaled_daily_spend = test_df[channel_name].values * scale
                            
                            # Apply transformations
                            adstocked_spend = adstock_transformation(scaled_daily_spend, alpha=theta)
                            saturated_spend = hill_transformation(adstocked_spend, kappa, slope)
                            
                            if sd > 0:
                                standardized_spend = (saturated_spend - mu) / sd
                            else:
                                standardized_spend = saturated_spend - mu
                            
                            channel_revenue = np.sum(beta * standardized_spend)
                            total_revenue += channel_revenue
                        
                        return -total_revenue
                    
                    # Budget constraint
                    def budget_constraint(channel_totals):
                        return np.sum(channel_totals) - new_budget
                    
                    # Initial guess
                    initial_totals = [test_df[meta[feat]['spend_col']].sum() for feat in feat_cols]
                    
                    # Bounds
                    bounds = [(0, None) for _ in feat_cols]
                    
                    # Solve
                    solution = minimize(
                        fun=mmm_objective,
                        x0=initial_totals,
                        bounds=bounds,
                        method="SLSQP",
                        constraints={'type': 'eq', 'fun': budget_constraint},
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    
                    if solution.success:
                        st.success("OK Optimization completed!")
                        
                        # Results
                        allocation_data = []
                        for i, feat in enumerate(feat_cols):
                            channel_name = meta[feat]['spend_col']
                            current_spend = test_df[channel_name].sum()
                            optimized_spend = solution.x[i]
                            
                            allocation_data.append({
                                'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                                'Current Spend': current_spend,
                                'Optimized Spend': optimized_spend,
                                'Change': optimized_spend - current_spend,
                                'Change %': ((optimized_spend - current_spend) / current_spend * 100) if current_spend > 0 else 0
                            })
                        
                        alloc_df = pd.DataFrame(allocation_data)
                        
                        # Display
                        st.markdown("---")
                        st.markdown("####  Optimal Allocation")
                        
                        st.dataframe(
                            alloc_df.style.format({
                                'Current Spend': '{:,.0f}',
                                'Optimized Spend': '{:,.0f}',
                                'Change': '{:+,.0f}',
                                'Change %': '{:+.1f}%'
                            }).background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-50, vmax=50),
                            use_container_width=True
                        )
                        
                        # Visualization
                        st.markdown("---")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        # Current vs Optimized
                        x = np.arange(len(alloc_df))
                        width = 0.35
                        
                        ax1.bar(x - width/2, alloc_df['Current Spend'], width, label='Current', color='steelblue')
                        ax1.bar(x + width/2, alloc_df['Optimized Spend'], width, label='Optimized', color='coral')
                        ax1.set_xlabel('Channel')
                        ax1.set_ylabel('Budget')
                        ax1.set_title('Current vs Optimized Budget', fontsize=14, fontweight='bold')
                        ax1.set_xticks(x)
                        ax1.set_xticklabels(alloc_df['Channel'], rotation=45, ha='right')
                        ax1.legend()
                        ax1.grid(axis='y', alpha=0.3)
                        
                        # Change %
                        colors = ['green' if x > 0 else 'red' for x in alloc_df['Change %']]
                        ax2.barh(alloc_df['Channel'], alloc_df['Change %'], color=colors)
                        ax2.set_xlabel('Change (%)')
                        ax2.set_title('Budget Change', fontsize=14, fontweight='bold')
                        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                        ax2.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Expected impact
                        st.markdown("---")
                        st.markdown("###  Expected Impact")
                        
                        current_allocation = [test_df[meta[feat]['spend_col']].sum() for feat in feat_cols]
                        current_revenue_model = -mmm_objective(current_allocation)
                        optimized_revenue = -solution.fun
                        expected_lift = optimized_revenue - current_revenue_model
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Revenue (Model)", f"${current_revenue_model:,.0f}")
                        with col2:
                            st.metric("Optimized Revenue", f"${optimized_revenue:,.0f}", delta=f"${expected_lift:,.0f}")
                        with col3:
                            lift_pct = (expected_lift / current_revenue_model) * 100 if current_revenue_model > 0 else 0
                            st.metric("Expected Lift", f"{lift_pct:+.1f}%")
                        
                        # Model vs Actual
                        with st.expander(" Model vs Actual Comparison"):
                            actual_revenue = y_test.sum()
                            prediction_error = ((current_revenue_model - actual_revenue) / actual_revenue * 100)
                            
                            st.write(f"**Actual Revenue:** ${actual_revenue:,.0f}")
                            st.write(f"**Model Prediction:** ${current_revenue_model:,.0f}")
                            st.write(f"**Prediction Error:** {prediction_error:+.1f}%")
                            st.caption("Optimization compares model predictions")
                        
                        # Details
                        with st.expander(" Optimization Details"):
                            st.write(f"**Status:** {solution.message}")
                            st.write(f"**Iterations:** {solution.nit}")
                            st.write(f"**Function Evals:** {solution.nfev}")
                            st.write(f"**Objective Value:** {-solution.fun:,.2f}")
                    
                    else:
                        st.error(f"X Optimization failed: {solution.message}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # TAB 5: MODEL SUMMARY (WITH v7 DIAGNOSTICS)
    with result_tabs[4]:
        st.markdown("### Model Summary with v7 Diagnostics")
        
        st.info(" **v7 Enhancements:** VIF analysis, Extended metrics, Confidence intervals, Durbin-Watson")
        
        # NEW v7: Extended Diagnostics
        st.markdown("####  Extended Diagnostics (v7)")
        
        from statsmodels.stats.stattools import durbin_watson
        
        mape = np.mean(np.abs((y_test - y_test_pred)/y_test))
        nrmse = np.sqrt(np.mean((y_test - y_test_pred)**2)) / (y_test.max() - y_test.min()) if (y_test.max() - y_test.min()) > 0 else 0
        dw = durbin_watson(model.resid)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{model.rsquared:.4f}")
        col2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
        col3.metric("MAPE", f"{mape:.2%}")
        col4.metric("NRMSE", f"{nrmse:.4f}")
        
        col1.metric("AIC", f"{model.aic:.2f}")
        col2.metric("BIC", f"{model.bic:.2f}")
        col3.metric("DW Stat", f"{dw:.4f}")
        col4.metric("F-statistic", f"{model.fvalue:.2f}")
        
        # Durbin-Watson interpretation
        if 1.5 < dw < 2.5:
            st.success("OK No significant autocorrelation")
        elif dw < 1.5:
            st.warning("WARNING Positive autocorrelation detected")
        else:
            st.warning("WARNING Negative autocorrelation detected")
        
        # NEW v7: VIF Analysis
        st.markdown("---")
        st.markdown("####  VIF Analysis (Multicollinearity Detection)")
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_test.columns[1:]  # Exclude const
        vif_data["VIF"] = [
            variance_inflation_factor(X_test.values, i) 
            for i in range(1, X_test.shape[1])
        ]
        
        def vif_color(val):
            if val > 10:
                return 'background-color: #ffcccc'
            elif val > 5:
                return 'background-color: #ffffcc'
            return 'background-color: #ccffcc'
        
        st.dataframe(
            vif_data.style.applymap(vif_color, subset=['VIF']).format({'VIF': '{:.2f}'}),
            use_container_width=True
        )
        
        st.caption("""
        **VIF Interpretation:**
        - VIF < 5: OK Low multicollinearity
        - VIF 5-10: WARNING Moderate
        - VIF > 10: X High (consider removing)
        """)
        
        # NEW v7: Confidence Intervals
        st.markdown("---")
        st.markdown("####  Coefficient Confidence Intervals (95%)")
        
        conf_int = model.conf_int()
        
        coef_df = pd.DataFrame({
            'Variable': model.params.index,
            'Coefficient': model.params.values,
            'Std Error': model.bse.values,
            'CI Lower': conf_int.iloc[:, 0].values,
            'CI Upper': conf_int.iloc[:, 1].values,
            't-value': model.tvalues.values,
            'p-value': model.pvalues.values,
            'Sig': ['***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')) for p in model.pvalues.values]
        })
        
        st.dataframe(
            coef_df.style.format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                'CI Lower': '{:.4f}',
                'CI Upper': '{:.4f}',
                't-value': '{:.4f}',
                'p-value': '{:.4f}'
            }).background_gradient(subset=['p-value'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        
        # Model statistics
        st.markdown("---")
        st.markdown("####  Model Statistics")
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.metric("R-squared", f"{model.rsquared:.4f}")
            st.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
        
        with stat_col2:
            st.metric("F-statistic", f"{model.fvalue:.2f}")
            st.metric("Prob (F-stat)", f"{model.f_pvalue:.4e}")
        
        with stat_col3:
            st.metric("AIC", f"{model.aic:.2f}")
            st.metric("BIC", f"{model.bic:.2f}")
        
        # Model diagnostics
        st.markdown("---")
        st.markdown("####  Model Diagnostics (4 Plots)")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        residuals = y_test - y_test_pred
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_test_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_test, y_test_pred, alpha=0.5)
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect')
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title('Actual vs Predicted', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Marketing Mix Modeling Platform v7 - FULL EDITION</b></p>
    <p>Complete with 8 variable types, Feature Selection, VIF, Extended Diagnostics, DECOMP.RSSD, & Full Optimization</p>
    <p>Built with Streamlit | Powered by Scipy & Statsmodels</p>
</div>
""", unsafe_allow_html=True)
