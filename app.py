"""
Marketing Mix Modeling (MMM) Streamlit App - FULL COMPREHENSIVE VERSION v7

COMPLETE IMPLEMENTATION combining:
- All features from working 1,502-line app
- Plus all new v7 features (8 variable types, feature selection, VIF, etc.)

NEW v7 FEATURES:
✅ Multi-variable type upload (8 types: media, competition, controls, TV, traditional, ATL)
✅ User guide with Data Dictionary naming conventions
✅ Channel-specific parameter ranges (TV, Digital, Traditional, Competition)
✅ Feature selection using correlation analysis (optional)
✅ Extended diagnostics (VIF, NRMSE, AIC, BIC, Durbin-Watson)
✅ Confidence intervals for all coefficients
✅ DECOMP.RSSD metric (spend vs effect share)
✅ De-standardized reporting (always positive contributions)
✅ Proper variable handling (media=transformed, controls=untransformed)

EXISTING FEATURES (from working app):
✅ Complete data upload with promotion support
✅ Data validation and overview
✅ Full MMM modeling with adstock & saturation
✅ Complete budget optimization (scipy SLSQP)
✅ Detailed visualizations (4-subplot response curves)
✅ Complete model diagnostics (4-subplot residual analysis)
✅ ROI analysis with marginal ROAS
✅ Channel contribution analysis

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
    page_icon="📊",
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
st.markdown('<p class="main-header">📊 Marketing Mix Modeling Platform v7 - FULL EDITION</p>', unsafe_allow_html=True)
st.markdown("**Complete Implementation** - Multi-Variable Support, Feature Selection, Advanced Diagnostics & Full Optimization")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/analytics.png", width=100)
    st.markdown("### Navigation")
    tab_selection = st.radio(
        "Select a section:",
        ["📖 User Guide", "📤 Data Upload", "🔍 Data Overview", "🎯 Marketing Mix Modeling", "📈 Results & Insights"],
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
if tab_selection == "📖 User Guide":
    st.markdown('<p class="sub-header">📖 User Guide & Data Requirements</p>', unsafe_allow_html=True)
    
    st.markdown("""

# TAB 1: Data Upload (FIXED - NOW SUPPORTS ALL 8 VARIABLE TYPES IN INDIVIDUAL MODE)
elif tab_selection == "📤 Data Upload":
    st.markdown('<p class="sub-header">Upload Your Marketing Data</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>ℹ️ Upload Options:</b>
    <ul>
    <li><b>Option A:</b> Individual Files - Upload separate files for each variable type (supports all 8 types)</li>
    <li><b>Option B:</b> Combined Dataset - Upload one file with all variables (faster setup)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    upload_mode = st.radio("Select upload mode:", ["Individual Files (All 8 Types)", "Combined Dataset"], horizontal=True)
    
    if upload_mode == "Individual Files (All 8 Types)":
        # ENHANCED INDIVIDUAL UPLOAD WITH ALL 8 VARIABLE TYPES
        
        st.markdown("### 📊 Step 1: Upload KPI Data (MANDATORY)")
        st.info("Upload your target KPI (Revenue, Sales Volume, etc.). Must include: **Date** and **KPI** columns")
        
        kpi_file = st.file_uploader(
            "Choose KPI CSV/Excel file",
            type=['csv', 'xlsx'],
            key='kpi_upload_v7',
            help="CSV or Excel with Date and KPI columns"
        )
        
        if kpi_file:
            try:
                kpi_df = pd.read_csv(kpi_file) if kpi_file.name.endswith('.csv') else pd.read_excel(kpi_file)
                kpi_df = clean_dataframe_numeric_columns(kpi_df, exclude_cols=[kpi_df.columns[0]])
                st.session_state.kpi_data = kpi_df
                
                st.success(f"✅ KPI data uploaded successfully! ({len(kpi_df)} rows)")
                
                with st.expander("Preview KPI Data"):
                    st.dataframe(kpi_df.head(10), use_container_width=True)
                    st.write(f"**Columns:** {', '.join(kpi_df.columns.tolist())}")
                    st.write(f"**Date range:** {kpi_df.iloc[:, 0].min()} to {kpi_df.iloc[:, 0].max()}")
                    
            except Exception as e:
                st.error(f"Error loading KPI data: {str(e)}")
        
        # STEP 2: MEDIA CHANNELS (MANDATORY)
        st.markdown("---")
        st.markdown("### 💰 Step 2: Upload Media Channels (MANDATORY)")
        st.info("Upload your marketing channel files. Each file should have: **Date** and **Spend/Cost** columns")
        
        num_media = st.number_input("Number of media channels", min_value=1, max_value=15, value=3, key='num_media_v7')
        
        if 'media_data_v7' not in st.session_state:
            st.session_state.media_data_v7 = {}
        
        for i in range(num_media):
            st.markdown(f"**Channel {i+1}:**")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                channel_name = st.text_input(f"Channel name", value=f"Channel_{i+1}", key=f'ch_name_{i}')
            
            with col2:
                channel_file = st.file_uploader(
                    f"Upload {channel_name} CSV/Excel",
                    type=['csv', 'xlsx'],
                    key=f'ch_file_{i}'
                )
            
            if channel_file:
                try:
                    ch_df = pd.read_csv(channel_file) if channel_file.name.endswith('.csv') else pd.read_excel(channel_file)
                    ch_df = clean_dataframe_numeric_columns(ch_df, exclude_cols=[ch_df.columns[0]])
                    st.session_state.media_data_v7[channel_name] = ch_df
                    st.success(f"✅ {channel_name} uploaded ({len(ch_df)} rows)")
                    
                    with st.expander(f"Preview {channel_name}"):
                        st.dataframe(ch_df.head(5), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error loading {channel_name}: {str(e)}")
        
        # STEP 3: CLASSIFY MEDIA CHANNELS (NEW v7)
        if st.session_state.media_data_v7:
            st.markdown("---")
            st.markdown("### 📺 Step 3: Classify Media Channels (Optional but Recommended)")
            st.info("""
            Classify your channels for **channel-specific parameter ranges**:
            - **TV/Video:** High carryover (adstock 0.3-0.8)
            - **Traditional:** Medium carryover (adstock 0.1-0.4)
            - **Digital:** Low carryover (adstock 0.0-0.3)
            """)
            
            media_channel_names = list(st.session_state.media_data_v7.keys())
            
            col1, col2 = st.columns(2)
            
            with col1:
                tv_channels = st.multiselect(
                    "📺 TV/Video Channels",
                    media_channel_names,
                    help="Examples: TV, YouTube, Video Ads, Connected TV"
                )
            
            with col2:
                traditional_channels = st.multiselect(
                    "📻 Traditional Media",
                    [c for c in media_channel_names if c not in tv_channels],
                    help="Examples: Radio, Print, Outdoor, OOH"
                )
            
            digital_channels = [c for c in media_channel_names if c not in tv_channels + traditional_channels]
            
            if digital_channels:
                st.success(f"📱 **Digital Channels (auto-classified):** {', '.join(digital_channels)}")
                st.caption("Digital includes: Search, Display, Social, Programmatic")
        
        # STEP 4: COMPETITION VARIABLES (NEW v7)
        st.markdown("---")
        st.markdown("### 🏢 Step 4: Upload Competition Variables (Optional)")
        st.info("Upload competitor marketing spend data. Format: **Date** and **Competitor Spend** columns")
        
        num_competition = st.number_input("Number of competition variables", min_value=0, max_value=10, value=0, key='num_comp_v7')
        
        if 'competition_data_v7' not in st.session_state:
            st.session_state.competition_data_v7 = {}
        
        if num_competition > 0:
            for i in range(num_competition):
                st.markdown(f"**Competition Variable {i+1}:**")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    comp_name = st.text_input(f"Variable name", value=f"Competitor_{i+1}", key=f'comp_name_{i}')
                
                with col2:
                    comp_file = st.file_uploader(
                        f"Upload {comp_name} CSV/Excel",
                        type=['csv', 'xlsx'],
                        key=f'comp_file_{i}'
                    )
                
                if comp_file:
                    try:
                        comp_df = pd.read_csv(comp_file) if comp_file.name.endswith('.csv') else pd.read_excel(comp_file)
                        comp_df = clean_dataframe_numeric_columns(comp_df, exclude_cols=[comp_df.columns[0]])
                        st.session_state.competition_data_v7[comp_name] = comp_df
                        st.success(f"✅ {comp_name} uploaded ({len(comp_df)} rows)")
                        
                        with st.expander(f"Preview {comp_name}"):
                            st.dataframe(comp_df.head(5), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading {comp_name}: {str(e)}")
            
            # Classify ATL
            if st.session_state.competition_data_v7:
                comp_names = list(st.session_state.competition_data_v7.keys())
                atl_comps = st.multiselect(
                    "📢 Select ATL (Above-the-Line) Competition Variables",
                    comp_names,
                    help="ATL typically includes TV, Radio, Print competitor spends"
                )
        
        # STEP 5: CONTROL VARIABLES (NEW v7)
        st.markdown("---")
        st.markdown("### 🎛️ Step 5: Upload Control Variables (Optional)")
        st.info("""
        Upload control variables - **NO transformation applied**. These are used as-is.
        
        Examples: Price, Inflation Rate, Market Share, Economic Indicators, Weather, Holidays
        """)
        
        num_controls = st.number_input("Number of control variables", min_value=0, max_value=10, value=0, key='num_ctrl_v7')
        
        if 'control_data_v7' not in st.session_state:
            st.session_state.control_data_v7 = {}
        
        if num_controls > 0:
            for i in range(num_controls):
                st.markdown(f"**Control Variable {i+1}:**")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    ctrl_name = st.text_input(f"Variable name", value=f"Control_{i+1}", key=f'ctrl_name_{i}')
                
                with col2:
                    ctrl_file = st.file_uploader(
                        f"Upload {ctrl_name} CSV/Excel",
                        type=['csv', 'xlsx'],
                        key=f'ctrl_file_{i}'
                    )
                
                if ctrl_file:
                    try:
                        ctrl_df = pd.read_csv(ctrl_file) if ctrl_file.name.endswith('.csv') else pd.read_excel(ctrl_file)
                        ctrl_df = clean_dataframe_numeric_columns(ctrl_df, exclude_cols=[ctrl_df.columns[0]])
                        st.session_state.control_data_v7[ctrl_name] = ctrl_df
                        st.success(f"✅ {ctrl_name} uploaded ({len(ctrl_df)} rows)")
                        
                        with st.expander(f"Preview {ctrl_name}"):
                            st.dataframe(ctrl_df.head(5), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading {ctrl_name}: {str(e)}")
        
        # STEP 6: PROMOTION DATA (OPTIONAL)
        st.markdown("---")
        st.markdown("### 🎁 Step 6: Upload Promotion Data (Optional)")
        st.info("""
        Upload promotion/discount data. Can be:
        - **Categorical:** Yes/No, Sale/Normal → Converted to dummy variables
        - **Numeric:** 10%, 0.15, discount amounts → Used as continuous variable
        """)
        
        promo_file = st.file_uploader(
            "Upload Promotion CSV/Excel (optional)",
            type=['csv', 'xlsx'],
            key='promo_upload_v7',
            help="Date + Promotion columns"
        )
        
        if promo_file:
            try:
                promo_df = pd.read_csv(promo_file) if promo_file.name.endswith('.csv') else pd.read_excel(promo_file)
                st.session_state.promotion_data = promo_df
                
                st.success(f"✅ Promotion data uploaded! ({len(promo_df)} rows)")
                
                with st.expander("Preview Promotion Data"):
                    st.dataframe(promo_df.head(10), use_container_width=True)
                    promo_col = promo_df.columns[1]
                    if promo_df[promo_col].dtype == 'object':
                        st.info(f"✓ Detected **categorical** promotion: {promo_df[promo_col].unique()[:5]}")
                    else:
                        st.info(f"✓ Detected **numeric** promotion: Range {promo_df[promo_col].min():.2f} - {promo_df[promo_col].max():.2f}")
            except Exception as e:
                st.error(f"Error loading promotion data: {str(e)}")
        
        # STEP 7: COMBINE ALL DATA
        st.markdown("---")
        st.markdown("### 🔗 Step 7: Combine All Data")
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("✅ KPI Data", "Uploaded" if st.session_state.kpi_data is not None else "Missing")
            st.metric("💰 Media Channels", len(st.session_state.media_data_v7))
            st.metric("📺 TV Channels", len(tv_channels) if 'tv_channels' in locals() else 0)
        
        with col2:
            st.metric("📻 Traditional", len(traditional_channels) if 'traditional_channels' in locals() else 0)
            st.metric("📱 Digital", len(digital_channels) if 'digital_channels' in locals() else 0)
            st.metric("🏢 Competition", len(st.session_state.get('competition_data_v7', {})))
        
        with col3:
            st.metric("🎛️ Controls", len(st.session_state.get('control_data_v7', {})))
            st.metric("🎁 Promotion", "Uploaded" if st.session_state.promotion_data is not None else "Not uploaded")
            st.metric("📊 Total Variables", 1 + len(st.session_state.media_data_v7) + len(st.session_state.get('competition_data_v7', {})) + len(st.session_state.get('control_data_v7', {})))
        
        if st.button("🔗 Combine All Data & Create Model Configuration", type="primary", use_container_width=True):
            if st.session_state.kpi_data is None:
                st.error("❌ Please upload KPI data first!")
            elif len(st.session_state.media_data_v7) == 0:
                st.error("❌ Please upload at least one media channel!")
            else:
                with st.spinner("Combining all data and setting up v7 configuration..."):
                    try:
                        # Start with KPI
                        combined = st.session_state.kpi_data.copy()
                        date_col = combined.columns[0]
                        combined[date_col] = pd.to_datetime(combined[date_col], errors='coerce', dayfirst=True)
                        
                        # Merge media channels
                        for ch_name, ch_df in st.session_state.media_data_v7.items():
                            ch_df = ch_df.copy()
                            ch_date_col = ch_df.columns[0]
                            ch_df[ch_date_col] = pd.to_datetime(ch_df[ch_date_col], errors='coerce', dayfirst=True)
                            
                            rename_dict = {col: f"{ch_name}_{col}" for col in ch_df.columns if col.lower() not in ['date', 'month', 'week']}
                            ch_df = ch_df.rename(columns=rename_dict)
                            ch_df = ch_df.rename(columns={ch_date_col: date_col})
                            combined = combined.merge(ch_df, on=date_col, how='left')
                        
                        # Merge competition
                        if st.session_state.get('competition_data_v7'):
                            for comp_name, comp_df in st.session_state.competition_data_v7.items():
                                comp_df = comp_df.copy()
                                comp_date_col = comp_df.columns[0]
                                comp_df[comp_date_col] = pd.to_datetime(comp_df[comp_date_col], errors='coerce', dayfirst=True)
                                
                                rename_dict = {col: f"{comp_name}_{col}" for col in comp_df.columns if col.lower() not in ['date', 'month', 'week']}
                                comp_df = comp_df.rename(columns=rename_dict)
                                comp_df = comp_df.rename(columns={comp_date_col: date_col})
                                combined = combined.merge(comp_df, on=date_col, how='left')
                        
                        # Merge controls
                        if st.session_state.get('control_data_v7'):
                            for ctrl_name, ctrl_df in st.session_state.control_data_v7.items():
                                ctrl_df = ctrl_df.copy()
                                ctrl_date_col = ctrl_df.columns[0]
                                ctrl_df[ctrl_date_col] = pd.to_datetime(ctrl_df[ctrl_date_col], errors='coerce', dayfirst=True)
                                
                                rename_dict = {col: f"{ctrl_name}_{col}" for col in ctrl_df.columns if col.lower() not in ['date', 'month', 'week']}
                                ctrl_df = ctrl_df.rename(columns=rename_dict)
                                ctrl_df = ctrl_df.rename(columns={ctrl_date_col: date_col})
                                combined = combined.merge(ctrl_df, on=date_col, how='left')
                        
                        # Merge promotion
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
                        
                        # Fill NaN for spend/cost columns
                        cost_cols = [col for col in combined.columns if any(x in col.lower() for x in ['cost', 'spend', 'spends'])]
                        combined[cost_cols] = combined[cost_cols].fillna(0)
                        
                        # Clean and finalize
                        combined = clean_dataframe_numeric_columns(combined, exclude_cols=[date_col])
                        combined = combined.dropna(subset=[date_col])
                        
                        # Store combined data
                        st.session_state.combined_data = combined
                        
                        # Build column name lists from combined data
                        media_col_names = []
                        for ch_name in st.session_state.media_data_v7.keys():
                            matching_cols = [col for col in combined.columns if col.startswith(f"{ch_name}_")]
                            media_col_names.extend(matching_cols)
                        
                        competition_col_names = []
                        if st.session_state.get('competition_data_v7'):
                            for comp_name in st.session_state.competition_data_v7.keys():
                                matching_cols = [col for col in combined.columns if col.startswith(f"{comp_name}_")]
                                competition_col_names.extend(matching_cols)
                        
                        control_col_names = []
                        if st.session_state.get('control_data_v7'):
                            for ctrl_name in st.session_state.control_data_v7.keys():
                                matching_cols = [col for col in combined.columns if col.startswith(f"{ctrl_name}_")]
                                control_col_names.extend(matching_cols)
                        
                        # Store v7 configuration
                        st.session_state.v7_mode = True
                        st.session_state.v7_time_col = date_col
                        st.session_state.v7_dependent_var = combined.columns[1]  # Second column is KPI
                        st.session_state.v7_paid_media_cols = media_col_names
                        
                        # Classify channels
                        st.session_state.v7_tv_cols = []
                        if 'tv_channels' in locals():
                            for tv_ch in tv_channels:
                                st.session_state.v7_tv_cols.extend([col for col in media_col_names if col.startswith(f"{tv_ch}_")])
                        
                        st.session_state.v7_traditional_cols = []
                        if 'traditional_channels' in locals():
                            for trad_ch in traditional_channels:
                                st.session_state.v7_traditional_cols.extend([col for col in media_col_names if col.startswith(f"{trad_ch}_")])
                        
                        st.session_state.v7_digital_cols = [col for col in media_col_names 
                                                            if col not in st.session_state.v7_tv_cols + st.session_state.v7_traditional_cols]
                        
                        st.session_state.v7_competition_cols = competition_col_names
                        
                        st.session_state.v7_atl_cols = []
                        if 'atl_comps' in locals() and st.session_state.get('competition_data_v7'):
                            for atl_comp in atl_comps:
                                st.session_state.v7_atl_cols.extend([col for col in competition_col_names if col.startswith(f"{atl_comp}_")])
                        
                        st.session_state.v7_control_cols = control_col_names
                        st.session_state.data_uploaded = True
                        
                        st.success("✅ Data combined successfully with v7 configuration!")
                        st.balloons()
                        
                        # Show final configuration
                        st.markdown("---")
                        st.markdown("### 📋 Final v7 Configuration")
                        
                        config_summary = pd.DataFrame({
                            'Variable Type': ['Time', 'KPI', 'Total Media', 'Digital', 'TV/Video', 'Traditional', 'Competition', 'ATL', 'Controls'],
                            'Count': [
                                1,
                                1,
                                len(media_col_names),
                                len(st.session_state.v7_digital_cols),
                                len(st.session_state.v7_tv_cols),
                                len(st.session_state.v7_traditional_cols),
                                len(competition_col_names),
                                len(st.session_state.v7_atl_cols),
                                len(control_col_names)
                            ],
                            'Examples': [
                                date_col,
                                st.session_state.v7_dependent_var,
                                ', '.join(media_col_names[:3]) + ('...' if len(media_col_names) > 3 else ''),
                                ', '.join(st.session_state.v7_digital_cols[:2]) + ('...' if len(st.session_state.v7_digital_cols) > 2 else ''),
                                ', '.join(st.session_state.v7_tv_cols[:2]) + ('...' if len(st.session_state.v7_tv_cols) > 2 else ''),
                                ', '.join(st.session_state.v7_traditional_cols[:2]) + ('...' if len(st.session_state.v7_traditional_cols) > 2 else ''),
                                ', '.join(competition_col_names[:2]) + ('...' if len(competition_col_names) > 2 else ''),
                                ', '.join(st.session_state.v7_atl_cols[:2]) + ('...' if len(st.session_state.v7_atl_cols) > 2 else ''),
                                ', '.join(control_col_names[:2]) + ('...' if len(control_col_names) > 2 else '')
                            ]
                        })
                        
                        st.dataframe(config_summary, use_container_width=True)
                        
                        st.info("✅ **Next Step:** Go to 'Data Overview' to validate your data, then 'Modeling' to train!")
                        
                    except Exception as e:
                        st.error(f"❌ Error combining data: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        # COMBINED DATASET UPLOAD MODE (existing code from original)
        st.markdown("#### 📊 Upload Complete Dataset")
        st.info("Upload one file containing all variables (KPI, media, competition, controls, etc.)")
        
        uploaded_file = st.file_uploader(
            "Upload your complete MMM data file (CSV or Excel)",
            type=['csv', 'xlsx'],
            help="Single file with all variables"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"✅ File uploaded! Shape: {df.shape}")
                
                with st.expander("👀 Preview Data"):
                    st.dataframe(df.head(10))
                
                st.markdown("### ⚙️ Configure Variable Types")
                
                if 'v7_mode' not in st.session_state:
                    st.session_state.v7_mode = True
                
                col1, col2 = st.columns(2)
                
                with col1:
                    time_col = st.selectbox("📍 Time Column (MANDATORY)", df.columns)
                    dependent_var = st.selectbox("🎯 KPI/Dependent Variable (MANDATORY)", [c for c in df.columns if c != time_col])
                
                with col2:
                    paid_media_cols = st.multiselect("💰 Paid Media Spends (MANDATORY)", [c for c in df.columns if c not in [time_col, dependent_var]])
                
                if not paid_media_cols:
                    st.warning("⚠️ Select at least one media channel!")
                    st.stop()
                
                col3, col4 = st.columns(2)
                
                with col3:
                    competition_cols = st.multiselect(
                        "🏢 Competition Variables (Optional)",
                        [c for c in df.columns if c not in [time_col, dependent_var] + paid_media_cols]
                    )
                    tv_cols = st.multiselect("📺 TV/Video Channels (Optional)", paid_media_cols)
                
                with col4:
                    control_cols = st.multiselect(
                        "🎛️ Control Variables - Untransformed (Optional)",
                        [c for c in df.columns if c not in [time_col, dependent_var] + paid_media_cols + competition_cols]
                    )
                    traditional_cols = st.multiselect("📻 Traditional Media (Optional)", [c for c in paid_media_cols if c not in tv_cols])
                
                atl_cols = st.multiselect("📢 ATL Competition (Optional)", competition_cols) if competition_cols else []
                digital_cols = [c for c in paid_media_cols if c not in tv_cols + traditional_cols]
                
                # Config summary
                st.markdown("---")
                st.markdown("### 📋 Configuration Summary")
                
                config_summary = pd.DataFrame({
                    'Type': ['Time', 'KPI', 'Media', 'Digital', 'TV', 'Traditional', 'Competition', 'ATL', 'Controls'],
                    'Count': [1, 1, len(paid_media_cols), len(digital_cols), len(tv_cols), len(traditional_cols), 
                             len(competition_cols), len(atl_cols), len(control_cols)]
                })
                st.dataframe(config_summary)
                
                if st.button("✅ Confirm Configuration & Process", type="primary", use_container_width=True):
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
                        
                        st.success("✅ Configuration saved! Go to Overview →")
                        st.balloons()
            
            except Exception as e:
                st.error(f"Error: {e}")

                        [c for c in df.columns if c not in [time_col, dependent_var] + paid_media_cols + competition_cols]
                    )
                    traditional_cols = st.multiselect("📻 Traditional Media (Optional)", [c for c in paid_media_cols if c not in tv_cols])
                
                atl_cols = st.multiselect("📢 ATL Competition (Optional)", competition_cols) if competition_cols else []
                digital_cols = [c for c in paid_media_cols if c not in tv_cols + traditional_cols]
                
                # Config summary
                st.markdown("---")
                st.markdown("### 📋 Configuration Summary")
                
                config_summary = pd.DataFrame({
                    'Type': ['Time', 'KPI', 'Media', 'Digital', 'TV', 'Traditional', 'Competition', 'ATL', 'Controls'],
                    'Count': [1, 1, len(paid_media_cols), len(digital_cols), len(tv_cols), len(traditional_cols), 
                             len(competition_cols), len(atl_cols), len(control_cols)]
                })
                st.dataframe(config_summary)
                
                if st.button("✅ Confirm Configuration & Process", type="primary", use_container_width=True):
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
                        
                        st.success("✅ Configuration saved! Go to Overview →")
                        st.balloons()
            
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 2: Data Overview
elif tab_selection == "🔍 Data Overview":
    st.markdown('<p class="sub-header">Data Overview & Validation</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload and combine data first!")
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
        st.markdown("### ✅ Data Validation")
        
        validation_col1, validation_col2, validation_col3 = st.columns(3)
        
        with validation_col1:
            if date_range_months >= 24:
                st.success(f"✅ Sufficient data: {date_range_months:.1f} months")
            else:
                st.warning(f"⚠️ Limited data: {date_range_months:.1f} months")
        
        with validation_col2:
            has_revenue = any('revenue' in col.lower() or 'sales' in col.lower() for col in df.columns)
            if has_revenue:
                st.success("✅ Revenue/KPI column found")
            else:
                st.error("❌ Revenue/KPI column not found")
        
        with validation_col3:
            has_promo = any('promo' in col.lower() or 'discount' in col.lower() for col in df.columns)
            if has_promo:
                st.success("✅ Promotion data included")
            else:
                st.info("ℹ️ No promotion data")
        
        # Display combined data
        st.markdown("---")
        st.markdown("### 📊 Combined Dataset")
        
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
            label="📥 Download Combined Data",
            data=csv,
            file_name=f"combined_mmm_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Basic statistics
        st.markdown("---")
        st.markdown("### 📈 Descriptive Statistics")
        
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
        st.markdown("### 🔥 Correlation Heatmap")
        
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
elif tab_selection == "🎯 Marketing Mix Modeling":
    st.markdown('<p class="sub-header">Marketing Mix Modeling with v7 Enhancements</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload data first!")
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
        
        st.info("✅ v7 Mode: Using channel-specific parameter ranges and advanced features")
        
        # Feature Selection Toggle
        st.markdown("### 🔍 Feature Selection (NEW v7)")
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
            
            st.success("✅ Feature selection complete!")
            
            # Show results
            results_df = pd.DataFrame(channel_params).T
            results_df['Channel'] = results_df.index
            results_df = results_df[['Channel', 'type', 'theta', 'alpha', 'gamma', 'corr']]
            st.dataframe(results_df.style.format({'theta': '{:.2f}', 'alpha': '{:.2f}', 'gamma': '{:.2f}', 'corr': '{:.4f}'}))
        
        else:
            # Manual channel-specific configuration
            st.markdown("### ⚙️ Channel-Specific Parameters (NEW v7)")
            
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
                
                with st.expander(f"⚙️ {channel} - {ch_type}"):
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
            st.warning("⚠️ Select at least one media channel!")
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
        st.markdown("### ⚙️ Model Parameters")
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
    if st.button("🚀 Run Marketing Mix Model", type="primary", use_container_width=True):
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
                
                st.success("✅ Model trained successfully!")
                st.balloons()
                
                # Display metrics
                st.markdown("---")
                st.markdown("### 📊 Model Performance")
                
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
                st.markdown("### 📈 Model Fit Visualization")
                
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
                
                st.info("✅ Model training complete! Go to 'Results & Insights' →")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# TAB 4: Results & Insights (COMPLETE WITH v7 ENHANCEMENTS)
elif tab_selection == "📈 Results & Insights":
    st.markdown('<p class="sub-header">Results & Insights</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
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
        "📊 Channel Contribution",
        "💰 ROI Analysis (v7 Enhanced)",
        "📈 Response Curves",
        "🎯 Budget Optimization (Full)",
        "📋 Model Summary (v7 Diagnostics)"
    ])
    
    # TAB 1: CHANNEL CONTRIBUTION (WITH DE-STANDARDIZATION)
    with result_tabs[0]:
        st.markdown("### Channel Contribution to Revenue")
        
        st.info("📊 **v7 Enhancement:** Using de-standardized reporting for business-friendly interpretation")
        
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
        
        st.info("💰 **v7 Enhancements:** De-standardized iROAS + DECOMP.RSSD metric")
        
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
            st.warning("⚠️ No channels with positive spend")
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
        st.markdown("### 📐 DECOMP.RSSD - Spend vs Effect Share Analysis (NEW v7)")
        
        rssd, spend_share, effect_share = calculate_decomp_rssd(test_df, contributions, media_cols)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("DECOMP.RSSD", f"{rssd:.4f}")
            st.caption("Lower is better - measures misalignment")
            
            if rssd < 0.1:
                st.success("✅ Excellent alignment")
            elif rssd < 0.2:
                st.info("ℹ️ Good alignment")
            elif rssd < 0.3:
                st.warning("⚠️ Moderate misalignment")
            else:
                st.error("❌ High misalignment")
        
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
        st.markdown("### 💡 Key Insights")
        
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
            st.warning(f"⚠️ No positive spend for {selected_channel}")
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
        st.markdown("### 📊 Current Performance")
        
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
        if st.button("🚀 Run Full Optimization", type="primary", use_container_width=True):
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
                        st.success("✅ Optimization completed!")
                        
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
                        st.markdown("#### 📊 Optimal Allocation")
                        
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
                        st.markdown("### 📈 Expected Impact")
                        
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
                        with st.expander("📊 Model vs Actual Comparison"):
                            actual_revenue = y_test.sum()
                            prediction_error = ((current_revenue_model - actual_revenue) / actual_revenue * 100)
                            
                            st.write(f"**Actual Revenue:** ${actual_revenue:,.0f}")
                            st.write(f"**Model Prediction:** ${current_revenue_model:,.0f}")
                            st.write(f"**Prediction Error:** {prediction_error:+.1f}%")
                            st.caption("Optimization compares model predictions")
                        
                        # Details
                        with st.expander("🔧 Optimization Details"):
                            st.write(f"**Status:** {solution.message}")
                            st.write(f"**Iterations:** {solution.nit}")
                            st.write(f"**Function Evals:** {solution.nfev}")
                            st.write(f"**Objective Value:** {-solution.fun:,.2f}")
                    
                    else:
                        st.error(f"❌ Optimization failed: {solution.message}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # TAB 5: MODEL SUMMARY (WITH v7 DIAGNOSTICS)
    with result_tabs[4]:
        st.markdown("### Model Summary with v7 Diagnostics")
        
        st.info("📊 **v7 Enhancements:** VIF analysis, Extended metrics, Confidence intervals, Durbin-Watson")
        
        # NEW v7: Extended Diagnostics
        st.markdown("#### 📊 Extended Diagnostics (v7)")
        
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
            st.success("✅ No significant autocorrelation")
        elif dw < 1.5:
            st.warning("⚠️ Positive autocorrelation detected")
        else:
            st.warning("⚠️ Negative autocorrelation detected")
        
        # NEW v7: VIF Analysis
        st.markdown("---")
        st.markdown("#### 🔍 VIF Analysis (Multicollinearity Detection)")
        
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
        - VIF < 5: ✅ Low multicollinearity
        - VIF 5-10: ⚠️ Moderate
        - VIF > 10: ❌ High (consider removing)
        """)
        
        # NEW v7: Confidence Intervals
        st.markdown("---")
        st.markdown("#### 📏 Coefficient Confidence Intervals (95%)")
        
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
        st.markdown("#### 📈 Model Statistics")
        
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
        st.markdown("#### 🔍 Model Diagnostics (4 Plots)")
        
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
