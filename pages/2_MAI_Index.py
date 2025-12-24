# %%writefile mai_dashboard.py
"""
MAI Index Market Breadth Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta, date
import requests
from io import StringIO
from typing import Optional, Tuple
import pytz

# =====================================================
# AUTHENTICATION CHECK - MUST BE AT THE TOP
# =====================================================
if "password_correct" not in st.session_state or not st.session_state.get("password_correct", False):
    st.set_page_config(page_title="MAI Index - Login Required", layout="wide")
    st.error("🔒 Please log in from the home page first")
    st.page_link("Cover.py", label="← Go to Login Page", icon="🏠")
    st.stop()
    
# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="MAI Index Market Analysis", layout="wide")

SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"
GID = "540279382"

URL_PRIMARY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
URL_FALLBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

st.title("📈 MAI Index Market Analysis")

# -------------------------
# Helper Functions
# -------------------------
def _looks_like_html(text: str) -> bool:
    t = (text or "").lstrip().lower()
    return (
        t.startswith("<!doctype html")
        or t.startswith("<html")
        or ("servicelogin" in t)
        or ("accounts.google.com" in t)
    )

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"nan": "", "None": "", "null": ""})
    s = (
        s.str.replace(",", "", regex=False)
         .str.replace("%", "", regex=False)
         .str.replace("−", "-", regex=False)
         .str.replace("—", "-", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")

def _parse_date_series(s: pd.Series) -> pd.Series:
    """Robust date parsing"""
    ss = s.astype(str).str.strip()
    dt = pd.to_datetime(ss, errors="coerce", infer_datetime_format=True)
    ok = int(dt.notna().sum())
    
    # Fallback: Excel serial dates
    if ok < max(3, int(len(ss) * 0.2)):
        num = pd.to_numeric(ss, errors="coerce")
        if num.notna().sum() >= max(3, int(len(ss) * 0.5)) and num.median(skipna=True) > 10000:
            dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    
    return dt.dt.normalize()

def make_rangebreaks(dates: pd.Series, include_holidays: bool, max_holidays: int = 250):
    """
    FAST rangebreaks:
    - always skip weekends
    - optionally skip missing business days (holidays) BUT capped
    """
    rbs = [dict(bounds=["sat", "mon"])]
    
    if not include_holidays:
        return rbs
    
    dt = pd.to_datetime(dates, errors="coerce").dt.normalize()
    dt = dt.dropna()
    if dt.empty:
        return rbs
    
    obs = pd.DatetimeIndex(dt.unique())
    bdays = pd.date_range(obs.min(), obs.max(), freq="B")
    missing = bdays.difference(obs)
    
    # Cap to avoid Plotly freezing
    if len(missing) > 0 and len(missing) <= max_holidays:
        rbs.append(dict(values=list(missing)))
    
    return rbs

@st.cache_data(ttl=600, show_spinner=False)
def load_mai_data(url_primary: str, url_fallback: str) -> pd.DataFrame:
    """Load MAI data from Google Sheets"""
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    
    last_err = None
    for url in (url_primary, url_fallback):
        try:
            r = session.get(url, headers=headers, timeout=20, allow_redirects=True)
            r.raise_for_status()
            txt = r.text or ""
            if _looks_like_html(txt):
                last_err = f"HTML/login page returned from: {url}"
                continue
            df = pd.read_csv(StringIO(txt))
            return df
        except Exception as e:
            last_err = str(e)
    
    raise ValueError(f"Google Sheet did not return CSV. Last error: {last_err}")

def parse_mai_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean MAI data"""
    # Parse date column (Column A)
    df['Date'] = _parse_date_series(df.iloc[:, 0])
    
    # OHLC columns (B, C, D, E)
    df['Open'] = _clean_numeric_series(df.iloc[:, 1])
    df['High'] = _clean_numeric_series(df.iloc[:, 2])
    df['Low'] = _clean_numeric_series(df.iloc[:, 3])
    df['Close'] = _clean_numeric_series(df.iloc[:, 4])
    
    # Moving Average columns (F-O)
    df['Above_EMA10'] = _clean_numeric_series(df.iloc[:, 5])
    df['Above_EMA20'] = _clean_numeric_series(df.iloc[:, 6])
    df['Above_EMA50'] = _clean_numeric_series(df.iloc[:, 7])
    df['Above_EMA100'] = _clean_numeric_series(df.iloc[:, 8])
    df['Above_EMA200'] = _clean_numeric_series(df.iloc[:, 9])
    
    # New High/Low columns (P-W) - columns 15-22
    df['NH20'] = _clean_numeric_series(df.iloc[:, 15])
    df['NH65'] = _clean_numeric_series(df.iloc[:, 16])
    df['NH130'] = _clean_numeric_series(df.iloc[:, 17])
    df['NH260'] = _clean_numeric_series(df.iloc[:, 18])
    df['NL20'] = _clean_numeric_series(df.iloc[:, 19])
    df['NL65'] = _clean_numeric_series(df.iloc[:, 20])
    df['NL130'] = _clean_numeric_series(df.iloc[:, 21])
    df['NL260'] = _clean_numeric_series(df.iloc[:, 22])
    
    # Modified New High/Low columns (X-AE) - columns 23-30
    df['Ratio_NHNL20'] = _clean_numeric_series(df.iloc[:, 23]) if df.shape[1] > 23 else pd.Series([pd.NA] * len(df))
    df['Ratio_NHNL20_MA20'] = _clean_numeric_series(df.iloc[:, 24]) if df.shape[1] > 24 else pd.Series([pd.NA] * len(df))
    df['Ratio_NHNL50'] = _clean_numeric_series(df.iloc[:, 25]) if df.shape[1] > 25 else pd.Series([pd.NA] * len(df))
    df['Ratio_NHNL50_MA50'] = _clean_numeric_series(df.iloc[:, 26]) if df.shape[1] > 26 else pd.Series([pd.NA] * len(df))
    df['Ratio_NHNL200'] = _clean_numeric_series(df.iloc[:, 27]) if df.shape[1] > 27 else pd.Series([pd.NA] * len(df))
    df['Ratio_NHNL200_MA50'] = _clean_numeric_series(df.iloc[:, 28]) if df.shape[1] > 28 else pd.Series([pd.NA] * len(df))
    df['Diff_Short_Long_NHNL'] = _clean_numeric_series(df.iloc[:, 29]) if df.shape[1] > 29 else pd.Series([pd.NA] * len(df))
    df['Diff_Short_Long_NHNL_MA60'] = _clean_numeric_series(df.iloc[:, 30]) if df.shape[1] > 30 else pd.Series([pd.NA] * len(df))
    
    # Additional columns for new tabs
    # Column AF-AR (columns 31-43)
    df['NH10'] = _clean_numeric_series(df.iloc[:, 31]) if df.shape[1] > 31 else pd.Series([pd.NA] * len(df))
    df['NL10'] = _clean_numeric_series(df.iloc[:, 32]) if df.shape[1] > 32 else pd.Series([pd.NA] * len(df))
    df['Diff_NHNL10'] = _clean_numeric_series(df.iloc[:, 33]) if df.shape[1] > 33 else pd.Series([pd.NA] * len(df))
    df['Diff_NHNL20'] = _clean_numeric_series(df.iloc[:, 34]) if df.shape[1] > 34 else pd.Series([pd.NA] * len(df))
    df['NH50'] = _clean_numeric_series(df.iloc[:, 35]) if df.shape[1] > 35 else pd.Series([pd.NA] * len(df))
    df['NL50'] = _clean_numeric_series(df.iloc[:, 36]) if df.shape[1] > 36 else pd.Series([pd.NA] * len(df))
    df['Diff_NHNL50'] = _clean_numeric_series(df.iloc[:, 37]) if df.shape[1] > 37 else pd.Series([pd.NA] * len(df))
    df['NH100'] = _clean_numeric_series(df.iloc[:, 38]) if df.shape[1] > 38 else pd.Series([pd.NA] * len(df))
    df['NL100'] = _clean_numeric_series(df.iloc[:, 39]) if df.shape[1] > 39 else pd.Series([pd.NA] * len(df))
    df['Diff_NHNL100'] = _clean_numeric_series(df.iloc[:, 40]) if df.shape[1] > 40 else pd.Series([pd.NA] * len(df))
    df['NH200'] = _clean_numeric_series(df.iloc[:, 41]) if df.shape[1] > 41 else pd.Series([pd.NA] * len(df))
    df['NL200'] = _clean_numeric_series(df.iloc[:, 42]) if df.shape[1] > 42 else pd.Series([pd.NA] * len(df))
    df['Diff_NHNL200'] = _clean_numeric_series(df.iloc[:, 43]) if df.shape[1] > 43 else pd.Series([pd.NA] * len(df))
    
    # Double Moving Averages (Column AS-AT) - columns 44-45
    df['Percentage_Above_Both'] = _clean_numeric_series(df.iloc[:, 44]) if df.shape[1] > 44 else pd.Series([pd.NA] * len(df))
    df['Percentage_Below_Both'] = _clean_numeric_series(df.iloc[:, 45]) if df.shape[1] > 45 else pd.Series([pd.NA] * len(df))
    
    # Keep only processed columns
    processed_cols = ['Date', 'Open', 'High', 'Low', 'Close',
                      'Above_EMA10', 'Above_EMA20', 'Above_EMA50', 'Above_EMA100', 'Above_EMA200',
                      'NH20', 'NH65', 'NH130', 'NH260',
                      'NL20', 'NL65', 'NL130', 'NL260',
                      'Ratio_NHNL20', 'Ratio_NHNL20_MA20', 'Ratio_NHNL50', 'Ratio_NHNL50_MA50',
                      'Ratio_NHNL200', 'Ratio_NHNL200_MA50', 'Diff_Short_Long_NHNL', 'Diff_Short_Long_NHNL_MA60',
                      'NH10', 'NL10', 'Diff_NHNL10', 'Diff_NHNL20',
                      'NH50', 'NL50', 'Diff_NHNL50',
                      'NH100', 'NL100', 'Diff_NHNL100',
                      'NH200', 'NL200', 'Diff_NHNL200',
                      'Percentage_Above_Both', 'Percentage_Below_Both']
    
    df = df[processed_cols].dropna(subset=['Date']).sort_values('Date')
    
    return df

# -------------------------
# Load Data
# -------------------------
try:
    raw_df = load_mai_data(URL_PRIMARY, URL_FALLBACK)
    df = parse_mai_data(raw_df)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if df.empty:
    st.warning("No valid data available")
    st.stop()

# -------------------------
# Controls
# -------------------------
# Get Bangkok timezone
bangkok_tz = pytz.timezone('Asia/Bangkok')
last_update = pd.Timestamp.now(tz=bangkok_tz)

col1, col2, col3 = st.columns([1.5, 1.5, 3])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col2:
    months_to_show = st.selectbox("Time Period (month)", [1, 3, 6, 12], index=2)

with col3:
    # Show last request time
    st.info(f"⏱️ Last request: {last_update.strftime('%Y-%m-%d %H:%M:%S')} Bangkok Time")

# Calculate date range based on selected period
st.markdown("---")
max_date = df['Date'].max()
min_date = df['Date'].min()

# Calculate start date based on selected months
start_date = (max_date - timedelta(days=30 * int(months_to_show))).date()
end_date = max_date.date()

# Ensure start_date is not before min_date
if start_date < min_date.date():
    start_date = min_date.date()

# Filter data by date range
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
dff = df[(df['Date'] >= start_ts) & (df['Date'] <= end_ts)].copy()

if dff.empty:
    st.warning("No data in selected date range")
    st.stop()

st.write(f"Showing: **{start_date} → {end_date}** ({len(dff)} data points)")

# Create rangebreaks to remove gaps (weekends and holidays)
rangebreaks = make_rangebreaks(dff['Date'], include_holidays=True, max_holidays=250)

# -------------------------
# PANEL 1: MAI Index Candlestick
# -------------------------
st.markdown("---")
with st.expander("📈 MAI Index", expanded=True):
    fig1 = go.Figure(
        go.Candlestick(
            x=dff['Date'],
            open=dff['Open'],
            high=dff['High'],
            low=dff['Low'],
            close=dff['Close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name='MAI Index'
        )
    )
    
    # Apply rangebreaks to remove gaps
    if rangebreaks:
        fig1.update_xaxes(rangebreaks=rangebreaks)
    
    fig1.update_layout(
        height=500,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis_title='MAI Index',
        xaxis_title='Date',
        hovermode='x unified',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': True})

# -------------------------
# PANEL 2: New Highs & Lows
# -------------------------
st.markdown("---")
with st.expander("📈 New Highs & Lows", expanded=True):
    # Panel 1: NH20 & NL20
    show_nh_nl_20 = st.checkbox("📊 New High & New Low 4 Weeks", value=True, key="show_nh_nl_20")
    if show_nh_nl_20:
        fig_nh_nl_20 = go.Figure()
        
        if dff['NH20'].notna().any():
            fig_nh_nl_20.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NH20'],
                name='Percentage of Members with New 4 Week Highs',
                marker_color='rgba(0, 255, 0, 0.3)',
                showlegend=True
            ))
        
        if dff['NL20'].notna().any():
            fig_nh_nl_20.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NL20'] * -1,
                name='Percentage of Members with New 4 Week Lows',
                marker_color='rgba(255, 107, 107, 0.3)',
                showlegend=True
            ))
        
        if dff['NH20'].notna().any():
            fig_nh_nl_20.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NH20'],
                name='Percentage of Members with New 4 Week Highs (Line)',
                line=dict(width=2, color='#00ff00'),
                mode='lines',
                showlegend=True
            ))
        
        if dff['NL20'].notna().any():
            fig_nh_nl_20.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NL20'],
                name='Percentage of Members with New 4 Week Lows (Line)',
                line=dict(width=2, color='#ff6b6b', dash='solid'),
                mode='lines',
                showlegend=True
            ))
        
        if rangebreaks:
            fig_nh_nl_20.update_xaxes(rangebreaks=rangebreaks)
        
        fig_nh_nl_20.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis_title='New Highs & New Lows (4 weeks)',
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig_nh_nl_20, use_container_width=True, config={'displayModeBar': True})
        st.markdown("---")
    
    # Panel 2: NH65 & NL65
    show_nh_nl_65 = st.checkbox("📊 New High & New Low 13 Weeks", value=True, key="show_nh_nl_65")
    if show_nh_nl_65:
        fig_nh_nl_65 = go.Figure()
        
        if dff['NH65'].notna().any():
            fig_nh_nl_65.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NH65'],
                name='Percentage of Members with New 13 Week Highs',
                marker_color='rgba(38, 166, 154, 0.3)',
                showlegend=True
            ))
        
        if dff['NL65'].notna().any():
            fig_nh_nl_65.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NL65'] * -1,
                name='Percentage of Members with New 13 Week Lows',
                marker_color='rgba(239, 83, 80, 0.3)',
                showlegend=True
            ))
        
        if dff['NH65'].notna().any():
            fig_nh_nl_65.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NH65'],
                name='Percentage of Members with New 13 Week Highs (Line)',
                line=dict(width=2, color='#26a69a'),
                mode='lines',
                showlegend=True
            ))
        
        if dff['NL65'].notna().any():
            fig_nh_nl_65.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NL65'],
                name='Percentage of Members with New 13 Week Lows (Line)',
                line=dict(width=2, color='#ef5350', dash='solid'),
                mode='lines',
                showlegend=True
            ))
        
        if rangebreaks:
            fig_nh_nl_65.update_xaxes(rangebreaks=rangebreaks)
        
        fig_nh_nl_65.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis_title='New Highs & New Lows (13 weeks)',
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig_nh_nl_65, use_container_width=True, config={'displayModeBar': True})
        st.markdown("---")
    
    # Panel 3: NH130 & NL130
    show_nh_nl_130 = st.checkbox("📊 New High & New Low 26 Weeks", value=True, key="show_nh_nl_130")
    if show_nh_nl_130:
        fig_nh_nl_130 = go.Figure()
        
        if dff['NH130'].notna().any():
            fig_nh_nl_130.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NH130'],
                name='Percentage of Members with New 26 Week Highs',
                marker_color='rgba(46, 204, 113, 0.3)',
                showlegend=True
            ))
        
        if dff['NL130'].notna().any():
            fig_nh_nl_130.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NL130'] * -1,
                name='Percentage of Members with New 26 Week Lows',
                marker_color='rgba(231, 76, 60, 0.3)',
                showlegend=True
            ))
        
        if dff['NH130'].notna().any():
            fig_nh_nl_130.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NH130'],
                name='Percentage of Members with New 26 Week Highs (Line)',
                line=dict(width=2, color='#2ecc71'),
                mode='lines',
                showlegend=True
            ))
        
        if dff['NL130'].notna().any():
            fig_nh_nl_130.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NL130'],
                name='Percentage of Members with New 26 Week Lows (Line)',
                line=dict(width=2, color='#e74c3c', dash='solid'),
                mode='lines',
                showlegend=True
            ))
        
        if rangebreaks:
            fig_nh_nl_130.update_xaxes(rangebreaks=rangebreaks)
        
        fig_nh_nl_130.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis_title='New Highs & New Lows (26 weeks)',
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig_nh_nl_130, use_container_width=True, config={'displayModeBar': True})
        st.markdown("---")
    
    # Panel 4: NH260 & NL260
    show_nh_nl_260 = st.checkbox("📊 New High & New Low 52 Weeks", value=True, key="show_nh_nl_260")
    if show_nh_nl_260:
        fig_nh_nl_260 = go.Figure()
        
        if dff['NH260'].notna().any():
            fig_nh_nl_260.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NH260'],
                name='Percentage of Members with New 52 Week Highs',
                marker_color='rgba(39, 174, 96, 0.3)',
                showlegend=True
            ))
        
        if dff['NL260'].notna().any():
            fig_nh_nl_260.add_trace(go.Bar(
                x=dff['Date'],
                y=dff['NL260'] * -1,
                name='Percentage of Members with New 52 Week Lows',
                marker_color='rgba(192, 57, 43, 0.3)',
                showlegend=True
            ))
        
        if dff['NH260'].notna().any():
            fig_nh_nl_260.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NH260'],
                name='Percentage of Members with New 52 Week Highs (Line)',
                line=dict(width=2.5, color='#27ae60'),
                mode='lines',
                showlegend=True
            ))
        
        if dff['NL260'].notna().any():
            fig_nh_nl_260.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['NL260'],
                name='Percentage of Members with New 52 Week Lows (Line)',
                line=dict(width=2.5, color='#c0392b', dash='solid'),
                mode='lines',
                showlegend=True
            ))
        
        if rangebreaks:
            fig_nh_nl_260.update_xaxes(rangebreaks=rangebreaks)
        
        fig_nh_nl_260.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            yaxis_title='New Highs & New Lows (52 weeks)',
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig_nh_nl_260, use_container_width=True, config={'displayModeBar': True})

# -------------------------
# PANEL 3: Market Breadth Analysis (without New Highs & Lows tab)
# -------------------------
st.markdown("---")
with st.expander("📊 Market Breadth Analysis", expanded=True):
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Simple Moving Averages", 
        "📉 Modified New High & New Lows",
        "📊 Double Moving Averages",
        "📈 Different New Highs and New Lows"
    ])
    
    # Tab 1: Moving Averages
    with tab1:
        # Multiply by 100 to convert to percentage
        dff_pct = dff.copy()
        dff_pct['Above_EMA10'] = dff['Above_EMA10'] * 100
        dff_pct['Above_EMA20'] = dff['Above_EMA20'] * 100
        dff_pct['Above_EMA50'] = dff['Above_EMA50'] * 100
        dff_pct['Above_EMA100'] = dff['Above_EMA100'] * 100
        dff_pct['Above_EMA200'] = dff['Above_EMA200'] * 100
        
        fig_above = go.Figure()
        
        # Above EMA lines (green shades)
        if dff_pct['Above_EMA10'].notna().any():
            fig_above.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA10'],
                name='Percentage of Members with Px > 10 Day Simple Moving Average',
                line=dict(width=1, color='#00ff00'),
                mode='lines'
            ))
        
        if dff_pct['Above_EMA20'].notna().any():
            fig_above.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA20'],
                name='Percentage of Members with Px > 20 Day Simple Moving Average',
                line=dict(width=1.5, color='#26a69a'),
                mode='lines'
            ))
        
        if dff_pct['Above_EMA50'].notna().any():
            fig_above.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA50'],
                name='Percentage of Members with Px > 50 Day Simple Moving Average',
                line=dict(width=2, color='#2ecc71'),
                mode='lines'
            ))
        
        if dff_pct['Above_EMA100'].notna().any():
            fig_above.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA100'],
                name='Percentage of Members with Px > 100 Day Simple Moving Average',
                line=dict(width=2.5, color='#27ae60'),
                mode='lines'
            ))
        
        if dff_pct['Above_EMA200'].notna().any():
            fig_above.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA200'],
                name='Percentage of Members with Px > 200 Day Simple Moving Average',
                line=dict(width=3, color='#1e8449'),
                mode='lines'
            ))
        
        if rangebreaks:
            fig_above.update_xaxes(rangebreaks=rangebreaks)
        
        fig_above.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            yaxis=dict(
                title='Above Simple Moving Averages',
                range=[0, 100]
            ),
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        st.plotly_chart(fig_above, use_container_width=True, config={'displayModeBar': True})
    
    # Tab 2: Modified New High & New Lows
    with tab2:
        fig_ratio = go.Figure()
        
        if dff['Ratio_NHNL20'].notna().any():
            fig_ratio.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Ratio_NHNL20'] * 100,
                name='Ratio New Highs & New Lows 4 Weeks',
                line=dict(width=2, color='#00ff00', dash='solid'),
                mode='lines'
            ))
        
        if dff['Ratio_NHNL20_MA20'].notna().any():
            fig_ratio.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Ratio_NHNL20_MA20'] * 100,
                name='Ratio New Highs & New Lows 4 Weeks (Threshold line)',
                line=dict(width=2, color='#00ff00', dash='dash'),
                mode='lines'
            ))
        
        if dff['Ratio_NHNL50'].notna().any():
            fig_ratio.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Ratio_NHNL50'] * 100,
                name='Ratio New Highs & New Lows 10 Weeks',
                line=dict(width=2, color='#2ecc71', dash='solid'),
                mode='lines'
            ))
        
        if dff['Ratio_NHNL50_MA50'].notna().any():
            fig_ratio.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Ratio_NHNL50_MA50'] * 100,
                name='Ratio New Highs & New Lows 10 Weeks (Threshold line)',
                line=dict(width=2, color='#2ecc71', dash='dash'),
                mode='lines'
            ))
        
        if dff['Ratio_NHNL200'].notna().any():
            fig_ratio.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Ratio_NHNL200'] * 100,
                name='Ratio New Highs & New Lows 40 Weeks',
                line=dict(width=2, color='#1e8449', dash='solid'),
                mode='lines'
            ))
        
        if dff['Ratio_NHNL200_MA50'].notna().any():
            fig_ratio.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Ratio_NHNL200_MA50'] * 100,
                name='Ratio New Highs & New Lows 40 Weeks (Threshold line)',
                line=dict(width=2, color='#1e8449', dash='dash'),
                mode='lines'
            ))
        
        if rangebreaks:
            fig_ratio.update_xaxes(rangebreaks=rangebreaks)
        
        fig_ratio.update_layout(
            height=450,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            yaxis=dict(
                title='Ratio New Highs and New Lows',
                range=[0, 100]
            ),
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        st.plotly_chart(fig_ratio, use_container_width=True, config={'displayModeBar': True})
                
        st.markdown("---")
        
        fig_diff = go.Figure()
        
        if dff['Diff_Short_Long_NHNL'].notna().any():
            fig_diff.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Diff_Short_Long_NHNL'],
                name='Diff. Short-Long New Highs and New Lows',
                line=dict(width=2, color='#3498db', dash='solid'),
                mode='lines'
            ))
        
        if dff['Diff_Short_Long_NHNL_MA60'].notna().any():
            fig_diff.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Diff_Short_Long_NHNL_MA60'],
                name='Threshold Line',
                line=dict(width=2, color='#3498db', dash='dash'),
                mode='lines'
            ))
        
        if rangebreaks:
            fig_diff.update_xaxes(rangebreaks=rangebreaks)
        
        fig_diff.update_layout(
            height=450,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            yaxis_title='Diff. Short-Long New Highs and New Lows',
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        st.plotly_chart(fig_diff, use_container_width=True, config={'displayModeBar': True})
    
    # Tab 3: Double Moving Averages
    with tab3:
        fig_dma = go.Figure()
        
        if dff['Percentage_Above_Both'].notna().any():
            fig_dma.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Percentage_Above_Both'],
                name='Percentage of Members Above 50-DMA and 200-DMA',
                line=dict(width=2, color='#00ff00'),
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.2)'
            ))
        
        if dff['Percentage_Below_Both'].notna().any():
            fig_dma.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Percentage_Below_Both'],
                name='Percentage of Members Below 50-DMA and 200-DMA',
                line=dict(width=2, color='#ff0000'),
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ))
        
        if rangebreaks:
            fig_dma.update_xaxes(rangebreaks=rangebreaks)
        
        fig_dma.update_layout(
            height=450,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            yaxis=dict(
                title='Double Moving Averages',
                range=[0, 100]
            ),
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        st.plotly_chart(fig_dma, use_container_width=True, config={'displayModeBar': True})
    
    # Tab 4: Different New Highs and New Lows
    with tab4:
        # Panel 1: 2 Weeks
        show_diff_2w = st.checkbox("📊 New High & New Low 2 Weeks", value=True, key="show_diff_2w")
        if show_diff_2w:
            fig_diff_2w = go.Figure()
            
            if dff['Diff_NHNL10'].notna().any():
                colors = ['#00ff00' if val >= 0 else '#ff0000' for val in dff['Diff_NHNL10']]
                fig_diff_2w.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['Diff_NHNL10'],
                    name='Different Percentage of Members with New 2 Weeks Highs and New 2 Weeks Lows',
                    marker_color=colors,
                    showlegend=True
                ))
            
            if dff['NH10'].notna().any():
                fig_diff_2w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH10'],
                    name='Percentage of Members with New 2 Weeks Highs',
                    line=dict(width=2, color='#00ff00'),
                    mode='lines',
                    showlegend=True
                ))
            
            if dff['NL10'].notna().any():
                fig_diff_2w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NL10'],
                    name='Percentage of Members with New 2 Weeks Lows',
                    line=dict(width=2, color='#ff0000'),
                    mode='lines',
                    showlegend=True
                ))
            
            if rangebreaks:
                fig_diff_2w.update_xaxes(rangebreaks=rangebreaks)
            
            fig_diff_2w.update_layout(
                height=400,
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                yaxis_title='New Highs & New Lows (2 weeks)',
                xaxis_title='Date',
                margin=dict(l=10, r=10, t=60, b=10)
            )
            
            st.plotly_chart(fig_diff_2w, use_container_width=True, config={'displayModeBar': True})
            st.markdown("---")
        
        # Panel 2: 4 Weeks
        show_diff_4w = st.checkbox("📊 New High & New Low 4 Weeks", value=True, key="show_diff_4w")
        if show_diff_4w:
            fig_diff_4w = go.Figure()
            
            if dff['Diff_NHNL20'].notna().any():
                colors = ['#00ff00' if val >= 0 else '#ff0000' for val in dff['Diff_NHNL20']]
                fig_diff_4w.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['Diff_NHNL20'],
                    name='Different Percentage of Members with New 4 Weeks Highs and New 4 Weeks Lows',
                    marker_color=colors,
                    showlegend=True
                ))
            
            if dff['NH20'].notna().any():
                fig_diff_4w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH20'],
                    name='Percentage of Members with New 4 Weeks Highs',
                    line=dict(width=2, color='#00ff00'),
                    mode='lines',
                    showlegend=True
                ))
            
            if dff['NL20'].notna().any():
                fig_diff_4w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NL20'],
                    name='Percentage of Members with New 4 Weeks Lows',
                    line=dict(width=2, color='#ff0000'),
                    mode='lines',
                    showlegend=True
                ))
            
            if rangebreaks:
                fig_diff_4w.update_xaxes(rangebreaks=rangebreaks)
            
            fig_diff_4w.update_layout(
                height=400,
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                yaxis_title='New Highs & New Lows (4 weeks)',
                xaxis_title='Date',
                margin=dict(l=10, r=10, t=60, b=10)
            )
            
            st.plotly_chart(fig_diff_4w, use_container_width=True, config={'displayModeBar': True})
            st.markdown("---")
        
        # Panel 3: 10 Weeks
        show_diff_10w = st.checkbox("📊 New High & New Low 10 Weeks", value=True, key="show_diff_10w")
        if show_diff_10w:
            fig_diff_10w = go.Figure()
            
            if dff['Diff_NHNL50'].notna().any():
                colors = ['#00ff00' if val >= 0 else '#ff0000' for val in dff['Diff_NHNL50']]
                fig_diff_10w.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['Diff_NHNL50'],
                    name='Different Percentage of Members with New 10 Weeks Highs and New 10 Weeks Lows',
                    marker_color=colors,
                    showlegend=True
                ))
            
            if dff['NH50'].notna().any():
                fig_diff_10w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH50'],
                    name='Percentage of Members with New 10 Weeks Highs',
                    line=dict(width=2, color='#00ff00'),
                    mode='lines',
                    showlegend=True
                ))
            
            if dff['NL50'].notna().any():
                fig_diff_10w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NL50'],
                    name='Percentage of Members with New 10 Weeks Lows',
                    line=dict(width=2, color='#ff0000'),
                    mode='lines',
                    showlegend=True
                ))
            
            if rangebreaks:
                fig_diff_10w.update_xaxes(rangebreaks=rangebreaks)
            
            fig_diff_10w.update_layout(
                height=400,
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                yaxis_title='New Highs & New Lows (10 weeks)',
                xaxis_title='Date',
                margin=dict(l=10, r=10, t=60, b=10)
            )
            
            st.plotly_chart(fig_diff_10w, use_container_width=True, config={'displayModeBar': True})
            st.markdown("---")
        
        # Panel 4: 20 Weeks
        show_diff_20w = st.checkbox("📊 New High & New Low 20 Weeks", value=True, key="show_diff_20w")
        if show_diff_20w:
            fig_diff_20w = go.Figure()
            
            if dff['Diff_NHNL100'].notna().any():
                colors = ['#00ff00' if val >= 0 else '#ff0000' for val in dff['Diff_NHNL100']]
                fig_diff_20w.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['Diff_NHNL100'],
                    name='Different Percentage of Members with New 20 Weeks Highs and New 20 Weeks Lows',
                    marker_color=colors,
                    showlegend=True
                ))
            
            if dff['NH100'].notna().any():
                fig_diff_20w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH100'],
                    name='Percentage of Members with New 20 Weeks Highs',
                    line=dict(width=2, color='#00ff00'),
                    mode='lines',
                    showlegend=True
                ))
            
            if dff['NL100'].notna().any():
                fig_diff_20w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NL100'],
                    name='Percentage of Members with New 20 Weeks Lows',
                    line=dict(width=2, color='#ff0000'),
                    mode='lines',
                    showlegend=True
                ))
            
            if rangebreaks:
                fig_diff_20w.update_xaxes(rangebreaks=rangebreaks)
            
            fig_diff_20w.update_layout(
                height=400,
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                yaxis_title='New Highs & New Lows (20 weeks)',
                xaxis_title='Date',
                margin=dict(l=10, r=10, t=60, b=10)
            )
            
            st.plotly_chart(fig_diff_20w, use_container_width=True, config={'displayModeBar': True})
            st.markdown("---")
        
        # Panel 5: 40 Weeks
        show_diff_40w = st.checkbox("📊 New High & New Low 40 Weeks", value=True, key="show_diff_40w")
        if show_diff_40w:
            fig_diff_40w = go.Figure()
            
            if dff['Diff_NHNL200'].notna().any():
                colors = ['#00ff00' if val >= 0 else '#ff0000' for val in dff['Diff_NHNL200']]
                fig_diff_40w.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['Diff_NHNL200'],
                    name='Different Percentage of Members with New 40 Weeks Highs and New 40 Weeks Lows',
                    marker_color=colors,
                    showlegend=True
                ))
            
            if dff['NH200'].notna().any():
                fig_diff_40w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH200'],
                    name='Percentage of Members with New 40 Weeks Highs',
                    line=dict(width=2, color='#00ff00'),
                    mode='lines',
                    showlegend=True
                ))
            
            if dff['NL200'].notna().any():
                fig_diff_40w.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NL200'],
                    name='Percentage of Members with New 40 Weeks Lows',
                    line=dict(width=2, color='#ff0000'),
                    mode='lines',
                    showlegend=True
                ))
            
            if rangebreaks:
                fig_diff_40w.update_xaxes(rangebreaks=rangebreaks)
            
            fig_diff_40w.update_layout(
                height=400,
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                yaxis_title='New Highs & New Lows (40 weeks)',
                xaxis_title='Date',
                margin=dict(l=10, r=10, t=60, b=10)
            )
            
            st.plotly_chart(fig_diff_40w, use_container_width=True, config={'displayModeBar': True})

# Footer
st.markdown("---")
st.caption(f"📊 MAI Index Dashboard | {len(dff)} data points | {dff['Date'].min().date()} to {dff['Date'].max().date()}")
# %%