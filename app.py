# %%writefile set_dashboard.py
"""
SET Index Market Breadth Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta, date
import requests
from io import StringIO
from typing import Optional, Tuple
import pytz

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="SET Index Market Analysis", layout="wide")

SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"
GID = "1644473343"

URL_PRIMARY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
URL_FALLBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

st.title("📈 SET Index Market Analysis")

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
def load_set_data(url_primary: str, url_fallback: str) -> pd.DataFrame:
    """Load SET data from Google Sheets"""
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

def parse_set_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean SET data"""
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
    df['Below_EMA10'] = _clean_numeric_series(df.iloc[:, 10])
    df['Below_EMA20'] = _clean_numeric_series(df.iloc[:, 11])
    df['Below_EMA50'] = _clean_numeric_series(df.iloc[:, 12])
    df['Below_EMA100'] = _clean_numeric_series(df.iloc[:, 13])
    df['Below_EMA200'] = _clean_numeric_series(df.iloc[:, 14])
    
    # New High/Low columns (P-W)
    df['NH20'] = _clean_numeric_series(df.iloc[:, 15])
    df['NH65'] = _clean_numeric_series(df.iloc[:, 16])
    df['NH130'] = _clean_numeric_series(df.iloc[:, 17])
    df['NH260'] = _clean_numeric_series(df.iloc[:, 18])
    df['NL20'] = _clean_numeric_series(df.iloc[:, 19])
    df['NL65'] = _clean_numeric_series(df.iloc[:, 20])
    df['NL130'] = _clean_numeric_series(df.iloc[:, 21])
    df['NL260'] = _clean_numeric_series(df.iloc[:, 22])
    
    # Keep only processed columns
    processed_cols = ['Date', 'Open', 'High', 'Low', 'Close',
                      'Above_EMA10', 'Above_EMA20', 'Above_EMA50', 'Above_EMA100', 'Above_EMA200',
                      'Below_EMA10', 'Below_EMA20', 'Below_EMA50', 'Below_EMA100', 'Below_EMA200',
                      'NH20', 'NH65', 'NH130', 'NH260',
                      'NL20', 'NL65', 'NL130', 'NL260']
    
    df = df[processed_cols].dropna(subset=['Date']).sort_values('Date')
    
    return df

# -------------------------
# Load Data
# -------------------------
try:
    raw_df = load_set_data(URL_PRIMARY, URL_FALLBACK)
    df = parse_set_data(raw_df)
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
# First Panel: SET Index Candlestick
# -------------------------
st.markdown("---")
with st.expander("📈 SET Index", expanded=True):
    fig1 = go.Figure(
        go.Candlestick(
            x=dff['Date'],
            open=dff['Open'],
            high=dff['High'],
            low=dff['Low'],
            close=dff['Close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name='SET Index'
        )
    )
    
    # Apply rangebreaks to remove gaps
    if rangebreaks:
        fig1.update_xaxes(rangebreaks=rangebreaks)
    
    fig1.update_layout(
        height=500,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis_title='SET Index',
        xaxis_title='Date',
        hovermode='x unified',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': True})

# -------------------------
# Second Panel: Market Breadth Analysis
# -------------------------
st.markdown("---")
with st.expander("📊 Market Breadth Analysis", expanded=True):
    tab1, tab2 = st.tabs(["📊 Moving Averages", "📈 New Highs & New Lows"])
    
# Tab 1: Moving Averages
    with tab1:
        # Multiply by 100 to convert to percentage
        dff_pct = dff.copy()
        dff_pct['Above_EMA10'] = dff['Above_EMA10'] * 100
        dff_pct['Above_EMA20'] = dff['Above_EMA20'] * 100
        dff_pct['Above_EMA50'] = dff['Above_EMA50'] * 100
        dff_pct['Above_EMA100'] = dff['Above_EMA100'] * 100
        dff_pct['Above_EMA200'] = dff['Above_EMA200'] * 100
        dff_pct['Below_EMA10'] = dff['Below_EMA10'] * 100
        dff_pct['Below_EMA20'] = dff['Below_EMA20'] * 100
        dff_pct['Below_EMA50'] = dff['Below_EMA50'] * 100
        dff_pct['Below_EMA100'] = dff['Below_EMA100'] * 100
        dff_pct['Below_EMA200'] = dff['Below_EMA200'] * 100
        
        fig2 = go.Figure()
        
        # Above EMA lines (green shades) - First line in legend
        if dff_pct['Above_EMA10'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA10'],
                name='Percentage of Members with Px > 10 Day Exponential Moving Average',
                line=dict(width=1, color='#00ff00'),
                mode='lines',
                legendgroup='above',
                legendrank=1
            ))
        
        if dff_pct['Above_EMA20'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA20'],
                name='Percentage of Members with Px > 20 Day Exponential Moving Average',
                line=dict(width=1.5, color='#26a69a'),
                mode='lines',
                legendgroup='above',
                legendrank=2
            ))
        
        if dff_pct['Above_EMA50'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA50'],
                name='Percentage of Members with Px > 50 Day Exponential Moving Average',
                line=dict(width=2, color='#2ecc71'),
                mode='lines',
                legendgroup='above',
                legendrank=3
            ))
        
        if dff_pct['Above_EMA100'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA100'],
                name='Percentage of Members with Px > 100 Day Exponential Moving Average',
                line=dict(width=2.5, color='#27ae60'),
                mode='lines',
                legendgroup='above',
                legendrank=4
            ))
        
        if dff_pct['Above_EMA200'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Above_EMA200'],
                name='Percentage of Members with Px > 200 Day Exponential Moving Average',
                line=dict(width=3, color='#1e8449'),
                mode='lines',
                legendgroup='above',
                legendrank=5
            ))
        
        # Below EMA lines (red shades) - Second line in legend
        if dff_pct['Below_EMA10'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Below_EMA10'],
                name='Percentage of Members with Px < 10 Day Exponential Moving Average',
                line=dict(width=1, color='#ff6b6b', dash='solid'),
                mode='lines',
                legendgroup='below',
                legendrank=6
            ))
        
        if dff_pct['Below_EMA20'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Below_EMA20'],
                name='Percentage of Members with Px < 20 Day Exponential Moving Average',
                line=dict(width=1.5, color='#ef5350', dash='solid'),
                mode='lines',
                legendgroup='below',
                legendrank=7
            ))
        
        if dff_pct['Below_EMA50'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Below_EMA50'],
                name='Percentage of Members with Px < 50 Day Exponential Moving Average',
                line=dict(width=2, color='#e74c3c', dash='solid'),
                mode='lines',
                legendgroup='below',
                legendrank=8
            ))
        
        if dff_pct['Below_EMA100'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Below_EMA100'],
                name='Percentage of Members with Px < 100 Day Exponential Moving Average',
                line=dict(width=2.5, color='#c0392b', dash='solid'),
                mode='lines',
                legendgroup='below',
                legendrank=9
            ))
        
        if dff_pct['Below_EMA200'].notna().any():
            fig2.add_trace(go.Scatter(
                x=dff_pct['Date'],
                y=dff_pct['Below_EMA200'],
                name='Percentage of Members with Px < 200 Day Exponential Moving Average',
                line=dict(width=3, color='#a93226', dash='solid'),
                mode='lines',
                legendgroup='below',
                legendrank=10
            ))
        
        # Apply rangebreaks to remove gaps
        if rangebreaks:
            fig2.update_xaxes(rangebreaks=rangebreaks)
        
        fig2.update_layout(
            height=550,
            template='plotly_dark',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                traceorder='normal'
            ),
            yaxis=dict(
                title='Moving Averages',
                range=[0, 100]
            ),
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=80, b=10)
        )
        
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': True})
    
# Tab 2: New Highs & Lows with 4 collapsible panels
    with tab2:
        # Panel 1: NH20 & NL20
        show_nh_nl_20 = st.checkbox("📊 New High & New Low 4 Weeks", value=True, key="show_nh_nl_20")
        if show_nh_nl_20:
            fig_nh_nl_20 = go.Figure()
            
            # Bar - New High 20
            if dff['NH20'].notna().any():
                fig_nh_nl_20.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NH20'],
                    name='Percentage of Members with New 4 Week Highs',
                    marker_color='rgba(0, 255, 0, 0.3)',
                    showlegend=True
                ))
            
            # Bar - New Low 20 (negative for display)
            if dff['NL20'].notna().any():
                fig_nh_nl_20.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NL20'] * -1,
                    name='Percentage of Members with New 4 Week Lows',
                    marker_color='rgba(255, 107, 107, 0.3)',
                    showlegend=True
                ))
            
            # Line - New High 20
            if dff['NH20'].notna().any():
                fig_nh_nl_20.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH20'],
                    name='Percentage of Members with New 4 Week Highs (Line)',
                    line=dict(width=2, color='#00ff00'),
                    mode='lines',
                    showlegend=True
                ))
            
            # Line - New Low 20 (positive values, no multiplication)
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
        show_nh_nl_65 = st.checkbox("📊 New High & New Low 13 Weeks", value=False, key="show_nh_nl_65")
        if show_nh_nl_65:
            fig_nh_nl_65 = go.Figure()
            
            # Bar - New High 65
            if dff['NH65'].notna().any():
                fig_nh_nl_65.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NH65'],
                    name='Percentage of Members with New 13 Week Highs',
                    marker_color='rgba(38, 166, 154, 0.3)',
                    showlegend=True
                ))
            
            # Bar - New Low 65 (negative for display)
            if dff['NL65'].notna().any():
                fig_nh_nl_65.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NL65'] * -1,
                    name='Percentage of Members with New 13 Week Lows',
                    marker_color='rgba(239, 83, 80, 0.3)',
                    showlegend=True
                ))
            
            # Line - New High 65
            if dff['NH65'].notna().any():
                fig_nh_nl_65.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH65'],
                    name='Percentage of Members with New 13 Week Highs (Line)',
                    line=dict(width=2, color='#26a69a'),
                    mode='lines',
                    showlegend=True
                ))
            
            # Line - New Low 65 (positive values, no multiplication)
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
        show_nh_nl_130 = st.checkbox("📊 New High & New Low 26 Weeks", value=False, key="show_nh_nl_130")
        if show_nh_nl_130:
            fig_nh_nl_130 = go.Figure()
            
            # Bar - New High 130
            if dff['NH130'].notna().any():
                fig_nh_nl_130.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NH130'],
                    name='Percentage of Members with New 26 Week Highs',
                    marker_color='rgba(46, 204, 113, 0.3)',
                    showlegend=True
                ))
            
            # Bar - New Low 130 (negative for display)
            if dff['NL130'].notna().any():
                fig_nh_nl_130.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NL130'] * -1,
                    name='Percentage of Members with New 26 Week Lows',
                    marker_color='rgba(231, 76, 60, 0.3)',
                    showlegend=True
                ))
            
            # Line - New High 130
            if dff['NH130'].notna().any():
                fig_nh_nl_130.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH130'],
                    name='Percentage of Members with New 26 Week Highs (Line)',
                    line=dict(width=2, color='#2ecc71'),
                    mode='lines',
                    showlegend=True
                ))
            
            # Line - New Low 130 (positive values, no multiplication)
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
        show_nh_nl_260 = st.checkbox("📊 New High & New Low 52 Weeks", value=False, key="show_nh_nl_260")
        if show_nh_nl_260:
            fig_nh_nl_260 = go.Figure()
            
            # Bar - New High 260
            if dff['NH260'].notna().any():
                fig_nh_nl_260.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NH260'],
                    name='Percentage of Members with New 52 Week Highs',
                    marker_color='rgba(39, 174, 96, 0.3)',
                    showlegend=True
                ))
            
            # Bar - New Low 260 (negative for display)
            if dff['NL260'].notna().any():
                fig_nh_nl_260.add_trace(go.Bar(
                    x=dff['Date'],
                    y=dff['NL260'] * -1,
                    name='Percentage of Members with New 52 Week Lows',
                    marker_color='rgba(192, 57, 43, 0.3)',
                    showlegend=True
                ))
            
            # Line - New High 260
            if dff['NH260'].notna().any():
                fig_nh_nl_260.add_trace(go.Scatter(
                    x=dff['Date'],
                    y=dff['NH260'],
                    name='Percentage of Members with New 52 Week Highs (Line)',
                    line=dict(width=2.5, color='#27ae60'),
                    mode='lines',
                    showlegend=True
                ))
            
            # Line - New Low 260 (positive values, no multiplication)
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

# Footer
st.markdown("---")
st.caption(f"📊 SET Index Dashboard | {len(dff)} data points | {dff['Date'].min().date()} to {dff['Date'].max().date()}")