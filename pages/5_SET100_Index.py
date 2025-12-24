"""
SET50 Index Market Breadth Dashboard
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
    st.set_page_config(page_title="SET50 Index - Login Required", layout="wide")
    st.error("🔒 Please log in from the home page first")
    st.page_link("Cover.py", label="← Go to Login Page", icon="🏠")
    st.stop()
    
# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="SET50 Index Market Analysis", layout="wide")

SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"
GID = "1903958263"   # <<< SET50 INDEX GID

URL_PRIMARY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
URL_FALLBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

st.title("📈 SET50 Index Market Analysis")

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
    ss = s.astype(str).str.strip()
    dt = pd.to_datetime(ss, errors="coerce", infer_datetime_format=True)

    if dt.notna().sum() < max(3, int(len(ss) * 0.2)):
        num = pd.to_numeric(ss, errors="coerce")
        if num.notna().sum() >= max(3, int(len(ss) * 0.5)) and num.median(skipna=True) > 10000:
            dt = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")

    return dt.dt.normalize()

def make_rangebreaks(dates: pd.Series, max_holidays: int = 250):
    rbs = [dict(bounds=["sat", "mon"])]
    dt = pd.to_datetime(dates, errors="coerce").dropna()
    if dt.empty:
        return rbs

    obs = pd.DatetimeIndex(dt.unique())
    bdays = pd.date_range(obs.min(), obs.max(), freq="B")
    missing = bdays.difference(obs)

    if len(missing) <= max_holidays:
        rbs.append(dict(values=list(missing)))

    return rbs

@st.cache_data(ttl=600, show_spinner=False)
def load_set50_data(url_primary: str, url_fallback: str) -> pd.DataFrame:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in (url_primary, url_fallback):
        r = session.get(url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        if not _looks_like_html(r.text):
            return pd.read_csv(StringIO(r.text))

    raise ValueError("Google Sheet did not return CSV")

def parse_set50_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = _parse_date_series(df.iloc[:, 0])

    df['Open']  = _clean_numeric_series(df.iloc[:, 1])
    df['High']  = _clean_numeric_series(df.iloc[:, 2])
    df['Low']   = _clean_numeric_series(df.iloc[:, 3])
    df['Close'] = _clean_numeric_series(df.iloc[:, 4])

    df['Above_EMA10']  = _clean_numeric_series(df.iloc[:, 5])
    df['Above_EMA20']  = _clean_numeric_series(df.iloc[:, 6])
    df['Above_EMA50']  = _clean_numeric_series(df.iloc[:, 7])
    df['Above_EMA100'] = _clean_numeric_series(df.iloc[:, 8])
    df['Above_EMA200'] = _clean_numeric_series(df.iloc[:, 9])

    df['NH20']  = _clean_numeric_series(df.iloc[:, 15])
    df['NH65']  = _clean_numeric_series(df.iloc[:, 16])
    df['NH130'] = _clean_numeric_series(df.iloc[:, 17])
    df['NH260'] = _clean_numeric_series(df.iloc[:, 18])

    df['NL20']  = _clean_numeric_series(df.iloc[:, 19])
    df['NL65']  = _clean_numeric_series(df.iloc[:, 20])
    df['NL130'] = _clean_numeric_series(df.iloc[:, 21])
    df['NL260'] = _clean_numeric_series(df.iloc[:, 22])
    
    # Double Moving Averages (Column AK-AL) - columns 36-37
    df['Percentage_Above_Both'] = _clean_numeric_series(df.iloc[:, 36]) if df.shape[1] > 36 else pd.Series([pd.NA] * len(df))
    df['Percentage_Below_Both'] = _clean_numeric_series(df.iloc[:, 37]) if df.shape[1] > 37 else pd.Series([pd.NA] * len(df))

    keep = [
        'Date', 'Open', 'High', 'Low', 'Close',
        'Above_EMA10','Above_EMA20','Above_EMA50','Above_EMA100','Above_EMA200',
        'NH20','NH65','NH130','NH260',
        'NL20','NL65','NL130','NL260',
        'Percentage_Above_Both', 'Percentage_Below_Both'
    ]

    return df[keep].dropna(subset=['Date']).sort_values('Date')

# -------------------------
# Load Data
# -------------------------
raw_df = load_set50_data(URL_PRIMARY, URL_FALLBACK)
df = parse_set50_data(raw_df)

# -------------------------
# Controls
# -------------------------
bangkok_tz = pytz.timezone("Asia/Bangkok")
last_update = pd.Timestamp.now(tz=bangkok_tz)

col1, col2, col3 = st.columns([1.5, 1.5, 3])

with col1:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with col2:
    months = st.selectbox("Time Period (month)", [1, 3, 6, 12], index=2)

with col3:
    st.info(f"⏱️ Last request: {last_update.strftime('%Y-%m-%d %H:%M:%S')} Bangkok Time")

# -------------------------
# Date Filtering
# -------------------------
st.markdown("---")
end_date = df['Date'].max()
start_date = max(df['Date'].min(), end_date - timedelta(days=30 * months))
dff = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

st.write(f"Showing: **{start_date.date()} → {end_date.date()}** ({len(dff)} data points)")

rangebreaks = make_rangebreaks(dff['Date'])

# -------------------------
# PANEL 1: SET50 Candlestick
# -------------------------
st.markdown("---")
with st.expander("📈 SET50 Index", expanded=True):
    fig = go.Figure(go.Candlestick(
        x=dff['Date'],
        open=dff['Open'],
        high=dff['High'],
        low=dff['Low'],
        close=dff['Close'],
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    fig.update_xaxes(rangebreaks=rangebreaks)
    fig.update_layout(
        height=500,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis_title='SET50 Index',
        xaxis_title='Date',
        hovermode='x unified',
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

# -------------------------
# PANEL 2: New Highs & Lows
# -------------------------
st.markdown("---")
with st.expander("📈 New Highs & Lows", expanded=True):
    
    # Panel 1: NH20 & NL20
    show_nh_nl_4w = st.checkbox("📊 New High & New Low 4 Weeks", value=True, key="show_nh_nl_4w")
    if show_nh_nl_4w:
        fig_nh_nl_4w = go.Figure()

        # Bars
        fig_nh_nl_4w.add_bar(
            x=dff['Date'], y=dff['NH20'],
            name='Number of Members with New 4 Week Highs (Bar)',
            marker_color='rgba(0,255,0,0.3)'
        )
        fig_nh_nl_4w.add_bar(
            x=dff['Date'], y=-dff['NL20'],
            name='Number of Members with New 4 Week Lows (Bar)',
            marker_color='rgba(255,107,107,0.3)'
        )

        # Lines
        fig_nh_nl_4w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NH20'],
            mode='lines',
            name='Number of Members with New 4 Week Highs (Line)',
            line=dict(color='rgb(0,255,0)', width=2)
        ))
        fig_nh_nl_4w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NL20'],
            mode='lines',
            name='Number of Members with New 4 Week Lows (Line)',
            line=dict(color='rgb(255,107,107)', width=2)
        ))

        if rangebreaks:
            fig_nh_nl_4w.update_xaxes(rangebreaks=rangebreaks)

        fig_nh_nl_4w.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            yaxis=dict(
                title='Number of Members with New Highs & New Lows (4 Weeks)',
                range=[-50, 50],
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.4)'
            ),
            xaxis_title='Date',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_nh_nl_4w, use_container_width=True, config={'displayModeBar': True})
        st.markdown("---")

    # Panel 2: NH65 & NL65
    show_nh_nl_13w = st.checkbox("📊 New High & New Low 13 Weeks", value=True, key="show_nh_nl_13w")
    if show_nh_nl_13w:
        fig_nh_nl_13w = go.Figure()

        # Bars
        fig_nh_nl_13w.add_bar(
            x=dff['Date'], y=dff['NH65'],
            name='Number of Members with New 13 Week Highs (Bar)',
            marker_color='rgba(38,166,154,0.3)'
        )
        fig_nh_nl_13w.add_bar(
            x=dff['Date'], y=-dff['NL65'],
            name='Number of Members with New 13 Week Lows (Bar)',
            marker_color='rgba(239,83,80,0.3)'
        )

        # Lines
        fig_nh_nl_13w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NH65'],
            mode='lines',
            name='Number of Members with New 13 Week Highs (Line)',
            line=dict(color='rgb(38,166,154)', width=2)
        ))
        fig_nh_nl_13w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NL65'],
            mode='lines',
            name='Number of Members with New 13 Week Lows (Line)',
            line=dict(color='rgb(239,83,80)', width=2)
        ))

        if rangebreaks:
            fig_nh_nl_13w.update_xaxes(rangebreaks=rangebreaks)

        fig_nh_nl_13w.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            yaxis=dict(
                title='Number of Members with New Highs & New Lows (13 Weeks)',
                range=[-50, 50],
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.4)'
            ),
            xaxis_title='Date',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_nh_nl_13w, use_container_width=True, config={'displayModeBar': True})
        st.markdown("---")

    # Panel 3: NH130 & NL130
    show_nh_nl_26w = st.checkbox("📊 New High & New Low 26 Weeks", value=True, key="show_nh_nl_26w")
    if show_nh_nl_26w:
        fig_nh_nl_26w = go.Figure()

        # Bars
        fig_nh_nl_26w.add_bar(
            x=dff['Date'], y=dff['NH130'],
            name='Number of Members with New 26 Week Highs (Bar)',
            marker_color='rgba(46,204,113,0.3)'
        )
        fig_nh_nl_26w.add_bar(
            x=dff['Date'], y=-dff['NL130'],
            name='Number of Members with New 26 Week Lows (Bar)',
            marker_color='rgba(231,76,60,0.3)'
        )

        # Lines
        fig_nh_nl_26w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NH130'],
            mode='lines',
            name='Number of Members with New 26 Week Highs (Line)',
            line=dict(color='rgb(46,204,113)', width=2)
        ))
        fig_nh_nl_26w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NL130'],
            mode='lines',
            name='Number of Members with New 26 Week Lows (Line)',
            line=dict(color='rgb(231,76,60)', width=2)
        ))

        if rangebreaks:
            fig_nh_nl_26w.update_xaxes(rangebreaks=rangebreaks)

        fig_nh_nl_26w.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            yaxis=dict(
                title='Number of Members with New Highs & New Lows (26 Weeks)',
                range=[-50, 50],
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.4)'
            ),
            xaxis_title='Date',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_nh_nl_26w, use_container_width=True, config={'displayModeBar': True})
        st.markdown("---")

    # Panel 4: NH260 & NL260
    show_nh_nl_52w = st.checkbox("📊 New High & New Low 52 Weeks", value=True, key="show_nh_nl_52w")
    if show_nh_nl_52w:
        fig_nh_nl_52w = go.Figure()

        # Bars
        fig_nh_nl_52w.add_bar(
            x=dff['Date'], y=dff['NH260'],
            name='Number of Members with New 52 Week Highs (Bar)',
            marker_color='rgba(39,174,96,0.3)'
        )
        fig_nh_nl_52w.add_bar(
            x=dff['Date'], y=-dff['NL260'],
            name='Number of Members with New 52 Week Lows (Bar)',
            marker_color='rgba(192,57,43,0.3)'
        )

        # Lines
        fig_nh_nl_52w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NH260'],
            mode='lines',
            name='Number of Members with New 52 Week Highs (Line)',
            line=dict(color='rgb(39,174,96)', width=2)
        ))
        fig_nh_nl_52w.add_trace(go.Scatter(
            x=dff['Date'], y=dff['NL260'],
            mode='lines',
            name='Number of Members with New 52 Week Lows (Line)',
            line=dict(color='rgb(192,57,43)', width=2)
        ))

        if rangebreaks:
            fig_nh_nl_52w.update_xaxes(rangebreaks=rangebreaks)

        fig_nh_nl_52w.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            barmode='overlay',
            yaxis=dict(
                title='Number of Members with New Highs & New Lows (52 Weeks)',
                range=[-50, 50],
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.4)'
            ),
            xaxis_title='Date',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_nh_nl_52w, use_container_width=True, config={'displayModeBar': True})

# -------------------------
# PANEL 3: Market Breadth Analysis (without New Highs & Lows tab)
# -------------------------
st.markdown("---")
with st.expander("📊 Market Breadth Analysis", expanded=True):

    tab1, tab2 = st.tabs([
        "📊 Simple Moving Averages", 
        "📊 Double Simple Moving Averages"
    ])

    # =====================================================
    # TAB 1: Moving Averages
    # =====================================================
    with tab1:
        dff_pct = dff.copy()

        for c in ['Above_EMA10','Above_EMA20','Above_EMA50','Above_EMA100','Above_EMA200']:
            dff_pct[c] = dff_pct[c] * 100

        # ---------- Moving Averages ----------
        fig_above = go.Figure()

        # Above EMA (Green)
        fig_above.add_trace(go.Scatter(
            x=dff_pct['Date'], y=dff_pct['Above_EMA10']/100,
            name='Number of Members with Px > 10 Day SMA',
            line=dict(width=1, color='#00ff00')
        ))
        fig_above.add_trace(go.Scatter(
            x=dff_pct['Date'], y=dff_pct['Above_EMA20']/100,
            name='Number of Members with Px > 20 Day SMA',
            line=dict(width=1.5, color='#26a69a')
        ))
        fig_above.add_trace(go.Scatter(
            x=dff_pct['Date'], y=dff_pct['Above_EMA50']/100,
            name='Number of Members with Px > 50 Day SMA',
            line=dict(width=2, color='#2ecc71')
        ))
        fig_above.add_trace(go.Scatter(
            x=dff_pct['Date'], y=dff_pct['Above_EMA100']/100,
            name='Number of Members with Px > 100 Day SMA',
            line=dict(width=2.5, color='#27ae60')
        ))
        fig_above.add_trace(go.Scatter(
            x=dff_pct['Date'], y=dff_pct['Above_EMA200']/100,
            name='Number of Members with Px > 200 Day SMA',
            line=dict(width=3, color='#1e8449')
        ))

        if rangebreaks:
            fig_above.update_xaxes(rangebreaks=rangebreaks)

        fig_above.update_layout(
            height=400,
            template='plotly_dark',
            hovermode='x unified',
            yaxis=dict(range=[0, 50], title='Number of Members'),
            xaxis_title='Date',
            legend=dict(
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='center', x=0.5
            ),
            margin=dict(l=10, r=10, t=60, b=10)
        )

        st.plotly_chart(fig_above, use_container_width=True, config={'displayModeBar': True})
    
    # =====================================================
    # TAB 2: Double Moving Averages
    # =====================================================
    with tab2:
        fig_dma = go.Figure()
        
        # Number Above Both (Green line)
        if dff['Percentage_Above_Both'].notna().any():
            fig_dma.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Percentage_Above_Both'],
                name='Number of Members Above 50-DMA and 200-DMA',
                line=dict(width=2, color='#00ff00'),
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.2)'
            ))
        
        # Number Below Both (Red line)
        if dff['Percentage_Below_Both'].notna().any():
            fig_dma.add_trace(go.Scatter(
                x=dff['Date'],
                y=dff['Percentage_Below_Both'],
                name='Number of Members Below 50-DMA and 200-DMA',
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
                range=[0, 50]
            ),
            xaxis_title='Date',
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        st.plotly_chart(fig_dma, use_container_width=True, config={'displayModeBar': True})
            
# Footer
st.markdown("---")
st.caption(f"📊 SET50 Index Dashboard | {len(dff)} data points | {dff['Date'].min().date()} to {dff['Date'].max().date()}")