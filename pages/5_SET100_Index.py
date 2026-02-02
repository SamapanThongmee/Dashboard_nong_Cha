"""
SET100 Index Market Breadth Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import requests
from io import StringIO
import pytz

# =====================================================
# AUTHENTICATION CHECK
# =====================================================
if "password_correct" not in st.session_state or not st.session_state.get("password_correct", False):
    st.set_page_config(page_title="SET100 Index - Login Required", layout="wide")
    st.error("🔒 Please log in from the home page first")
    st.page_link("Cover.py", label="← Go to Login Page", icon="🏠")
    st.stop()

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="SET100 Index Market Analysis", layout="wide")

SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"
GID = "80578723"   # SET100 GID

URL_PRIMARY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
URL_FALLBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

st.title("📈 SET100 Index Market Analysis")

# -------------------------
# Helper Functions
# -------------------------
def _looks_like_html(text: str) -> bool:
    t = (text or "").lower()
    return "<html" in t or "accounts.google.com" in t

def _clean_numeric_series(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False),
        errors="coerce"
    )

def make_rangebreaks(dates):
    return [dict(bounds=["sat", "mon"])]

@st.cache_data(ttl=600)
def load_data():
    for url in (URL_PRIMARY, URL_FALLBACK):
        r = requests.get(url, timeout=20)
        if not _looks_like_html(r.text):
            return pd.read_csv(StringIO(r.text))
    raise ValueError("CSV load failed")

def parse_data(df):
    df['Date'] = pd.to_datetime(df.iloc[:, 0])
    for i, c in enumerate(
        ['Open','High','Low','Close',
         'Above_EMA10','Above_EMA20','Above_EMA50','Above_EMA100','Above_EMA200'],
        start=1
    ):
        df[c] = _clean_numeric_series(df.iloc[:, i])

    for c, idx in zip(
        ['NH20','NH65','NH130','NH260','NL20','NL65','NL130','NL260'],
        [15,16,17,18,19,20,21,22]
    ):
        df[c] = _clean_numeric_series(df.iloc[:, idx])

    df['Percentage_Above_Both'] = _clean_numeric_series(df.iloc[:, 36])
    df['Percentage_Below_Both'] = _clean_numeric_series(df.iloc[:, 37])
    return df.sort_values('Date')

df = parse_data(load_data())

# -------------------------
# Controls
# -------------------------
months = st.selectbox("Time Period (month)", [1, 3, 6, 12], index=2)

end = df['Date'].max()
start = end - timedelta(days=30 * months)
dff = df[df['Date'].between(start, end)]
rangebreaks = make_rangebreaks(dff['Date'])

# -------------------------
# PANEL 1: Candlestick
# -------------------------
with st.expander("📈 SET100 Index", expanded=True):
    fig = go.Figure(go.Candlestick(
        x=dff['Date'],
        open=dff['Open'],
        high=dff['High'],
        low=dff['Low'],
        close=dff['Close']
    ))
    fig.update_xaxes(rangebreaks=rangebreaks)
    fig.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# PANEL 2: New Highs & Lows
# -------------------------
with st.expander("📈 New Highs & Lows", expanded=True):

    COLOR = {
        "20": ("#0066CC", "#FF8C00"),
        "65": ("#0B1A38", "#E51C23"),
        "130": ("#259B24", "#9E9E9E"),
        "260": ("#00CC66", "#FF3399"),
    }

    for label, nh, nl in [
        ("20 Days","NH20","NL20"),
        ("65 Days","NH65","NL65"),
        ("130 Days","NH130","NL130"),
        ("260 Days","NH260","NL260"),
    ]:
        fig = go.Figure()
        c1, c2 = COLOR[label.split()[0]]
        fig.add_bar(x=dff['Date'], y=dff[nh], marker_color=c1, opacity=0.3, name=f"New High {label}")
        fig.add_bar(x=dff['Date'], y=-dff[nl], marker_color=c2, opacity=0.3, name=f"New Low {label}")
        fig.add_scatter(x=dff['Date'], y=dff[nh], line=dict(color=c1), name=f"New High {label}")
        fig.add_scatter(x=dff['Date'], y=dff[nl], line=dict(color=c2), name=f"New Low {label}")
        fig.update_xaxes(rangebreaks=rangebreaks)
        fig.update_layout(
            height=400,
            template="plotly_dark",
            yaxis=dict(range=[-100,100]),
            title=f"New High & New Low {label}"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# PANEL 3: Market Breadth Analysis
# -------------------------
with st.expander("📊 Market Breadth Analysis", expanded=True):

    tab1, tab2 = st.tabs(["📊 Moving Averages", "📊 Double Moving Averages"])

    # Moving Averages
    with tab1:
        fig = go.Figure()
        fig.add_scatter(x=dff['Date'], y=dff['Above_EMA10'], name="Px > 10 Day Moving Average", line=dict(color="#81D4FA"))
        fig.add_scatter(x=dff['Date'], y=dff['Above_EMA20'], name="Px > 20 Day Moving Average", line=dict(color="#fb8c00"))
        fig.add_scatter(x=dff['Date'], y=dff['Above_EMA50'], name="Px > 50 Day Moving Average", line=dict(color="#259b24"))
        fig.add_scatter(x=dff['Date'], y=dff['Above_EMA100'], name="Px > 100 Day Moving Average", line=dict(color="#e51c23"))
        fig.add_scatter(x=dff['Date'], y=dff['Above_EMA200'], name="Px > 200 Day Moving Average", line=dict(color="#512da8"))
        fig.update_xaxes(rangebreaks=rangebreaks)
        fig.update_layout(height=400, template="plotly_dark", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

    # Double Moving Averages
    with tab2:
        fig = go.Figure()
        fig.add_scatter(
            x=dff['Date'], y=dff['Percentage_Above_Both'],
            fill='tozeroy', line=dict(color="#00ff00"),
            name="Above 50 & 200 DMA"
        )
        fig.add_scatter(
            x=dff['Date'], y=dff['Percentage_Below_Both'],
            fill='tozeroy', line=dict(color="#ff0000"),
            name="Below 50 & 200 DMA"
        )
        fig.update_xaxes(rangebreaks=rangebreaks)
        fig.update_layout(height=450, template="plotly_dark", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption(f"📊 SET100 Index Dashboard | {len(dff)} data points")
