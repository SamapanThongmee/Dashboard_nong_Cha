"""
Watchlist Page for SET100 Index Market Analysis Dashboard
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
    st.set_page_config(page_title="Watchlist - Login Required", layout="wide")
    st.error("🔒 Please log in from the home page first")
    st.page_link("Cover.py", label="← Go to Login Page", icon="🏠")
    st.stop()

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Watchlist", layout="wide")