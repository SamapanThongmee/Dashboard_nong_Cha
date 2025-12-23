"""
Watchlist Page for SET100 Index Market Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import requests
from io import StringIO
import pytz

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Watchlist", layout="wide")