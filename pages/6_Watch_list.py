# pages/6_Watch_list.py
"""
Watchlist Page (Read-only Table + Filters)
Data source: Google Sheet (WATCHLIST) gid=1057696914

Excel-like Freeze Column:
- Pin "Symbol" column on the left using AgGrid (streamlit-aggrid)
Install:
    pip install streamlit-aggrid
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import pytz

# ✅ AgGrid for pinned (frozen) columns
from st_aggrid import AgGrid, GridOptionsBuilder

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

# -------------------------
# Google Sheet Config
# -------------------------
SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"
GID = "1057696914"  # WATCHLIST sheet gid

URL_PRIMARY = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
URL_FALLBACK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}"

st.title("⭐ Watchlist")

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

def _try_parse_datetime_series(s: pd.Series) -> pd.Series:
    # Attempt to parse datetime; if fails mostly, return original series
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt.notna().sum() >= max(3, int(len(s) * 0.6)):
        return dt
    return s

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]
    # Drop empty columns
    df = df.dropna(axis=1, how="all")
    # Strip string cells
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan": "", "None": "", "null": ""})
            # Try parse dates for object columns
            df[c] = _try_parse_datetime_series(df[c])
    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_watchlist_data(url_primary: str, url_fallback: str) -> pd.DataFrame:
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
            return _normalize_df(df)
        except Exception as e:
            last_err = str(e)

    raise ValueError(f"Google Sheet did not return CSV. Last error: {last_err}")

def apply_global_search(df: pd.DataFrame, q: str) -> pd.DataFrame:
    q = (q or "").strip().lower()
    if not q:
        return df
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        col = df[c].astype(str).str.lower()
        mask = mask | col.str.contains(q, na=False)
    return df[mask]

def build_filters(df: pd.DataFrame):
    """
    Build automatic filters:
    - For object/categorical columns with <= 50 unique values -> multiselect
    - For numeric -> range slider
    - For datetime -> date range
    Returns: filtered df
    """
    st.sidebar.header("Filters")

    # Global search
    q = st.sidebar.text_input("Search (all columns)", value="", placeholder="e.g., PTT, banking, breakout...")
    out = apply_global_search(df, q)

    # Choose columns to filter (optional)
    with st.sidebar.expander("Column Filters", expanded=True):
        cols_to_filter = st.multiselect(
            "Select columns to filter",
            options=list(out.columns),
            default=[c for c in out.columns[: min(5, len(out.columns))]],
        )

    for c in cols_to_filter:
        if c not in out.columns:
            continue

        s = out[c]

        # Datetime filter
        if pd.api.types.is_datetime64_any_dtype(s):
            min_dt = s.min()
            max_dt = s.max()
            if pd.isna(min_dt) or pd.isna(max_dt):
                continue

            min_d = min_dt.date()
            max_d = max_dt.date()
            d1, d2 = st.sidebar.date_input(
                f"{c} (date range)",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )
            if isinstance(d1, (tuple, list)):
                d1, d2 = d1
            if d1 and d2:
                out = out[(out[c].dt.date >= d1) & (out[c].dt.date <= d2)]

        # Numeric filter
        elif pd.api.types.is_numeric_dtype(s):
            s2 = pd.to_numeric(s, errors="coerce")
            if s2.notna().sum() == 0:
                continue
            mn = float(s2.min())
            mx = float(s2.max())
            if mn == mx:
                continue
            lo, hi = st.sidebar.slider(f"{c} (range)", mn, mx, (mn, mx))
            out = out[(s2 >= lo) & (s2 <= hi)]

        # Categorical/text filter
        else:
            vals = pd.Series(s.astype(str)).replace({"nan": ""})
            uniq = sorted([v for v in vals.unique() if v != ""])
            if 1 <= len(uniq) <= 50:
                chosen = st.sidebar.multiselect(f"{c}", options=uniq, default=[])
                if chosen:
                    out = out[out[c].astype(str).isin(chosen)]
            else:
                txt = st.sidebar.text_input(f"{c} contains", value="", placeholder="type to filter…")
                if txt.strip():
                    out = out[out[c].astype(str).str.contains(txt, case=False, na=False)]

    return out, q, cols_to_filter

def render_table_with_frozen_symbol(df_to_show: pd.DataFrame, height: int = 650):
    """
    Render table with Excel-like frozen column:
    - Pins (freezes) 'Symbol' column on the left if it exists
    """
    gb = GridOptionsBuilder.from_dataframe(df_to_show)

    # make it spreadsheet-friendly
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    # ✅ Freeze "Symbol"
    if "Symbol" in df_to_show.columns:
        gb.configure_column("Symbol", pinned="left")

    # optional: nicer horizontal scroll behavior
    gb.configure_grid_options(domLayout="normal")

    grid_options = gb.build()

    AgGrid(
        df_to_show,
        gridOptions=grid_options,
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=False,
        theme="streamlit",  # keeps it close to Streamlit style
    )

# -------------------------
# Load Data
# -------------------------
try:
    df = load_watchlist_data(URL_PRIMARY, URL_FALLBACK)
except Exception as e:
    st.error(f"Failed to load Watchlist data: {e}")
    st.stop()

if df.empty:
    st.warning("Watchlist sheet is empty.")
    st.stop()

# -------------------------
# Header Controls
# -------------------------
bangkok_tz = pytz.timezone("Asia/Bangkok")
now_bkk = pd.Timestamp.now(tz=bangkok_tz)

c1, c2, c3 = st.columns([1.2, 2.2, 2.6])
with c1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with c2:
    st.caption("Data Source: Google Sheets → WATCHLIST")
with c3:
    st.info(f"⏱️ Last request: {now_bkk.strftime('%Y-%m-%d %H:%M:%S')} Bangkok Time")

st.markdown("---")

# -------------------------
# Filters + Table
# -------------------------
filtered, q, cols_used = build_filters(df)

top1, top2 = st.columns([1.3, 1.7])
with top1:
    st.metric("Rows", f"{len(filtered):,} / {len(df):,}")
with top2:
    st.caption(f"Active filters: Search='{q}' | Columns={', '.join(cols_used) if cols_used else 'None'}")

# -------------------------
# Show table (Frozen "Symbol" column)
# -------------------------
render_table_with_frozen_symbol(filtered, height=650)

# -------------------------
# Optional: Export filtered table
# -------------------------
st.download_button(
    "⬇️ Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="watchlist_filtered.csv",
    mime="text/csv",
)
