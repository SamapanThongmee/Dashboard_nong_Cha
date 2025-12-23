# pages/3_RRG.py
"""
Relative Rotation Graph (RRG) Dashboard for SET Market
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
import pytz

# =====================================================
# AUTHENTICATION CHECK - MUST BE AT THE TOP
# =====================================================
if "password_correct" not in st.session_state or not st.session_state.get("password_correct", False):
    st.set_page_config(page_title="RRG - Login Required", layout="wide")
    st.error("🔒 Please log in from the home page first")
    st.page_link("Cover.py", label="← Go to Login Page", icon="🏠")
    st.stop()

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="RRG Analysis", layout="wide")

st.title("🔄 Relative Rotation Graph Analysis")

# # Navigation buttons
# col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
# with col1:
#     st.page_link("Cover.py", label="← Home", icon="🏠")
# with col2:
#     st.page_link("pages/1_SET_Index.py", label="SET Index", icon="📊")
# with col3:
#     st.page_link("pages/2_MAI_Index.py", label="MAI Index", icon="📈")

# st.markdown("---")

SHEET_ID = "1faOXwIk7uR51IIeAMrrRPdorRsO7iJ3PDPn-mk5vc24"

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

@st.cache_data(ttl=600, show_spinner=False)
def load_rrg_df(sheet_id: str, gid: str) -> pd.DataFrame:
    """Load RRG data from Google Sheets"""
    url_primary = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    url_fallback = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    
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
    
    raise ValueError(f"RRG data load failed. Last error: {last_err}")

# -------------------------
# Controls
# -------------------------
bangkok_tz = pytz.timezone('Asia/Bangkok')
last_update = pd.Timestamp.now(tz=bangkok_tz)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col2:
    st.info(f"⏱️ Last request: {last_update.strftime('%Y-%m-%d %H:%M:%S')} Bangkok Time")

st.markdown("---")

# -------------------------
# Panel 1: RRG by Sector
# -------------------------
RRG_SECTOR_GID = "810916019"

with st.expander("📊 Relative Rotation Graph by Industry", expanded=True):
    try:
        rrg_sector_df = load_rrg_df(SHEET_ID, RRG_SECTOR_GID)
        
        # Parse date column
        rrg_sector_df['Date'] = pd.to_datetime(rrg_sector_df['Date'], errors='coerce')
        rrg_sector_df = rrg_sector_df.dropna(subset=['Date']).sort_values('Date')
        
        if rrg_sector_df.empty:
            st.warning("No valid RRG Sector data available")
        else:
            # Get available dates (last 20 data points)
            available_dates = rrg_sector_df['Date'].unique()
            num_points = min(20, len(available_dates))  # Changed from 50 to 20
            recent_dates = available_dates[-num_points:]
            
            # Create sliders for date selection and trail length
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                date_index_sector = st.slider(
                    "Select Date (0 = oldest, {} = most recent)".format(num_points - 1),
                    min_value=0,
                    max_value=num_points - 1,
                    value=num_points - 1,
                    step=1,
                    key="rrg_sector_date_slider"
                )
            
            with col2:
                selected_date_sector = recent_dates[date_index_sector]
                st.metric("Selected Date", pd.Timestamp(selected_date_sector).strftime('%Y-%m-%d'))
            
            with col3:
                trail_length_sector = st.slider(
                    "Trail Length",
                    min_value=0,
                    max_value=5,  # Changed from 10 to 5
                    value=0,  # Changed from 5 to 0
                    step=1,
                    key="trail_length_sector_slider"
                )
            
            # Get data for selected date
            selected_data_sector = rrg_sector_df[rrg_sector_df['Date'] == selected_date_sector].iloc[0]
            
            # Get historical data for trails
            if trail_length_sector > 0:
                trail_start_index = max(0, date_index_sector - trail_length_sector + 1)
                trail_dates = recent_dates[trail_start_index:date_index_sector + 1]
                trail_df_sector = rrg_sector_df[rrg_sector_df['Date'].isin(trail_dates)]
            else:
                trail_df_sector = pd.DataFrame()
            
            # Define sectors - find columns ending with "JdK RS-Momentum" and "JdK RS-Ratio"
            sectors = {}
            for col in rrg_sector_df.columns:
                if 'JdK RS-Momentum' in str(col):
                    sector_name = str(col).replace(' JdK RS-Momentum', '').strip()
                    ratio_col = f'{sector_name} JdK RS-Ratio'
                    if ratio_col in rrg_sector_df.columns:
                        sectors[sector_name] = (col, ratio_col)
            
            # -------------------------------------------------
            # Generate distinct colors for each sector (LIKE INDUSTRY)
            # -------------------------------------------------
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            num_sectors = len(sectors)
            cmap = plt.get_cmap('tab10' if num_sectors <= 10 else 'tab20')

            sector_colors = {}
            for idx, sector_name in enumerate(sorted(sectors.keys())):
                rgba = cmap(idx % cmap.N)
                sector_colors[sector_name] = mcolors.rgb2hex(rgba)
            
            base_color = sector_colors.get(sector_name, '#95a5a6')
            h = base_color.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            
            color = sector_colors.get(sector_name, '#95a5a6')


            
            # Prepare data for plotting
            plot_data_sector = []
            for sector_name, (momentum_col, ratio_col) in sectors.items():
                try:
                    momentum = pd.to_numeric(selected_data_sector[momentum_col], errors='coerce')
                    ratio = pd.to_numeric(selected_data_sector[ratio_col], errors='coerce')
                    
                    if pd.notna(momentum) and pd.notna(ratio):
                        plot_data_sector.append({
                            'Sector': sector_name,
                            'RS_Momentum': momentum,
                            'RS_Ratio': ratio
                        })
                except Exception as e:
                    st.warning(f"Could not parse {sector_name}: {e}")
            
            if plot_data_sector:
                plot_df_sector = pd.DataFrame(plot_data_sector)
                
                # Create RRG scatter plot
                fig_rrg_sector = go.Figure()
                
                # Fixed axis ranges
                x_min, x_max = 95, 105
                y_min, y_max = 95, 105
                
                # Add quadrant background colors
                fig_rrg_sector.add_shape(type="rect", x0=100, y0=100, x1=x_max, y1=y_max,
                                 fillcolor="#d5f4e6", opacity=0.3, layer="below", line_width=0)
                fig_rrg_sector.add_shape(type="rect", x0=x_min, y0=100, x1=100, y1=y_max,
                                 fillcolor="#d6eaf8", opacity=0.3, layer="below", line_width=0)
                fig_rrg_sector.add_shape(type="rect", x0=x_min, y0=y_min, x1=100, y1=100,
                                 fillcolor="#fadbd8", opacity=0.3, layer="below", line_width=0)
                fig_rrg_sector.add_shape(type="rect", x0=100, y0=y_min, x1=x_max, y1=100,
                                 fillcolor="#fdeaa8", opacity=0.3, layer="below", line_width=0)
                
                # Add trails with gradient
                if trail_length_sector > 0 and not trail_df_sector.empty:
                    for sector_name, (momentum_col, ratio_col) in sectors.items():
                        trail_x = []
                        trail_y = []
                        
                        for _, row in trail_df_sector.iterrows():
                            momentum = pd.to_numeric(row[momentum_col], errors='coerce')
                            ratio = pd.to_numeric(row[ratio_col], errors='coerce')
                            
                            if pd.notna(momentum) and pd.notna(ratio):
                                trail_x.append(ratio)
                                trail_y.append(momentum)
                        
                        if len(trail_x) > 1:
                            base_color = sector_colors.get(sector_name, '#95a5a6')
                            h = base_color.lstrip('#')
                            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                            
                            num_segments = len(trail_x) - 1
                            for i in range(num_segments):
                                position = (i + 1) / num_segments
                                line_width = 1 + (position * 5)
                                opacity = 0.2 + (position * 0.6)
                                color_with_opacity = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
                                
                                fig_rrg_sector.add_trace(
                                    go.Scatter(
                                        x=[trail_x[i], trail_x[i+1]],
                                        y=[trail_y[i], trail_y[i+1]],
                                        mode='lines',
                                        line=dict(width=line_width, color=color_with_opacity),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    )
                                )
                
                # Add quadrant lines
                fig_rrg_sector.add_hline(y=100, line_dash="solid", line_color="#7f8c8d", line_width=2, opacity=0.8)
                fig_rrg_sector.add_vline(x=100, line_dash="solid", line_color="#7f8c8d", line_width=2, opacity=0.8)
                
                # Add current position markers
                for _, row in plot_df_sector.iterrows():
                    sector_name = row['Sector']
                    color = sector_colors.get(sector_name, '#95a5a6')
                    
                    fig_rrg_sector.add_trace(
                        go.Scatter(
                            x=[row['RS_Ratio']],
                            y=[row['RS_Momentum']],
                            mode='markers+text',
                            marker=dict(size=16, color=color, line=dict(width=3, color='white'), symbol='circle'),
                            text=sector_name,
                            textposition='top center',
                            textfont=dict(size=12, color='#2c3e50', family='Arial Black'),
                            hovertemplate='<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<br><extra></extra>',
                            showlegend=False
                        )
                    )
                
                # Add quadrant labels
                fig_rrg_sector.add_annotation(x=104, y=104, text="Leading", showarrow=False,
                                      font=dict(size=22, color="#27ae60", family="Arial Black"))
                fig_rrg_sector.add_annotation(x=96, y=104, text="Improving", showarrow=False,
                                      font=dict(size=22, color="#2980b9", family="Arial Black"))
                fig_rrg_sector.add_annotation(x=96, y=96, text="Lagging", showarrow=False,
                                      font=dict(size=22, color="#c0392b", family="Arial Black"))
                fig_rrg_sector.add_annotation(x=104, y=96, text="Weakening", showarrow=False,
                                      font=dict(size=22, color="#d68910", family="Arial Black"))
                
                fig_rrg_sector.update_layout(
                    height=700,
                    template="plotly_white",
                    plot_bgcolor='#f8f9fa',
                    paper_bgcolor='white',
                    xaxis_title="JdK RS-Ratio",
                    yaxis_title="JdK RS-Momentum",
                    xaxis=dict(range=[x_min, x_max], zeroline=False, gridcolor='#ecf0f1'),
                    yaxis=dict(range=[y_min, y_max], zeroline=False, gridcolor='#ecf0f1'),
                    hovermode="closest",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig_rrg_sector, use_container_width=True, config={"displayModeBar": True})
                
    except Exception as e:
        st.error(f"Failed to load RRG Sector data: {e}")

st.markdown("---")

# -------------------------
# Panel 2: RRG by Industry
# -------------------------
RRG_INDUSTRY_GID = "983751635"

with st.expander("📊 Relative Rotation Graph by Sector", expanded=False):
    try:
        rrg_industry_df = load_rrg_df(SHEET_ID, RRG_INDUSTRY_GID)
        
        rrg_industry_df['Date'] = pd.to_datetime(rrg_industry_df['Date'], errors='coerce')
        rrg_industry_df = rrg_industry_df.dropna(subset=['Date']).sort_values('Date')
        
        if rrg_industry_df.empty:
            st.warning("No valid RRG Industry data available")
        else:
            available_dates = rrg_industry_df['Date'].unique()
            num_points = min(20, len(available_dates))  # Changed from 50 to 20
            recent_dates = available_dates[-num_points:]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                date_index_industry = st.slider(
                    "Select Date (0 = oldest, {} = most recent)".format(num_points - 1),
                    min_value=0,
                    max_value=num_points - 1,
                    value=num_points - 1,
                    step=1,
                    key="rrg_industry_date_slider"
                )
            
            with col2:
                selected_date_industry = recent_dates[date_index_industry]
                st.metric("Selected Date", pd.Timestamp(selected_date_industry).strftime('%Y-%m-%d'))
            
            with col3:
                trail_length_industry = st.slider(
                    "Trail Length",
                    min_value=0,
                    max_value=5,  # Changed from 10 to 5
                    value=0,  # Changed from 5 to 0
                    step=1,
                    key="trail_length_industry_slider"
                )
            
            selected_data_industry = rrg_industry_df[rrg_industry_df['Date'] == selected_date_industry].iloc[0]
            
            if trail_length_industry > 0:
                trail_start_index = max(0, date_index_industry - trail_length_industry + 1)
                trail_dates = recent_dates[trail_start_index:date_index_industry + 1]
                trail_df_industry = rrg_industry_df[rrg_industry_df['Date'].isin(trail_dates)]
            else:
                trail_df_industry = pd.DataFrame()
            
            # Define industries
            industries = {}
            for col in rrg_industry_df.columns:
                if 'JdK RS-Momentum' in str(col):
                    industry_name = str(col).replace(' JdK RS-Momentum', '').strip()
                    ratio_col = f'{industry_name} JdK RS-Ratio'
                    if ratio_col in rrg_industry_df.columns:
                        industries[industry_name] = (col, ratio_col)
            
            # Generate different colors for each industry
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            # Get a colormap and generate distinct colors
            num_industries = len(industries)
            cmap = plt.get_cmap('tab20')  # Use tab20 colormap for up to 20 distinct colors
            industry_colors = {}
            for idx, industry_name in enumerate(sorted(industries.keys())):
                color_rgba = cmap(idx % 20)  # Cycle through 20 colors if more than 20 industries
                color_hex = mcolors.rgb2hex(color_rgba)
                industry_colors[industry_name] = color_hex
            
            # Prepare data
            plot_data_industry = []
            for industry_name, (momentum_col, ratio_col) in industries.items():
                try:
                    momentum = pd.to_numeric(selected_data_industry[momentum_col], errors='coerce')
                    ratio = pd.to_numeric(selected_data_industry[ratio_col], errors='coerce')
                    
                    if pd.notna(momentum) and pd.notna(ratio):
                        plot_data_industry.append({
                            'Industry': industry_name,
                            'RS_Momentum': momentum,
                            'RS_Ratio': ratio
                        })
                except Exception:
                    pass
            
            if plot_data_industry:
                plot_df_industry = pd.DataFrame(plot_data_industry)
                
                fig_rrg_industry = go.Figure()
                
                x_min, x_max = 95, 105
                y_min, y_max = 95, 105
                
                # Add quadrant backgrounds
                fig_rrg_industry.add_shape(type="rect", x0=100, y0=100, x1=x_max, y1=y_max,
                                 fillcolor="#d5f4e6", opacity=0.3, layer="below", line_width=0)
                fig_rrg_industry.add_shape(type="rect", x0=x_min, y0=100, x1=100, y1=y_max,
                                 fillcolor="#d6eaf8", opacity=0.3, layer="below", line_width=0)
                fig_rrg_industry.add_shape(type="rect", x0=x_min, y0=y_min, x1=100, y1=100,
                                 fillcolor="#fadbd8", opacity=0.3, layer="below", line_width=0)
                fig_rrg_industry.add_shape(type="rect", x0=100, y0=y_min, x1=x_max, y1=100,
                                 fillcolor="#fdeaa8", opacity=0.3, layer="below", line_width=0)
                
                # Add trails with different colors
                if trail_length_industry > 0 and not trail_df_industry.empty:
                    for industry_name, (momentum_col, ratio_col) in industries.items():
                        trail_x = []
                        trail_y = []
                        
                        for _, row in trail_df_industry.iterrows():
                            momentum = pd.to_numeric(row[momentum_col], errors='coerce')
                            ratio = pd.to_numeric(row[ratio_col], errors='coerce')
                            
                            if pd.notna(momentum) and pd.notna(ratio):
                                trail_x.append(ratio)
                                trail_y.append(momentum)
                        
                        if len(trail_x) > 1:
                            base_color = industry_colors.get(industry_name, '#95a5a6')
                            h = base_color.lstrip('#')
                            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                            
                            num_segments = len(trail_x) - 1
                            for i in range(num_segments):
                                position = (i + 1) / num_segments
                                line_width = 1 + (position * 4)
                                opacity = 0.2 + (position * 0.5)
                                color_with_opacity = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'
                                
                                fig_rrg_industry.add_trace(
                                    go.Scatter(
                                        x=[trail_x[i], trail_x[i+1]],
                                        y=[trail_y[i], trail_y[i+1]],
                                        mode='lines',
                                        line=dict(width=line_width, color=color_with_opacity),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    )
                                )
                
                # Add quadrant lines
                fig_rrg_industry.add_hline(y=100, line_dash="solid", line_color="#7f8c8d", line_width=2, opacity=0.8)
                fig_rrg_industry.add_vline(x=100, line_dash="solid", line_color="#7f8c8d", line_width=2, opacity=0.8)
                
                # Add markers with different colors
                for _, row in plot_df_industry.iterrows():
                    industry_name = row['Industry']
                    color = industry_colors.get(industry_name, '#3498db')
                    
                    fig_rrg_industry.add_trace(
                        go.Scatter(
                            x=[row['RS_Ratio']],
                            y=[row['RS_Momentum']],
                            mode='markers+text',
                            marker=dict(size=12, color=color, line=dict(width=2, color='white')),
                            text=industry_name,
                            textposition='top center',
                            textfont=dict(size=10, color='#2c3e50'),
                            hovertemplate='<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<br><extra></extra>',
                            showlegend=False
                        )
                    )
                
                # Add quadrant labels
                fig_rrg_industry.add_annotation(x=104, y=104, text="Leading", showarrow=False,
                                      font=dict(size=22, color="#27ae60", family="Arial Black"))
                fig_rrg_industry.add_annotation(x=96, y=104, text="Improving", showarrow=False,
                                      font=dict(size=22, color="#2980b9", family="Arial Black"))
                fig_rrg_industry.add_annotation(x=96, y=96, text="Lagging", showarrow=False,
                                      font=dict(size=22, color="#c0392b", family="Arial Black"))
                fig_rrg_industry.add_annotation(x=104, y=96, text="Weakening", showarrow=False,
                                      font=dict(size=22, color="#d68910", family="Arial Black"))
                
                fig_rrg_industry.update_layout(
                    height=700,
                    template="plotly_white",
                    plot_bgcolor='#f8f9fa',
                    xaxis_title="JdK RS-Ratio",
                    yaxis_title="JdK RS-Momentum",
                    xaxis=dict(range=[x_min, x_max], zeroline=False, gridcolor='#ecf0f1'),
                    yaxis=dict(range=[y_min, y_max], zeroline=False, gridcolor='#ecf0f1'),
                    hovermode="closest",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig_rrg_industry, use_container_width=True, config={"displayModeBar": True})
                
    except Exception as e:
        st.error(f"Failed to load RRG Industry data: {e}")

st.markdown("---")

# -------------------------
# Panel 3: Breakdown RRG by Sector
# -------------------------
with st.expander("📊 Breakdown Relative Rotation Graph by Sector", expanded=False):

    # Sector selection and GID mapping
    # sector_gids = {
    #     'Industrials': '1704777',
    #     'Technology': '1208858936',
    #     'Property & Construction': '1941057031',
    #     'Agro & Food': '78853274',
    #     'Services': '1577038098',
    #     'Resources': '1570303424',
    #     'Financials': '2027807125',
    #     'Consumer Products': '886234418'
    # }

    sector_gids = {
        'Steel and Metal Products': '1475307044',
        'Information & Communication Technology': '2140439926',
        'Property Development': '108929325',
        'Food & Beverage': '1173011976',
        'Transportation & Logistics': '86837047',
        
        'Energy & Utilities': '807399401',
        'Automotive': '630755209',
        'Commerce': '670040576',
        'Finance & Securities': '1905330539',
        'Fashion': '900611223',

        'Health Care Services': '84173182',
        'Property Fund & REITs': '482036520',
        'Packaging': '1907932921',
        'Home & Office Products': '1316076253',
        'Industrial Materials & Machinery': '1811057170',

        'Media & Publishing': '801817558',
        'Personal Products & Pharmaceuticals': '910771882',
        'Construction Services': '573104815',
        'Tourism & Leisure': '2131961251',
        'Insurance': '1517004831',

        'Banking': '411551146',
        'Petrochemicals & Chemicals': '1053136329',
        'Professional Services': '77427195',
        'Electronic Components': '418657449',
        'Construction Materials': '1276729230',

        'Agribusiness': '1432592229',
        'Paper & Printing Materials': '587433854',
    }
    
    selected_breakdown_sector = st.selectbox(
        "Select Sector to View Breakdown",
        list(sector_gids.keys()),
        key="breakdown_sector_select"
    )

    breakdown_gid = sector_gids[selected_breakdown_sector]

    try:
        rrg_breakdown_df = load_rrg_df(SHEET_ID, breakdown_gid)

        rrg_breakdown_df['Date'] = pd.to_datetime(rrg_breakdown_df['Date'], errors='coerce')
        rrg_breakdown_df = rrg_breakdown_df.dropna(subset=['Date']).sort_values('Date')

        if rrg_breakdown_df.empty:
            st.warning(f"No valid RRG data available for {selected_breakdown_sector}")
            st.stop()

        # -------------------------
        # Date & trail controls
        # -------------------------
        available_dates = rrg_breakdown_df['Date'].unique()
        num_points = min(20, len(available_dates))
        recent_dates = available_dates[-num_points:]

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            date_index = st.slider(
                f"Select Date (0 = oldest, {num_points - 1} = most recent)",
                0, num_points - 1, num_points - 1,
                key="rrg_breakdown_date_slider"
            )

        with col2:
            selected_date = recent_dates[date_index]
            st.metric("Selected Date", pd.Timestamp(selected_date).strftime('%Y-%m-%d'))

        with col3:
            trail_length = st.slider(
                "Trail Length", 0, 5, 0,
                key="rrg_breakdown_trail_slider"
            )

        selected_row = rrg_breakdown_df[rrg_breakdown_df['Date'] == selected_date].iloc[0]

        if trail_length > 0:
            trail_dates = recent_dates[max(0, date_index - trail_length + 1):date_index + 1]
            trail_df = rrg_breakdown_df[rrg_breakdown_df['Date'].isin(trail_dates)]
        else:
            trail_df = pd.DataFrame()

        # -------------------------
        # Define breakdown items
        # -------------------------
        breakdown_items = {}
        for col in rrg_breakdown_df.columns:
            if 'JdK RS-Momentum' in col:
                name = col.replace(' JdK RS-Momentum', '').strip()
                ratio_col = f'{name} JdK RS-Ratio'
                if ratio_col in rrg_breakdown_df.columns:
                    breakdown_items[name] = (col, ratio_col)

        # -------------------------
        # Generate colors (dynamic)
        # -------------------------
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        cmap = plt.get_cmap('tab10' if len(breakdown_items) <= 10 else 'tab20')
        breakdown_colors = {
            name: mcolors.rgb2hex(cmap(i % cmap.N))
            for i, name in enumerate(sorted(breakdown_items.keys()))
        }

        # -------------------------
        # Prepare plot data
        # -------------------------
        plot_data = []
        for name, (m_col, r_col) in breakdown_items.items():
            m = pd.to_numeric(selected_row[m_col], errors='coerce')
            r = pd.to_numeric(selected_row[r_col], errors='coerce')
            if pd.notna(m) and pd.notna(r):
                plot_data.append({'Item': name, 'RS_Momentum': m, 'RS_Ratio': r})

        if not plot_data:
            st.warning("No valid breakdown points to plot")
            st.stop()

        plot_df = pd.DataFrame(plot_data)

        # -------------------------
        # Build figure
        # -------------------------
        fig = go.Figure()
        x_min, x_max = 95, 105
        y_min, y_max = 95, 105

        # Quadrants
        fig.add_shape(type="rect", x0=100, y0=100, x1=x_max, y1=y_max,
                      fillcolor="#d5f4e6", opacity=0.3, layer="below")
        fig.add_shape(type="rect", x0=x_min, y0=100, x1=100, y1=y_max,
                      fillcolor="#d6eaf8", opacity=0.3, layer="below")
        fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=100, y1=100,
                      fillcolor="#fadbd8", opacity=0.3, layer="below")
        fig.add_shape(type="rect", x0=100, y0=y_min, x1=x_max, y1=100,
                      fillcolor="#fdeaa8", opacity=0.3, layer="below")

       
        # -------------------------
        # Trails
        # -------------------------
        if trail_length > 0 and not trail_df.empty:
            for name, (m_col, r_col) in breakdown_items.items():
                xs, ys = [], []
                for _, row in trail_df.iterrows():
                    m = pd.to_numeric(row[m_col], errors='coerce')
                    r = pd.to_numeric(row[r_col], errors='coerce')
                    if pd.notna(m) and pd.notna(r):
                        xs.append(r)
                        ys.append(m)

                if len(xs) > 1:
                    h = breakdown_colors[name].lstrip('#')
                    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

                    for i in range(len(xs) - 1):
                        pos = (i + 1) / (len(xs) - 1)
                        fig.add_trace(go.Scatter(
                            x=[xs[i], xs[i+1]],
                            y=[ys[i], ys[i+1]],
                            mode='lines',
                            line=dict(
                                width=1 + 4 * pos,
                                color=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {0.2 + 0.5 * pos})'
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        # -------------------------
        # Markers
        # -------------------------
        for _, row in plot_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['RS_Ratio']],
                y=[row['RS_Momentum']],
                mode='markers+text',
                marker=dict(
                    size=14,
                    color=breakdown_colors[row['Item']],
                    line=dict(width=2, color='white')
                ),
                text=row['Item'],
                textposition='top center',
                textfont=dict(size=10),
                hovertemplate="<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>",
                showlegend=False
            ))

        # Center lines
        fig.add_hline(y=100, line_color="#7f8c8d", line_width=2)
        fig.add_vline(x=100, line_color="#7f8c8d", line_width=2)

        # Add these before fig.update_layout()
        fig.add_annotation(x=104, y=104, text="Leading", showarrow=False,
                            font=dict(size=22, color="#27ae60", family="Arial Black"))
        fig.add_annotation(x=96, y=104, text="Improving", showarrow=False,
                            font=dict(size=22, color="#2980b9", family="Arial Black"))
        fig.add_annotation(x=96, y=96, text="Lagging", showarrow=False,
                            font=dict(size=22, color="#c0392b", family="Arial Black"))
        fig.add_annotation(x=104, y=96, text="Weakening", showarrow=False,
                            font=dict(size=22, color="#d68910", family="Arial Black"))

        fig.update_layout(
                    height=700,
                    template="plotly_white",
                    plot_bgcolor='#f8f9fa',
                    paper_bgcolor='white',
                    title=f"{selected_breakdown_sector} – Breakdown RRG",
                    xaxis_title="JdK RS-Ratio",
                    yaxis_title="JdK RS-Momentum",
                    xaxis=dict(
                        range=[x_min, x_max],
                        zeroline=False,
                        gridcolor='#ecf0f1',
                        title_font=dict(color='#2c3e50', size=14),
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        range=[y_min, y_max],
                        zeroline=False,
                        gridcolor='#ecf0f1',
                        title_font=dict(color='#2c3e50', size=14),
                        tickfont=dict(size=12)
                    ),
                    hovermode="closest",
                    showlegend=False,
                    margin=dict(l=60, r=10, t=60, b=60)
                )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load Breakdown RRG data: {e}")


# Footer
st.markdown("---")
st.caption("📊 RRG Dashboard | Created by Nampu")