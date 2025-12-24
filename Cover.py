# Cover.py
import streamlit as st

# =====================================================
# Authentication Configuration
# =====================================================
USERS = {
    "tradaholic": "marketbreadth"
}

def check_password():
    """Returns True if the user has correct password."""
    
    def password_entered():
        """Checks whether password entered is correct."""
        if st.session_state["username"] in USERS and \
           st.session_state["password"] == USERS[st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.title("🔐 Thailand Market Analysis Dashboard")
    st.markdown("### Please log in to continue")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.text_input("Username", key="username", placeholder="Enter username")
        st.text_input("Password", type="password", key="password", placeholder="Enter password")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("Login", on_click=password_entered, use_container_width=True, type="primary")
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Username or password incorrect")
    
    return False

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Thailand Market Analysis",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# Check Authentication
# =====================================================
if not check_password():
    st.stop()

# =====================================================
# Logout Button (Top Right)
# =====================================================
col_title, col_logout = st.columns([5, 1])

with col_title:
    st.title("Thailand Market Analysis Dashboard")

with col_logout:
    st.write("")  # Spacer
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["password_correct"] = False
        st.rerun()

st.markdown("---")

st.subheader("Available Dashboards (For Educational Purposes Only)")

# =====================================================
# ROW 1
# =====================================================
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("### 📊 SET Index Analysis")
    st.write("Stock Exchange of Thailand (SET)")
    st.write("**Features:**")
    st.write("- SET Index Candlestick Chart")
    st.write("- Market Breadth: Simple Moving Averages")
    st.write("- Market Breadth: New Highs & New Lows")
    st.write("- Market Breadth: Modified New Highs & Lows")
    st.write("- Market Breadth: Double Simple Moving Averages")
    st.write("- Market Breadth: Different New Highs & New Lows")
    
    st.page_link(
        "pages/1_SET_Index.py",
        label="Go to SET Index Dashboard",
        icon="📊"
    )

with row1_col2:
    st.markdown("### 📈 MAI Index Analysis")
    st.write("Market for Alternative Investment (MAI)")
    st.write("**Features:**")
    st.write("- MAI Index Candlestick Chart")
    st.write("- Market Breadth: Simple Moving Averages")
    st.write("- Market Breadth: New Highs & New Lows")
    st.write("- Market Breadth: Modified New Highs & Lows")
    st.write("- Market Breadth: Double Simple Moving Averages")
    st.write("- Market Breadth: Different New Highs & New Lows")

    st.page_link(
        "pages/2_MAI_Index.py",
        label="Go to MAI Index Dashboard",
        icon="📈"
    )

st.markdown("---")

# =====================================================
# ROW 2
# =====================================================
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown("### 📈 SET50 Index Analysis")
    st.write("Stock Exchange of Thailand (SET50)")
    st.write("**Features:**")
    st.write("- SET50 Index Candlestick Chart")
    st.write("- Market Breadth: Simple Moving Averages")
    st.write("- Market Breadth: New Highs & Lows")
    st.write("- Market Breadth: Double Simple Moving Averages")
    
    st.page_link(
        "pages/4_SET50_Index.py",
        label="Go to SET50 Index Dashboard",
        icon="📈"
    )

with row2_col2:
    st.markdown("### 📈 SET100 Index Analysis")
    st.write("Stock Exchange of Thailand (SET100)")
    st.write("**Features:**")
    st.write("- SET100 Index Candlestick Chart")
    st.write("- Market Breadth: Simple Moving Averages")
    st.write("- Market Breadth: New Highs & Lows")
    st.write("- Market Breadth: Double Simple Moving Averages")
    
    st.page_link(
        "pages/5_SET100_Index.py",
        label="Go to SET100 Index Dashboard",
        icon="📈"
    )

st.markdown("---")

# =====================================================
# ROW 3 (RRG + Watchlist)
# =====================================================
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.markdown("### 🔄 Relative Rotation Graph")
    st.write("Relative Strength Analysis for Thailand Market")
    st.write("**Features:**")
    st.write("- RRG by Sector")
    st.write("- RRG by Industry")
    st.write("- Sector Breakdown")

    st.page_link(
        "pages/3_RRG.py",
        label="Go to RRG Dashboard",
        icon="🔄"
    )

with row3_col2:
    st.markdown("### ⭐ Watchlist")
    st.write("Custom Watchlist & Monitoring")
    st.write("**Features:**")
    st.write("- Personal stock watchlist")
    st.write("- Quick market monitoring")
    st.write("- Flexible symbol tracking")

    st.page_link(
        "pages/6_Watch_list.py",
        label="Go to Watchlist",
        icon="⭐"
    )

st.markdown("---")
st.caption("📊 Thailand Market Analysis Dashboard | Created by Nampu")