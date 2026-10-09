"""
Two pages: the backtest of one stock, and a screener of US stocks for recent NMACD signals.

Run locally:   streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="ATR Bands + NMACD backtest", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, p, li, label, input, h1, h2, h3, h4, h5, .stMarkdown,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 500; }
    .rule-label { color: #6B7780; font-size: 0.9rem; margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit forgets the settings of a page you leave. Saving them again on every run keeps the backtest rules
# (keys starting with bt_) and the screener filters (sc_) when you switch pages.
for key in list(st.session_state.keys()):
    if key.startswith(("bt_", "sc_")):
        st.session_state[key] = st.session_state[key]

st.navigation([st.Page("backtest_page.py", title="Backtest", icon="📈", default=True),
               st.Page("screener_page.py", title="NMACD screener", icon="🔎", url_path="screener")]).run()
