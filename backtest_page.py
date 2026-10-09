"""Backtest page: ATR Trend Bands + Normalized MACD + Normalized RSI, long only, with a full trade log."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest as B
import data_io

FONT = "IBM Plex Sans"
INK, MUTED, GRID = "#1A2530", "#6B7780", "#E3E8EB"
UP, DOWN, NEUTRAL, ACCENT, TRIG, RSI_C = "#2E8B57", "#B23A48", "#8C969E", "#2C5F8A", "#C08A2E", "#6B5B95"
BUNDLED = Path(__file__).resolve().parent / "data" / "Netflix_Historical_Data.csv"
ATR_MODES = {"Trend must be down (red)": "down", "Trend must be up (green)": "up",
             "Buy when it turns green": "green", "Any trend": "off"}
TIME_UNITS = {"1 day": "trading days", "1 week": "weeks", "1 month": "months"}   # other bar sizes: bars
SOURCES = ["Netflix file you sent", "Upload a CSV", "Yahoo Finance ticker"]
REPEAT = ["Ignore it: one position at a time", "Buy more, then sell everything together"]
FILLS = ["Next bar's open (TradingView default)", "The signal bar's close"]
# Every sidebar setting has a key starting with bt_, so it survives a trip to the screener page (see app.py).
DEFAULTS = {
    "bt_source": SOURCES[0], "bt_ticker": "NFLX", "bt_bar": "1 day",
    "bt_atr_entry": "Trend must be down (red)", "bt_use_near_band": True, "bt_near_band": 0.25, "bt_use_nmacd": True,
    "bt_nmacd_level": -0.8, "bt_use_nrsi": False, "bt_nrsi_lower": -0.8, "bt_window": 3,
    "bt_repeat": REPEAT[0], "bt_entry_pct": 25, "bt_max_entries": 4,
    "bt_exit_nmacd": True, "bt_exit_level": 0.99, "bt_exit_atr_green": False, "bt_exit_atr_red": False,
    "bt_exit_nrsi": False, "bt_nrsi_upper": 0.8, "bt_stop_loss": 0, "bt_max_bars": 0,
    "bt_fill": FILLS[0], "bt_cost": 5.0, "bt_capital": 100_000, "bt_log": True,
    "bt_atr_source": "close", "bt_atr_length": 5, "bt_atr_mult": 7.6, "bt_fast": 13, "bt_slow": 21, "bt_trigger": 9,
    "bt_normalize": 50, "bt_ma_type": 1, "bt_rsi_length": 14, "bt_rsi_normalize": 50,
}
ss = st.session_state
for k, v in DEFAULTS.items():
    if k not in ss:
        ss[k] = v
note = ss.pop("screener_note", None)        # set when a stock is opened from the screener


def spct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def money(x):
    return "–" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), font=dict(family=FONT, color=INK, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None))
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def segments(x, y, state, want):
    xs, ys = [], []
    for i in range(1, len(y)):
        if state[i] == want and np.isfinite(y[i - 1]) and np.isfinite(y[i]):
            xs += [x[i - 1], x[i], None]
            ys += [y[i - 1], y[i], None]
    return xs, ys


@st.cache_data(show_spinner="Reading prices…")
def load_file(content: bytes | None) -> pd.DataFrame:
    return data_io.read_csv(content if content is not None else BUNDLED)


@st.cache_data(ttl=900, show_spinner="Downloading prices from Yahoo Finance…")
def load_yahoo(ticker: str, bar_size: str) -> pd.DataFrame:
    return data_io.from_yahoo(ticker, bar_size)


@st.cache_data(show_spinner="Computing the indicators…")
def indicators_cached(prices: pd.DataFrame, ind: dict) -> pd.DataFrame:
    return B.indicators(prices, B.Settings(**ind))


def base_for(prices: pd.DataFrame, settings: dict) -> pd.DataFrame:
    return indicators_cached(prices, {k: settings[k] for k in B.INDICATOR_FIELDS})


@st.cache_data(show_spinner="Running the backtest…")
def run_cached(prices: pd.DataFrame, settings: dict):
    s = B.Settings(**settings)
    d, eq, tr = B.run(prices, s, base_for(prices, settings))
    return d, eq, tr, B.stats(eq, tr, d, s.capital)


@st.cache_data(show_spinner="Comparing buy signals…")
def compare_cached(prices: pd.DataFrame, settings: dict):
    return B.compare_signals(prices, B.Settings(**settings), base_for(prices, settings))


@st.cache_data(show_spinner="Testing other thresholds…")
def sensitivity_cached(prices: pd.DataFrame, settings: dict, grid: tuple):
    return B.sensitivity(prices, B.Settings(**settings), dict(grid), base_for(prices, settings))


# ----------------------------------------------------------------------------------------
# Sidebar: data and test period
# ----------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Prices", SOURCES, key="bt_source")
    upload, ticker, bar_choice = None, "", "1 day"
    if source == "Upload a CSV":
        upload = st.file_uploader("Daily prices (Nasdaq, Yahoo or TradingView export)", type=["csv"])
    elif source == "Yahoo Finance ticker":
        ticker = st.text_input("Ticker", key="bt_ticker").strip().upper()
        bar_choice = st.selectbox(
            "Bar size", list(data_io.TIMEFRAMES), key="bt_bar",
            help="One candle on the chart. Yahoo keeps 1-minute bars for the last 7 days, 5 to 30-minute bars for "
                 "60 days and 1-hour bars for 2 years. Daily, weekly and monthly bars go back to the listing. "
                 "Yahoo has no 4-hour bars, so they're built from 1-hour bars (09:30 and 13:30 for US stocks, like "
                 "TradingView). Intraday bars cover regular trading hours only.")
    unit = "bars" if source == "Upload a CSV" else TIME_UNITS.get(bar_choice, "bars")

if source == "Upload a CSV" and upload is None:
    st.info("Upload a CSV of daily prices in the sidebar.")
    st.stop()
if source == "Yahoo Finance ticker" and not ticker:
    st.info("Type a ticker in the sidebar.")
    st.stop()
prices = None
try:
    if source == "Netflix file you sent":
        prices, label = load_file(None), "Netflix (NFLX), from the file you sent"
    elif source == "Upload a CSV":
        prices, label = load_file(upload.getvalue()), upload.name
    else:
        prices, label = load_yahoo(ticker, bar_choice), f"{ticker} from Yahoo Finance"
except Exception as e:  # noqa: BLE001
    st.error(f"Couldn't load prices: {e}")
if prices is None:
    st.stop()

intraday = data_io.is_intraday(prices.index)
bars_desc = data_io.bar_size(prices.index)                                   # "daily", "1-hour", ...
bar_word = {"daily": "day", "weekly": "week", "monthly": "month"}.get(bars_desc, "bar")
first_day, last_day = prices.index[0].date(), prices.index[-1].date()
ten_years_ago = max(first_day, (pd.Timestamp(last_day) - pd.DateOffset(years=10)).date())   # the default test start

# ----------------------------------------------------------------------------------------
# Sidebar: rules
# ----------------------------------------------------------------------------------------
with st.sidebar:
    c1, c2 = st.columns(2)
    start = c1.date_input("Test from", ten_years_ago, min_value=first_day, max_value=last_day, format="YYYY-MM-DD")
    end = c2.date_input("Test to", last_day, min_value=first_day, max_value=last_day, format="YYYY-MM-DD")
    if start > end:
        start, end = end, start
        st.caption("'Test from' was after 'Test to', so the two dates were swapped.")
    st.caption("The charts and results cover these dates: the last 10 years unless you change them. The "
               "indicators also use the bars before them, so they're warmed up from the first test day.")

    st.header("Buy rules")
    atr_entry = ATR_MODES[st.radio("ATR Trend Bands", list(ATR_MODES), key="bt_atr_entry",
                                   help="'Buy when it turns green' makes the flip from red to green a buy signal "
                                        "of its own, which you can combine with the others.")]
    use_near_band = st.checkbox("Close near the lower band", key="bt_use_near_band",
                                help="In an uptrend the price usually sits near the upper band, so switch this "
                                     "off for trend-following buys.")
    if use_near_band:
        st.slider("Within this share of the corridor", 0.05, 1.0, step=0.05, key="bt_near_band",
                  help="0 = on the lower band, 1 = on the upper band. 0.25 means the bottom quarter.")
    near_band = ss["bt_near_band"]
    use_nmacd = st.checkbox("NMACD crosses above its trigger", key="bt_use_nmacd")
    if use_nmacd:
        st.slider("Highest NMACD level for the cross", -1.0, 1.0, step=0.05, key="bt_nmacd_level",
                                help="The NMACD value on the bar where it closes above the trigger must be at or "
                                     "below this. -0.8 means the cross happens near -1; 1 accepts any cross.")
    nmacd_level = ss["bt_nmacd_level"]
    use_nrsi = st.checkbox("Normalized RSI crosses up through its lower bound", key="bt_use_nrsi")
    if use_nrsi:
        st.slider("Lower bound", -1.0, 1.0, step=0.05, key="bt_nrsi_lower",
                  help="A signal when the normalized RSI closes above this after closing at or below it.")
    nrsi_lower = ss["bt_nrsi_lower"]
    n_signals = (atr_entry == "green") + use_nmacd + use_nrsi
    if n_signals >= 2:
        st.slider("Signals must fall within this many bars", 1, 20, key="bt_window",
                  help="1 = all on the same bar. 3 = within 3 bars in a row. The buy is on the bar where the last "
                       "of them arrives.")
    window = ss["bt_window"]

    st.header("Position")
    repeat = st.radio("A buy signal while already holding", REPEAT, key="bt_repeat",
                      help="'Buy more' buys again on every new signal. The next sell signal, stop loss or time "
                           "limit sells all the buys at once.")
    pyramid = repeat.startswith("Buy more")
    if pyramid:
        st.slider("Each buy uses this % of the account", 5, 100, step=5, key="bt_entry_pct",
                  help="Of the account value at the fill (cash plus the shares already held), and never more than "
                       "the cash left, so the test never borrows.")
        st.slider("Most buys held at once", 2, 20, key="bt_max_entries")
    entry_size, max_entries = ss["bt_entry_pct"] / 100, ss["bt_max_entries"]

    st.header("Sell rules")
    exit_nmacd = st.checkbox("NMACD hits +1", key="bt_exit_nmacd")
    if exit_nmacd:
        st.slider("NMACD 'hits +1': at or above", 0.80, 1.0, step=0.01, key="bt_exit_level",
                  help="The line can't print exactly +1 because of a tiny constant in its formula.")
    exit_level = ss["bt_exit_level"]
    exit_atr_green = st.checkbox("ATR Trend Bands turn green (flip up)", key="bt_exit_atr_green")
    exit_atr_red = st.checkbox("ATR Trend Bands turn red (flip down)", key="bt_exit_atr_red")
    exit_nrsi = st.checkbox("Normalized RSI falls back below its upper bound", key="bt_exit_nrsi")
    if exit_nrsi:
        st.slider("Upper bound", -1.0, 1.0, step=0.05, key="bt_nrsi_upper",
                  help="A sell when the normalized RSI closes below this after closing at or above it.")
    nrsi_upper = ss["bt_nrsi_upper"]
    stop_loss = st.slider("Optional stop loss (% below entry, 0 = off)", 0, 50, step=1, key="bt_stop_loss",
                          help="With 'Buy more' it's measured from the average price of the buys and sells all "
                               "of them.")
    max_bars = st.slider(f"Optional time limit ({unit}, 0 = off)", 0, 250, step=5, key="bt_max_bars",
                         help="With 'Buy more' it counts from the first buy.")
    if not (exit_nmacd or exit_atr_green or exit_atr_red or exit_nrsi or stop_loss or max_bars):
        st.caption("No sell rule is on, so every position is held to the end of the test.")

    st.header("Execution")
    fill_label = st.radio("Orders fill at", FILLS, key="bt_fill", help="A bar is one candle: one day on daily prices.")
    cost_bps = st.number_input("Commission and slippage per side (basis points)", min_value=0.0, max_value=100.0,
                               step=0.5, key="bt_cost")
    capital = st.number_input("Starting capital ($)", min_value=1_000, max_value=100_000_000, step=1_000,
                              key="bt_capital")
    log_scale = st.toggle("Log scale on the equity chart", key="bt_log")

    with st.expander("Indicator settings"):
        st.markdown("**ATR Trend Bands**")
        atr_source = st.selectbox("Source", ("close", "open", "high", "low", "hl2", "hlc3", "ohlc4"), key="bt_atr_source")
        atr_length = st.number_input("Length ATR", min_value=1, max_value=100, key="bt_atr_length")
        atr_mult = st.number_input("Multiplier ATR", min_value=0.1, max_value=50.0, step=0.1, format="%.1f",
                                   key="bt_atr_mult")
        st.markdown("**Normalized MACD**")
        fast = st.number_input("Fast MA", min_value=1, max_value=200, key="bt_fast")
        slow = st.number_input("Slow MA", min_value=2, max_value=400, key="bt_slow")
        trigger = st.number_input("Trigger", min_value=1, max_value=100, key="bt_trigger")
        normalize = st.number_input("Normalize", min_value=1, max_value=500, key="bt_normalize")
        ma_type = st.number_input("1=Ema, 2=Wma, 3=Sma", min_value=1, max_value=3, key="bt_ma_type")
        st.markdown("**Normalized RSI**")
        rsi_length = st.number_input("RSI length", min_value=1, max_value=200, key="bt_rsi_length")
        rsi_normalize = st.number_input("RSI normalize (bars, 1 = off)", min_value=1, max_value=500, key="bt_rsi_normalize",
                                        help="1 = plain RSI rescaled to -1..+1, so RSI 30 = -0.4 and RSI 70 = +0.4.")

# ----------------------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------------------
def when(ts) -> str:
    return f"{pd.Timestamp(ts):%b %d, %Y %H:%M}" if intraday else f"{pd.Timestamp(ts):%b %d, %Y}"


def X(ts):
    """x values for the charts. Intraday bars go on a category axis, so nights and weekends take no space."""
    idx = pd.DatetimeIndex(ts)
    return list(idx.strftime("%Y-%m-%d %H:%M")) if intraday else idx


in_test = (prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(end) + pd.Timedelta(days=1))
if in_test.sum() < 2:
    st.error("The test period has fewer than 2 bars. Pick dates further apart.")
    st.stop()

settings = B.Settings(atr_length=int(atr_length), atr_mult=float(atr_mult), atr_source=atr_source, fast=int(fast),
                      slow=int(slow), trigger=int(trigger), normalize=int(normalize), ma_type=int(ma_type),
                      rsi_length=int(rsi_length), rsi_normalize=int(rsi_normalize),
                      atr_entry=atr_entry, use_near_band=use_near_band, near_band=near_band, use_nmacd=use_nmacd,
                      nmacd_level=nmacd_level, use_nrsi=use_nrsi, nrsi_lower=nrsi_lower, window=int(window),
                      exit_nmacd=exit_nmacd, exit_level=exit_level, exit_atr_green=exit_atr_green,
                      exit_atr_red=exit_atr_red, exit_nrsi=exit_nrsi, nrsi_upper=nrsi_upper,
                      stop_loss=stop_loss / 100, max_bars=int(max_bars), pyramid=pyramid, entry_size=entry_size,
                      max_entries=int(max_entries), fill="open" if fill_label.startswith("Next") else "close",
                      cost_bps=float(cost_bps), capital=float(capital),
                      start=None if start <= first_day else start.isoformat(),
                      end=None if end >= last_day else end.isoformat())
d, eq, trades, stats = run_cached(prices, asdict(settings))
funnel = B.funnel(d, settings, f"{bar_word}s")

st.title("ATR Trend Bands + NMACD + Normalized RSI backtest")
mode = (f"each buy signal buys {entry_size:.0%} of the account, up to {max_entries} buys at once, and a sell "
        "signal sells them all" if pyramid else "one position at a time, all capital in each trade")
period = ("" if settings.start is None and settings.end is None else
          f" Testing {when(d.index[0])} to {when(d.index[-1])}: {len(d):,} bars.")
st.caption(f"{label}: {len(prices):,} {bars_desc} bars, {when(prices.index[0])} to {when(prices.index[-1])}."
           f"{period} Long only, {mode}.")
if note:
    st.info(note)

buys = {"down": ["The ATR Trend Bands trend is **down** (red)."],
        "up": ["The ATR Trend Bands trend is **up** (green)."],
        "green": ["The ATR Trend Bands **turn green** (the trend flips up)."], "off": []}[atr_entry]
if use_near_band:
    buys.append(f"The close is **near the lower band**: in the bottom {near_band:.0%} of the corridor.")
if use_nmacd:
    buys.append(f"The NMACD line **crosses above its trigger line** at a level at or below {nmacd_level:g}.")
if use_nrsi:
    buys.append(f"The normalized RSI **crosses up through {nrsi_lower:g}**.")
if n_signals >= 2:
    buys.append("All on the same bar." if window == 1 else
                f"The signals can come on different bars, all within {window} bars. The buy is on the bar where "
                "the last one arrives.")
if not buys:
    buys.append("Every bar: no buy rule is switched on.")
if pyramid:
    buys.append(f"While holding, **every new signal buys more**: {entry_size:.0%} of the account each time, up to "
                f"{max_entries} buys.")
same_bar = n_signals < 2 or window == 1
sells = []
if exit_nmacd:
    sells.append(f"the NMACD line **hits +1** (closes at or above {exit_level:g})")
if exit_atr_green:
    sells.append("the ATR Trend Bands **turn green** (the trend flips up)")
if exit_atr_red:
    sells.append("the ATR Trend Bands **turn red** (the trend flips down)")
if exit_nrsi:
    sells.append(f"the normalized RSI **falls back below {nrsi_upper:g}**")
if stop_loss:
    sells.append(f"the price falls {stop_loss}% below the {'average ' if pyramid else ''}entry price (stop loss)")
if max_bars:
    sells.append(f"{max_bars} {unit} have passed{' since the first buy' if pyramid else ''}")
sell_lines = [f"- {x[0].upper() + x[1:]}." if i == 0 else f"- Or {x}." for i, x in enumerate(sells)]

b, s_ = st.columns(2)
with b.container(border=True):
    buy_head = f"Buy when, all on the same {bar_word}'s close" if same_bar else "Buy when"
    st.markdown(f'<div class="rule-label">{buy_head}</div>', unsafe_allow_html=True)
    st.markdown("\n".join(f"- {x}" for x in buys))
with s_.container(border=True):
    st.markdown(f'<div class="rule-label">{"Sell everything when" if pyramid else "Sell when"}</div>',
                unsafe_allow_html=True)
    st.markdown("\n".join(sell_lines) if sell_lines else "- Never: positions are held to the end of the test.")
fill_text = f"the next {bar_word}'s open" if settings.fill == "open" else f"the signal {bar_word}'s close"
st.caption(f"Orders fill at {fill_text}, with {cost_bps:g} basis points of cost on each side.")

last = d.iloc[-1]
open_lots = trades[trades["Status"] == "Open"] if len(trades) else trades
notes = []
if len(open_lots) == 1:
    t = open_lots.iloc[0]
    notes.append(f"At the last close ({when(d.index[-1])}) the backtest is holding a position bought on "
                 f"{when(t['Entry date'])} at {t['Entry price']:.2f}, currently {spct(t['Return'])}.")
elif len(open_lots) > 1:
    inv = open_lots["Invested ($)"]
    avg_px = inv.sum() / (inv / open_lots["Entry price"]).sum()
    notes.append(f"At the last close ({when(d.index[-1])}) the backtest is holding {len(open_lots)} buys made "
                 f"from {when(open_lots['Entry date'].min())} to {when(open_lots['Entry date'].max())}, at an "
                 f"average price of {avg_px:.2f}, currently {spct(open_lots['P&L ($)'].sum() / inv.sum())}.")
if (bool(last["Entry signal"]) and settings.fill == "open" and d.index[-1] == prices.index[-1]
        and len(open_lots) < (max_entries if pyramid else 1)):
    notes.append(f"The rules fired a buy signal on the last close ({when(d.index[-1])}). In the backtest it would "
                 f"fill at the next {'session' if bar_word == 'day' else bar_word}'s open, which isn't in the data "
                 "yet. This is the rules' output, not advice.")
if notes:
    st.info(" ".join(notes))

# ----------------------------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------------------------
c = st.columns(6)
c[0].metric("Total P&L", money(stats["Total P&L ($)"]), spct(stats["Total return"]), delta_color="normal")
c[1].metric("Win rate", "–" if pd.isna(stats["Win rate"]) else f"{stats['Win rate']:.0%}",
            f"{int(round(stats['Win rate'] * stats['Closed trades'])) if stats['Closed trades'] else 0} of "
            f"{stats['Closed trades']} closed trades", delta_color="off")
c[2].metric("Average win / loss", f"{spct(stats['Average win'])} / {spct(stats['Average loss'])}")
c[3].metric("Profit factor", "–" if pd.isna(stats["Profit factor"]) else f"{stats['Profit factor']:.2f}")
c[4].metric("Max drawdown", spct(stats["Max drawdown"]))
c[5].metric("Buy & hold instead", spct(stats["Buy & hold return"]),
            f"max drawdown {stats['Buy & hold max drawdown'] * 100:.0f}%", delta_color="off")
c = st.columns(6)
c[0].metric("Trades", f"{stats['Trades']}", f"in {stats['Positions']} positions" if pyramid else None,
            delta_color="off")
c[1].metric("Average trade", spct(stats["Average trade"]))
c[2].metric("Best / worst trade", f"{spct(stats['Best trade'])} / {spct(stats['Worst trade'])}")
c[3].metric(f"Average {bar_word}s held",
            "–" if pd.isna(stats["Average bars held"]) else f"{stats['Average bars held']:.0f}")
c[4].metric("Time in the market", f"{stats['Time in market']:.0%}")
c[5].metric("Worst dip inside a trade", spct(stats["Worst drawdown inside a trade"]),
            help="Lowest low while holding, against the entry price.")
n_closed = stats["Closed positions"] if pyramid else stats["Closed trades"]
if n_closed < 30:
    noun = f"closed {'position' if pyramid else 'trade'}{'' if n_closed == 1 else 's'}"
    what = f"{noun} ({stats['Closed trades']} buys)" if pyramid else noun
    st.warning(f"Only {n_closed} {what}. With so few, the win rate and P&L can swing a lot from one trade, so "
               "treat them as anecdotes rather than evidence.")

# ----------------------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------------------
st.markdown("##### Equity")
bh = d["Close"] / d["Close"].iloc[0] * settings.capital
fig = go.Figure()
fig.add_trace(go.Scatter(x=X(bh.index), y=bh, name="Buy & hold", line=dict(color=NEUTRAL, width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=X(eq.index), y=eq, name="Strategy", line=dict(color=ACCENT, width=2.2)))
fig = style_fig(fig, 360)
fig.update_yaxes(type="log" if log_scale else "linear", tickprefix="$", tickformat=",.0f")
if intraday:
    fig.update_xaxes(type="category")
st.plotly_chart(fig, width="stretch")

st.markdown("##### Price with ATR Trend Bands, the NMACD and the normalized RSI")
x = list(X(d.index))
state = d["Trend"].map({"Up": 1, "Down": -1}).fillna(0).to_numpy()
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.56, 0.22, 0.22], vertical_spacing=0.03)
fig.add_trace(go.Scatter(x=x, y=d["Close"], name="Close", line=dict(color=INK, width=1.1)), row=1, col=1)
for band in ("Upper", "Lower"):
    y = d[band].to_numpy()
    for want, color in ((1, UP), (-1, DOWN)):
        xs, ys = segments(x, y, state, want)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.6),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Lower"], mode="lines", line=dict(width=0), showlegend=False,
                         hovertemplate="Lower band %{y:,.2f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Upper"], mode="lines", line=dict(width=0), showlegend=False,
                         hovertemplate="Upper band %{y:,.2f}<extra></extra>"), row=1, col=1)
sig = d[d["Entry signal"]]
fig.add_trace(go.Scatter(x=X(sig.index), y=sig["Close"], mode="markers", name="Buy signal",
                         marker=dict(symbol="circle-open", size=9, color=UP, line=dict(width=1.5))), row=1, col=1)
if len(trades):
    fig.add_trace(go.Scatter(x=X(trades["Entry date"]), y=trades["Entry price"], mode="markers", name="Buy",
                             marker=dict(symbol="triangle-up", size=12, color=UP)), row=1, col=1)
    closed = trades[trades["Status"] == "Closed"].drop_duplicates("Exit date")   # buys sold together: one marker
    fig.add_trace(go.Scatter(x=X(closed["Exit date"]), y=closed["Exit price"], mode="markers", name="Sell",
                             marker=dict(symbol="triangle-down", size=12, color=DOWN)), row=1, col=1)
fig.add_trace(go.Bar(x=x, y=d["Hist"], name="Histogram", marker_color="#B8C0C6", opacity=0.6), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=d["NMACD"], name="NMACD", line=dict(color=DOWN, width=1.6)), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=d["Trigger"], name="Trigger", line=dict(color=TRIG, width=1.4)), row=2, col=1)
nm_sig = d[d["NMACD signal"]]
fig.add_trace(go.Scatter(x=X(nm_sig.index), y=nm_sig["NMACD"], mode="markers", name="NMACD signal",
                         marker=dict(symbol="circle", size=7, color=UP)), row=2, col=1)
for yv, dash in ((1, "dot"), (-1, "dot"), (0, "solid")):
    fig.add_hline(y=yv, line=dict(color=MUTED, width=0.8, dash=dash), row=2, col=1)
fig.add_hline(y=nmacd_level, line=dict(color=UP, width=0.8, dash="dash"), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=d["NRSI"], name="Normalized RSI", line=dict(color=RSI_C, width=1.4)), row=3, col=1)
rs_sig = d[d["NRSI signal"]]
fig.add_trace(go.Scatter(x=X(rs_sig.index), y=rs_sig["NRSI"], mode="markers", name="NRSI signal",
                         marker=dict(symbol="circle", size=7, color=UP)), row=3, col=1)
for yv, color in ((nrsi_lower, UP), (nrsi_upper, DOWN)):
    fig.add_hline(y=yv, line=dict(color=color, width=0.8, dash="dash"), row=3, col=1)
fig = style_fig(fig, 780)
if intraday:
    fig.update_xaxes(type="category")
elif bars_desc == "daily" and not (prices.index.dayofweek >= 5).any():
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
fig.update_yaxes(range=[-1.15, 1.15], row=2, col=1)
fig.update_yaxes(range=[-1.15, 1.15], row=3, col=1)
st.plotly_chart(fig, width="stretch")
st.caption(f"Bands are green in an uptrend and red in a downtrend. Hollow circles on the price are the {bar_word}s "
           "where all the buy rules held. Dots on the NMACD and RSI panels are each indicator's own signals, "
           "shown even when it's switched off; the dashed lines are the levels set in the sidebar.")

# ----------------------------------------------------------------------------------------
# Trades and diagnostics
# ----------------------------------------------------------------------------------------
st.markdown("##### Trade log")
if len(trades):
    show = trades.iloc[::-1].copy()
    for col in ("Signal date", "Entry date", "Exit date"):
        show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "Entry price": st.column_config.NumberColumn(format="%.2f"),
        "Invested ($)": st.column_config.NumberColumn(format="dollar"),
        "Exit price": st.column_config.NumberColumn(format="%.2f"),
        "Return": st.column_config.NumberColumn(format="percent"),
        "P&L ($)": st.column_config.NumberColumn(format="dollar"),
        "Worst drawdown in trade": st.column_config.NumberColumn(format="percent"),
        "Band position at signal": st.column_config.NumberColumn(format="%.2f"),
        "NMACD at signal": st.column_config.NumberColumn(format="%.2f"),
        "NRSI at signal": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download the trade log (CSV)", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
else:
    st.write("No trades with these settings.")

st.markdown("##### Why there are so few trades")
st.dataframe(funnel, hide_index=True, width="stretch")
skipped = (f"With 'Buy more', a signal is skipped only once {max_entries} buys are held or the cash has run out, "
           f"so trades can still be fewer than signal {bar_word}s." if pyramid else
           f"Signal {bar_word}s that come while a position is already open are skipped, so trades can be fewer "
           f"than signal {bar_word}s.")
st.caption(f"Each row keeps only the {bar_word}s that also pass the rows above it. {skipped}")

pct = st.column_config.NumberColumn(format="percent")
st.markdown("##### NMACD, normalized RSI or both")
st.dataframe(compare_cached(prices, asdict(settings)), hide_index=True, width="stretch", column_config={
    "Win rate": pct, "Average trade": pct, "Total return": pct, "Max drawdown": pct})
atr_now = {"down": "trend down", "up": "trend up", "green": "turns green", "off": "not used"}[atr_entry]
st.caption(f"Everything else as set in the sidebar (ATR Trend Bands: {atr_now}; near the lower band: "
           f"{f'within {near_band:.0%}' if use_near_band else 'off'}), switching only the two oscillator buy "
           f"signals. 'Both' needs the two within {window} bars of each other. 'Neither' buys on the ATR rules alone.")

st.markdown("##### How the result changes with the thresholds")
rsi_grid = (-0.6, -0.5, -0.4, -0.3) if int(rsi_normalize) < 2 else (-0.9, -0.8, -0.6, -0.4)
options = {lab: (field, tuple(sorted(set(vals) | {round(getattr(settings, field), 4)})))
           for lab, field, vals, on in [
               ("Near the lower band (within)", "near_band", (0.10, 0.25, 0.50, 1.00), use_near_band),
               ("NMACD cross level (at or below)", "nmacd_level", (-0.9, -0.8, -0.6, -0.4), use_nmacd),
               ("Normalized RSI lower bound", "nrsi_lower", rsi_grid, use_nrsi),
               ("Signals within (bars)", "window", (1, 3, 5, 10), n_signals >= 2),
               ("NMACD exit level", "exit_level", (0.90, 0.95, 0.99), exit_nmacd),
               ("Normalized RSI upper bound", "nrsi_upper", (0.4, 0.6, 0.8, 0.9), exit_nrsi)] if on}
if not options:
    st.caption("Switch on a rule with a threshold (near the lower band, NMACD or normalized RSI) to see this table.")
else:
    c1, c2 = st.columns(2)
    vary = c1.selectbox("Vary", list(options))
    others = ["Nothing"] + [k for k in options if k != vary]
    against = c2.selectbox("Against", others, index=1 if len(others) > 1 else 0)
    picked = [vary] + ([against] if against != "Nothing" else [])
    sens = sensitivity_cached(prices, asdict(settings), tuple(options[k] for k in picked))
    sens = sens.rename(columns={options[k][0]: k for k in picked})
    st.dataframe(sens, hide_index=True, width="stretch", column_config={
        "Win rate": pct, "Average trade": pct, "Total return": pct, "Max drawdown": pct})
    st.caption("Reruns the backtest across several values of the settings picked above. Everything else stays as "
               "set in the sidebar. A result that only works at one exact value is usually luck.")

with st.expander("How the indicators and rules are defined"):
    terms = []
    if use_near_band:
        terms.append(f'- "Close to the lower band": the close sits in the bottom {near_band:.0%} of the corridor '
                     "between the bands.")
    if use_nmacd:
        terms.append(f'- "NMACD crosses above its trigger": the NMACD line closes above the trigger after closing '
                     f"at or below it the bar before, and its value on that bar is at or below {nmacd_level:g}.")
    if use_nrsi:
        terms.append(f'- "Normalized RSI crosses up": it closes above {nrsi_lower:g} after closing at or below it '
                     "the bar before.")
    if atr_entry == "green" or exit_atr_green:
        terms.append('- "Turn green": the ATR Trend Bands trend flips from down to up, the bar where the indicator '
                     "prints an L.")
    if exit_atr_red:
        terms.append('- "Turn red": the trend flips from up to down, the bar where the indicator prints an S.')
    if exit_nmacd:
        terms.append(f'- "Hits +1": the NMACD line closes at or above {exit_level:g}.')
    if exit_nrsi:
        terms.append(f'- "Falls back below its upper bound": the normalized RSI closes below {nrsi_upper:g} after '
                     "closing at or above it.")
    rsi_scale = (f"rescaled to between -1 and +1 against its own range over the last {int(rsi_normalize)} bars, the "
                 "same way as the NMACD. -1 means the RSI is at its lowest of that stretch" if int(rsi_normalize) >= 2
                 else "rescaled as (RSI - 50) / 50, so RSI 30 = -0.4 and RSI 70 = +0.4")
    buying_more = (f"""
**Buying more.** Each buy signal while holding buys another {entry_size:.0%} of the account value (cash plus
the shares already held, at the fill price), never more than the cash left, up to {max_entries} buys. Any exit
sells all of them at once. The stop loss is measured from the average price of the buys and the time limit
counts from the first buy. Each buy is a row in the trade log, so the win rate and average trade are per buy.
""" if pyramid else "")
    st.markdown(
        f"""
**Normalized MACD** (TradingView's "N MACD"). Fast and slow moving averages of the close
({fast} and {slow} bars; type {ma_type}: 1 = EMA, 2 = WMA, 3 = SMA). Their ratio is turned into a
momentum value, which is rescaled to between -1 and +1 against its own range over the last {normalize} bars.
The trigger line is a {trigger}-bar weighted average of the NMACD line.

**What +1 and -1 mean.** +1 means the fast average is further above the slow one than at any point in the
last {normalize} bars, and -1 means the reverse. Because the scale is relative to the last {normalize} bars,
the line can reach +1 after a modest bounce in a falling market. So "NMACD hit +1" doesn't mean the price is
above where you bought, which is why some exits here are losses.

**Normalized RSI.** Wilder's {int(rsi_length)}-bar RSI of the close (the same as TradingView's ta.rsi),
{rsi_scale}.

**ATR Trend Bands** use the same replica as the ATR Trend Bands app, with source {atr_source},
length {atr_length} and multiplier {atr_mult:g}. They're green while the trend is up and red while it's down.

**Turning the words into numbers.**
{chr(10).join(terms) if terms else "- No buy or sell rule with a threshold is switched on."}
{buying_more}
**Test period.** The last 10 years by default; change it with "Test from" and "Test to" in the sidebar. The
indicators are computed on all the prices, so they're already warmed up on the first test day. Trades, the
equity curve, buy & hold and every statistic cover only the test dates.

**What the test leaves out:** taxes, borrowing costs, dividends (Netflix pays none) and gaps through a stop,
beyond the fill rules above.
        """
    )
