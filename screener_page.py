"""NMACD screener: every liquid US stock with a Normalized MACD buy signal in the last few trading days."""
from __future__ import annotations

import threading

import pandas as pd
import streamlit as st

import scanner

ss = st.session_state
ATR_CHOICES = ["Any", "Down (red)", "Up (green)"]
for k, v in {"sc_level": ss.get("bt_nmacd_level", -0.8), "sc_days": 3, "sc_atr": "Any",
             "sc_min_volume": 1_000_000, "sc_min_price": 5.0}.items():
    if k not in ss:
        ss[k] = v
# The NMACD and ATR settings come from the Backtest page, so both pages agree on what a signal is.
ind = {k: ss.get(f"bt_{k}", v) for k, v in {"fast": 13, "slow": 21, "trigger": 9, "normalize": 50, "ma_type": 1}.items()}
atr = {k: ss.get(f"bt_atr_{k}", v) for k, v in {"source": "close", "length": 5, "mult": 7.6}.items()}


def today() -> str:
    return str(pd.Timestamp.now(tz="America/New_York").date())


@st.cache_resource
def store() -> dict:
    """The prices of every stock, downloaded once a day and shared by everyone using the app on this server."""
    return {"day": None, "data": {}, "universe": None, "failed": [], "when": None, "version": 0,
            "lock": threading.Lock()}


@st.cache_data(max_entries=4, show_spinner="Working out the NMACD of every liquid stock…")
def tails_cached(version: int, min_volume: float, min_price: float, ind: tuple):
    return scanner.nmacd_tails(store()["data"], min_volume, min_price, **dict(ind))


@st.cache_data(max_entries=5000)
def atr_cached(version: int, ticker: str, when: pd.Timestamp, atr: tuple):
    return scanner.atr_at(store()["data"][ticker], when, **dict(atr))


with st.sidebar:
    st.header("Signal")
    level = st.slider("Highest NMACD level for the cross", -1.0, 1.0, step=0.05, key="sc_level",
                      help="Same as on the Backtest page: the NMACD value on the bar where it closes above its trigger "
                           "must be at or below this.")
    days = st.slider("Within the last this many trading days", 1, 10, key="sc_days",
                     help="1 = only the latest bar, 3 = the latest bar and the two before it.")
    atr_pick = st.selectbox("ATR Trend Bands on the signal day", ATR_CHOICES, key="sc_atr")
    st.header("Liquidity")
    min_volume = st.number_input("Average volume at least (shares a day, last 20 days)", min_value=0,
                                 max_value=1_000_000_000, step=100_000, key="sc_min_volume")
    min_price = st.number_input("Price at least ($)", min_value=0.0, max_value=100_000.0, step=1.0, key="sc_min_price")
    st.caption(f"NMACD {ind['fast']}/{ind['slow']}, trigger {ind['trigger']}, normalize {ind['normalize']}, and the "
               "ATR Trend Bands, as set on the Backtest page.")

st.title("NMACD screener")
st.caption("Every US-listed stock (Nasdaq, NYSE, NYSE American, NYSE Arca, Cboe and IEX, without ETFs, warrants, "
           "units or preferred shares), with a year of daily prices from Yahoo Finance.")

box = store()
ready = box["day"] == today() and bool(box["data"])
if ready:
    c1, c2 = st.columns([4, 1])
    c1.caption(f"Prices downloaded at {box['when']:%H:%M} New York time today: {len(box['data']):,} stocks"
               + (f" ({len(box['failed']):,} more couldn't be downloaded)" if box["failed"] else "")
               + ". During market hours the latest bar is today's unfinished one, so a signal on it can still "
               "disappear by the close.")
    go = c2.button("Download again")
else:
    st.info("The first scan of the day downloads a year of daily prices for about 5,000 stocks from Yahoo Finance. "
            "It takes a few minutes; scans later today reuse the download.")
    go = st.button("Download prices and scan", type="primary")

if go:
    with box["lock"]:
        try:
            universe = scanner.load_universe()
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't get the list of US stocks: {e}")
            st.stop()
        bar = st.progress(0.0, text="Starting the download…")
        data, failed = scanner.download_all(list(universe["Ticker"]),
                                            progress=lambda f, text: bar.progress(min(f, 1.0), text=text))
        bar.empty()
        if not data:
            st.error("Yahoo Finance returned no prices. It may be limiting requests from this server; try again in "
                     "a few minutes.")
            st.stop()
        box.update(day=today(), data=data, universe=universe, failed=failed, version=box["version"] + 1,
                   when=pd.Timestamp.now(tz="America/New_York"))
    st.rerun()
if not ready:
    st.stop()

tails, liquid, latest = tails_cached(box["version"], float(min_volume), float(min_price), tuple(sorted(ind.items())))
hits = scanner.find_signals(tails, level, int(days))
rows = []
for _, h in hits.iterrows():
    trend, position = atr_cached(box["version"], h["Ticker"], h["Date"], tuple(sorted(atr.items())))
    info = liquid.loc[h["Ticker"]]
    rows.append({"Ticker": h["Ticker"], "Signal date": h["Date"], "Days ago": int(h["Bars ago"]),
                 "NMACD at cross": h["NMACD"], "Since signal": info["Last close"] / h["Close"] - 1,
                 "Last close": info["Last close"], "ATR trend at signal": trend, "Band position at signal": position,
                 "Avg volume (M shares)": info["Avg volume"] / 1e6,
                 "Avg dollar volume ($M)": info["Avg dollar volume"] / 1e6})
table = pd.DataFrame(rows)
if len(table):
    table = table.merge(box["universe"][["Ticker", "Name", "Exchange"]], on="Ticker", how="left")
    table = table[["Ticker", "Name", "Exchange"] + [c for c in table.columns if c not in ("Ticker", "Name", "Exchange")]]
    if atr_pick != "Any":
        table = table[table["ATR trend at signal"] == atr_pick.split()[0]]
    table = table.sort_values(["Days ago", "Avg dollar volume ($M)"], ascending=[True, False]).reset_index(drop=True)

plural = "s" if days > 1 else ""
st.markdown(f"##### {len(table)} stock{'' if len(table) == 1 else 's'} with an NMACD signal in the last {days} "
            f"trading day{plural}")
st.caption(f"Out of {len(liquid):,} stocks that traded on {latest:%b %d, %Y} and pass the liquidity filter. "
           "Days ago: 0 = the latest bar. Band position: 0 = on the lower band, 1 = on the upper band.")
if len(table):
    pct = st.column_config.NumberColumn(format="percent")
    two = st.column_config.NumberColumn(format="%.2f")
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key="screener_table", column_config={
                             "Signal date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                             "NMACD at cross": two, "Since signal": pct, "Last close": two,
                             "Band position at signal": two,
                             "Avg volume (M shares)": st.column_config.NumberColumn(format="%.1f"),
                             "Avg dollar volume ($M)": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Tick the box at the left of a row to open that stock's full backtest, with the rules set on the "
               "Backtest page. It only shows a buy there if all those rules held, not just the NMACD.")
    if event.selection.rows:
        pick = table.iloc[event.selection.rows[0]]
        for k, v in {"bt_source": "Yahoo Finance ticker", "bt_ticker": pick["Ticker"], "bt_bar": "1 day",
                     "bt_use_nmacd": True, "bt_nmacd_level": float(level)}.items():
            ss[k] = v
        ss["screener_note"] = (f"From the screener: {pick['Ticker']} ({pick['Name']}) had an NMACD signal on "
                               f"{pick['Signal date']:%b %d, %Y}, crossing at {pick['NMACD at cross']:.2f}. It's the "
                               "green dot on the NMACD panel below.")
        st.switch_page("backtest_page.py")
else:
    st.write("No stock matches. Try a higher NMACD level, more days, or a lower volume limit.")
