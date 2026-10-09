"""Long-only backtest mixing three indicators: ATR Trend Bands, Normalized MACD (NMACD) and Normalized RSI
(NRSI). Every buy and sell condition can be switched on or off.

Buy, checked at each bar's close:
  Filters, checked on the buy bar:
    - ATR Trend Bands (`atr_entry`): "down" = the trend is down (red), "up" = up (green), "off" = any trend,
      "green" = a signal of its own: the trend turned green within the signal window and is still green.
    - Near the lower band (`use_near_band`): the close is within `near_band` of the corridor, measured
      from the lower band (0 = on the lower band, 1 = on the upper band).
  Signals, each within the last `window` bars, the last of them on the buy bar:
    - NMACD (`use_nmacd`): the NMACD line closes above its trigger line, at a level at or below `nmacd_level`.
    - NRSI (`use_nrsi`): the normalized RSI closes above `nrsi_lower` after closing at or below it.
    - The ATR turning green, when atr_entry == "green".
  With no signal switched on, every bar that passes the filters is a buy signal.
Sell, whichever comes first of those switched on: NMACD reaches `exit_level` (it can't print exactly +1;
see nmacd.py), the ATR Trend Bands turn green (flip up), they turn red (flip down), the NRSI closes below
`nrsi_upper` after closing at or above it. Plus an optional stop loss / time limit.

Repeat buy signals (`pyramid`): off = one position at a time with all the cash, and buy signals while
holding are ignored. On = every buy signal buys another lot worth `entry_size` of the account value (never
more than the cash left), up to `max_entries` lots, and a sell signal sells all the lots together.
Test period (`start`, `end`, ISO dates): the indicators use all the data, so they're warmed up when the
test starts; trades, equity and statistics only cover the test period.
Orders fill at the next bar's open (TradingView's default) or the signal bar's close.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import reduce
from itertools import product
from operator import and_, or_

import numpy as np
import pandas as pd

import atr_bands
import nmacd
import nrsi

ATR_ENTRY = ("down", "up", "green", "off")
INDICATOR_FIELDS = ("atr_length", "atr_mult", "atr_source", "fast", "slow", "trigger", "normalize", "ma_type",
                    "rsi_length", "rsi_normalize")
# Sell switches: setting -> (signal column, reason shown in the trade log)
EXITS = {"exit_nmacd": ("NMACD exit", "NMACD hit +1"),
         "exit_atr_green": ("ATR turned green", "ATR turned green"),
         "exit_atr_red": ("ATR turned red", "ATR turned red"),
         "exit_nrsi": ("NRSI exit", "NRSI fell below upper bound")}


@dataclass(frozen=True)
class Settings:
    # Indicators
    atr_length: int = 5
    atr_mult: float = 7.6
    atr_source: str = "close"
    fast: int = 13
    slow: int = 21
    trigger: int = 9
    normalize: int = 50
    ma_type: int = 1
    rsi_length: int = 14
    rsi_normalize: int = 50
    # Buy
    atr_entry: str = "down"         # "down", "up", "green" (buy when it turns green) or "off"
    use_near_band: bool = True
    near_band: float = 0.25
    use_nmacd: bool = True
    nmacd_level: float = -0.8       # highest NMACD level accepted on the bar it crosses above its trigger
    use_nrsi: bool = False
    nrsi_lower: float = -0.8
    window: int = 3                 # bars within which all the buy signals switched on must happen
    # Sell
    exit_nmacd: bool = True
    exit_level: float = 0.99
    exit_atr_green: bool = False
    exit_atr_red: bool = False
    exit_nrsi: bool = False
    nrsi_upper: float = 0.8
    stop_loss: float = 0.0          # fraction below the (average) entry price, 0 = off
    max_bars: int = 0               # bars since the (first) buy, 0 = off
    # Position and execution
    pyramid: bool = False           # True = buy again on every signal while holding, sell everything together
    entry_size: float = 0.25        # with pyramid: share of the account value each buy uses
    max_entries: int = 4            # with pyramid: most buys held at once
    fill: str = "open"              # "open" = next bar's open, "close" = the signal bar's close
    cost_bps: float = 5.0           # per side
    capital: float = 100_000.0
    start: str | None = None        # test period, ISO dates; None = from the first / to the last bar
    end: str | None = None


def indicators(prices: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Every indicator value on all the data. Only the INDICATOR_FIELDS settings matter here."""
    bands = atr_bands.compute(prices, s.atr_source, s.atr_length, s.atr_mult)
    nm = nmacd.compute(prices["Close"], s.fast, s.slow, s.trigger, s.normalize, s.ma_type)
    rs = nrsi.compute(prices["Close"], s.rsi_length, s.rsi_normalize)
    d = pd.concat([prices[["Open", "High", "Low", "Close"]],
                   bands[["Upper", "Lower", "Width", "Trend", "Signal", "ATR"]].rename(columns={"Signal": "ATR signal"}),
                   nm[["NMACD", "Trigger", "Hist"]], rs[["RSI", "NRSI"]]], axis=1)
    d["Band position"] = (d["Close"] - d["Lower"]) / d["Width"]
    prev_n, prev_t = d["NMACD"].shift(1), d["Trigger"].shift(1)
    d["Cross up"] = (d["NMACD"] > d["Trigger"]) & (prev_n <= prev_t)
    d["ATR turned green"] = d["ATR signal"] == "L"
    d["ATR turned red"] = d["ATR signal"] == "S"
    return d


def buy_signal_columns(s: Settings) -> list[str]:
    """The buy signals (one-bar events) switched on."""
    return [col for col, on in (("ATR turned green", s.atr_entry == "green"), ("NMACD signal", s.use_nmacd),
                                ("NRSI signal", s.use_nrsi)) if on]


def signals(base: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Buy and sell conditions for these thresholds. Cheap: `base` comes from indicators()."""
    if s.atr_entry not in ATR_ENTRY:
        raise ValueError(f"atr_entry must be one of {ATR_ENTRY}, not {s.atr_entry!r}.")
    d = base.copy()
    prev_r = d["NRSI"].shift(1)
    d["NMACD signal"] = d["Cross up"] & (d["NMACD"] <= s.nmacd_level)
    d["NRSI signal"] = (d["NRSI"] > s.nrsi_lower) & (prev_r <= s.nrsi_lower)
    d["Near lower band"] = d["Band position"] <= s.near_band
    always = pd.Series(True, index=d.index)
    d["ATR filter"] = {"down": d["Trend"] == "Down", "up": d["Trend"] == "Up", "green": d["Trend"] == "Up",
                       "off": always}[s.atr_entry]
    entry = d["ATR filter"] & (d["Near lower band"] if s.use_near_band else always)
    events = buy_signal_columns(s)
    if events:
        w = max(int(s.window), 1)
        for e in events:
            d[f"{e} in window"] = d[e].astype(float).rolling(w, min_periods=1).max() > 0
        entry = entry & reduce(and_, [d[f"{e} in window"] for e in events]) & reduce(or_, [d[e] for e in events])
    d["Entry signal"] = entry
    d["NMACD exit"] = d["NMACD"] >= s.exit_level
    d["NRSI exit"] = (d["NRSI"] < s.nrsi_upper) & (prev_r >= s.nrsi_upper)
    on = [col for key, (col, _) in EXITS.items() if getattr(s, key)]
    d["Exit signal"] = reduce(or_, [d[col] for col in on]) if on else False
    return d


def test_period(d: pd.DataFrame, s: Settings) -> pd.DataFrame:
    keep = np.ones(len(d), dtype=bool)
    if s.start:
        keep &= d.index >= pd.Timestamp(s.start)
    if s.end:
        keep &= d.index < pd.Timestamp(s.end) + pd.Timedelta(days=1)
    return d[keep]


def run(prices: pd.DataFrame, s: Settings, base: pd.DataFrame | None = None):
    """Bar-by-bar simulation over the test period, long only, compounding.

    Without pyramiding: one position at a time with 100% of the cash; buy signals while holding are
    ignored. With pyramiding: each buy signal adds a lot (see the module notes) and every exit sells all
    lots at once; the stop loss is measured from the lots' average price and the time limit from the
    first buy. Each lot is one row of the trade log. Pass `base` (from indicators()) to skip recomputing.
    """
    d = test_period(signals(indicators(prices, s) if base is None else base, s), s)
    if len(d) < 2:
        raise ValueError("The test period needs at least 2 bars.")
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    entry_sig = d["Entry signal"].to_numpy()
    exits = [(reason, d[col].to_numpy()) for key, (col, reason) in EXITS.items() if getattr(s, key)]
    trend, band_pos = d["Trend"].to_numpy(), d["Band position"].to_numpy()
    nm_val, rs_val = d["NMACD"].to_numpy(), d["NRSI"].to_numpy()
    dates, n, cost = d.index, len(d), s.cost_bps / 1e4
    most = max(int(s.max_entries), 1) if s.pyramid else 1
    equity = np.empty(n)
    cash, lots, trades, positions = s.capital, [], [], 0
    pending_entry = pending_exit = None

    def held():
        return sum(t["shares"] for t in lots)

    def stop_price():
        if s.stop_loss <= 0 or not lots:
            return -np.inf
        avg = lots[0]["entry_px"] if len(lots) == 1 else sum(t["shares"] * t["entry_px"] for t in lots) / held()
        return avg * (1 - s.stop_loss)

    def exit_reason(i):
        why = [reason for reason, flags in exits if flags[i]]
        if why:
            return " and ".join(why)
        return "Time limit" if s.max_bars > 0 and i - lots[0]["entry_i"] >= s.max_bars else None

    def buy(i_signal, i_fill, px):
        nonlocal cash, positions
        amount = min(cash, s.entry_size * (cash + held() * px)) if s.pyramid else cash
        if amount <= 1e-9 * s.capital:             # no cash left: the signal is skipped
            return
        if not lots:
            positions += 1
        lots.append({"position": positions, "buy_no": len(lots) + 1, "signal_i": i_signal, "entry_i": i_fill,
                     "entry_px": px, "invested": amount, "shares": amount * (1 - cost) / px, "low": px})
        cash -= amount

    def sell_all(i_fill, px, reason, status="Closed"):
        nonlocal cash
        for t in lots:
            value = t["shares"] * px * (1 - cost)
            sig = t["signal_i"]
            row = {
                "Signal date": dates[sig], "Entry date": dates[t["entry_i"]], "Entry price": t["entry_px"],
                "Invested ($)": t["invested"], "Exit date": dates[i_fill], "Exit price": px, "Exit reason": reason,
                "Bars held": i_fill - t["entry_i"], "Return": value / t["invested"] - 1,
                "P&L ($)": value - t["invested"], "Worst drawdown in trade": t["low"] / t["entry_px"] - 1,
                "ATR trend at signal": trend[sig], "Band position at signal": band_pos[sig],
                "NMACD at signal": nm_val[sig], "NRSI at signal": rs_val[sig], "Status": status}
            trades.append({"Position": t["position"], "Buy #": t["buy_no"], **row} if s.pyramid else row)
            if status == "Closed":
                cash += value
        if status == "Closed":
            lots.clear()

    def track_low(i):
        for t in lots:
            t["low"] = min(t["low"], l[i])

    for i in range(n):
        # Orders decided at the previous close fill at this bar's open.
        if pending_exit and lots:
            sell_all(i, o[i], pending_exit)
            pending_exit = None
        if lots and i > lots[0]["signal_i"] and o[i] <= stop_price():   # gapped through the stop overnight
            track_low(i)
            sell_all(i, o[i], "Stop loss")
        if pending_entry is not None:
            if len(lots) < most:
                buy(pending_entry, i, o[i])
            pending_entry = None
        if lots:
            stop_px = stop_price()
            track_low(i)
            if i > lots[0]["signal_i"] and l[i] <= stop_px:
                sell_all(i, min(o[i], stop_px), "Stop loss")
        # Decisions at this bar's close: sell everything...
        if lots and i > lots[0]["signal_i"]:
            reason = exit_reason(i)
            if reason:
                if s.fill == "close":
                    sell_all(i, c[i], reason)
                elif i + 1 < n:
                    pending_exit = reason
        # ... and buy (again).
        if entry_sig[i] and pending_entry is None and not pending_exit and len(lots) < most:
            if s.fill == "close":
                buy(i, i, c[i])
            elif i + 1 < n:
                pending_entry = i
        equity[i] = cash + held() * c[i]
    if lots:                                       # still open: mark at the last close
        sell_all(n - 1, c[-1], "Still open", status="Open")
    return d, pd.Series(equity, index=dates), pd.DataFrame(trades)


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Bars in a year, to annualise the Sharpe ratio: 252 for daily stock prices, about 52 weekly and 12
    monthly, and bars per session x 252 intraday (365 days for markets that also trade at weekends)."""
    if len(index) < 3:
        return 252.0
    day = index.normalize()
    days = day.unique()
    per_year = 365.0 if (days.dayofweek >= 5).mean() > 0.1 else 252.0
    per_day = pd.Series(1, index=day).groupby(level=0).size().median()
    if per_day > 1:                                # intraday
        return float(per_day * per_year)
    gap = pd.Series(days).diff().dt.total_seconds().median() / 86_400
    years = (days[-1] - days[0]).days / 365.25
    return per_year if gap < 4 or years <= 0 else (len(days) - 1) / years


def stats(eq: pd.Series, trades: pd.DataFrame, prices: pd.DataFrame, capital: float) -> dict:
    """`prices` covers the same bars as `eq` (the test period), for buy & hold."""
    r = eq.pct_change().fillna(0.0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    bh = prices["Close"] / prices["Close"].iloc[0] * capital
    closed = trades[trades["Status"] == "Closed"] if len(trades) else trades
    wins = closed[closed["Return"] > 0] if len(closed) else closed
    losses = closed[closed["Return"] <= 0] if len(closed) else closed
    if len(trades):
        # Buys made while already holding belong to the same position and leave with it.
        pos_id = trades["Position"] if "Position" in trades else pd.Series(np.arange(len(trades)), index=trades.index)
        in_mkt = trades.groupby(pos_id)["Bars held"].max().sum()
        n_pos, n_closed_pos = pos_id.nunique(), pos_id[trades["Status"] == "Closed"].nunique()
    else:
        in_mkt = n_pos = n_closed_pos = 0
    out = {
        "Trades": len(trades), "Closed trades": len(closed), "Positions": n_pos, "Closed positions": n_closed_pos,
        "Win rate": len(wins) / len(closed) if len(closed) else np.nan,
        "Average win": wins["Return"].mean() if len(wins) else np.nan,
        "Average loss": losses["Return"].mean() if len(losses) else np.nan,
        "Average trade": closed["Return"].mean() if len(closed) else np.nan,
        "Best trade": closed["Return"].max() if len(closed) else np.nan,
        "Worst trade": closed["Return"].min() if len(closed) else np.nan,
        "Profit factor": (wins["P&L ($)"].sum() / -losses["P&L ($)"].sum()) if len(losses) and losses["P&L ($)"].sum() < 0 else np.nan,
        "Total P&L ($)": eq.iloc[-1] - capital,
        "Total return": eq.iloc[-1] / capital - 1,
        "CAGR": (eq.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Sharpe": r.mean() / r.std() * np.sqrt(bars_per_year(eq.index)) if r.std() > 0 else np.nan,
        "Average bars held": trades["Bars held"].mean() if len(trades) else np.nan,
        "Time in market": in_mkt / max(len(eq) - 1, 1),
        "Worst drawdown inside a trade": trades["Worst drawdown in trade"].min() if len(trades) else np.nan,
        "Buy & hold return": bh.iloc[-1] / capital - 1,
        "Buy & hold max drawdown": (bh / bh.cummax() - 1).min(),
        "Years": years,
    }
    return out


def funnel(d: pd.DataFrame, s: Settings, unit: str = "bars") -> pd.DataFrame:
    """How many bars of the test pass each buy condition, cumulatively."""
    events = buy_signal_columns(s)
    w = max(int(s.window), 1)
    together = len(events) >= 2 and w > 1
    ready = d["Trend"].ne("")
    if s.use_nmacd or s.exit_nmacd:
        ready &= d["NMACD"].notna()
    if s.use_nrsi or s.exit_nrsi:
        ready &= d["NRSI"].notna()
    steps, mask = [(f"All {unit} with the indicators ready", ready)], ready
    if s.atr_entry != "off":
        mask = mask & d["ATR filter"]
        steps.append((f"... ATR trend is {'down' if s.atr_entry == 'down' else 'up'}", mask))
    if s.use_near_band:
        mask = mask & d["Near lower band"]
        steps.append(("... and the close is near the lower band", mask))
    names = {"ATR turned green": "the ATR turned green",
             "NMACD signal": f"NMACD crossed above its trigger at or below {s.nmacd_level:g}",
             "NRSI signal": f"the normalized RSI crossed up through {s.nrsi_lower:g}"}
    for e in events:
        if e == "NMACD signal" and not together:
            mask = mask & d["Cross up"]
            steps.append(("... and NMACD crosses above its trigger", mask))
            names[e] = f"the cross is at or below {s.nmacd_level:g}"
        mask = mask & d[f"{e} in window" if together else e]
        steps.append((f"... and {names[e]}" + (f" within the last {w} bars" if together else ""), mask))
    if not mask.equals(d["Entry signal"]):
        steps.append(("Buy signals" + (" (the last of them on this bar)" if together else ""), d["Entry signal"]))
    return pd.DataFrame({"Condition": [x[0] for x in steps], unit.capitalize(): [int(x[1].sum()) for x in steps]})


def _summary(eq, tr, d, capital) -> dict:
    st = stats(eq, tr, d, capital)
    return {"Trades": st["Trades"], "Win rate": st["Win rate"], "Average trade": st["Average trade"],
            "Total return": st["Total return"], "Max drawdown": st["Max drawdown"]}


def compare_signals(prices: pd.DataFrame, s: Settings, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """The same backtest with the NMACD and NRSI buy signals switched off, on one at a time, and both on."""
    base = indicators(prices, s) if base is None else base
    rows = []
    for name, use_nm, use_rs in (("Neither", False, False), ("NMACD only", True, False),
                                 ("Normalized RSI only", False, True), ("Both", True, True)):
        d, eq, tr = run(prices, replace(s, use_nmacd=use_nm, use_nrsi=use_rs), base)
        rows.append({"Buy signals": name, "Signal bars": int(d["Entry signal"].sum()), **_summary(eq, tr, d, s.capital)})
    return pd.DataFrame(rows)


def sensitivity(prices: pd.DataFrame, s: Settings, grid: dict, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rerun the backtest for every combination of the values in `grid` (setting name -> values)."""
    base = indicators(prices, s) if base is None else base
    rows = []
    for combo in product(*grid.values()):
        values = dict(zip(grid, combo))
        d, eq, tr = run(prices, replace(s, **values), base)
        rows.append({**values, **_summary(eq, tr, d, s.capital)})
    return pd.DataFrame(rows)
