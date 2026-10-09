"""Screen US-listed stocks for recent Normalized MACD buy signals.

Universe: Nasdaq Trader's daily symbol directory, i.e. every stock listed on Nasdaq, NYSE, NYSE American,
NYSE Arca, Cboe BZX and IEX, without ETFs, test issues, warrants, rights, units, preferred shares and notes.
Prices: a year of daily bars per stock from Yahoo Finance, downloaded in batches. A year is plenty: over the
last weeks it gives the same NMACD as the full history, to within 1e-7.
Signal: the backtest's NMACD buy signal, i.e. the NMACD line closes above its trigger at a level at or below
`level`, on one of the last `days` bars.
"""
from __future__ import annotations

import io
import re
import time
import urllib.request

import numpy as np
import pandas as pd

import atr_bands
import nmacd

SYMBOL_FILES = {
    "nasdaqlisted.txt": ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"),
    "otherlisted.txt": ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                        "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"),
}
EXCHANGES = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}
NOT_A_STOCK = re.compile(r"\b(?:warrants?|rights?|units?|preferred|notes|debentures?|subordinated)\b|%", re.I)
SUFFIX = re.compile(r"[\s,]+(class [a-z]\s+)?(common stock|common shares|ordinary shares|american depositary shares"
                    r"|depositary shares)\b.*$", re.I)
MIN_BARS = 100                  # the NMACD needs about 80 bars to warm up
BATCH, PAUSE, RETRY_PAUSE = 200, 0.5, 5.0


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def clean_name(name: str) -> str:
    """'Apple Inc. - Common Stock' -> 'Apple Inc.'"""
    return SUFFIX.sub("", name.split(" - ")[0]).strip(" ,") or name


def parse_directory(text: str) -> pd.DataFrame:
    """One Nasdaq Trader symbol file (nasdaqlisted.txt or otherlisted.txt) -> Ticker (Yahoo style), Name, Exchange."""
    raw = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)   # keeps the ticker "NA"
    raw.columns = [c.strip() for c in raw.columns]
    sym = "Symbol" if "Symbol" in raw.columns else "ACT Symbol"
    raw = raw[~raw[sym].str.startswith("File Creation Time")]
    exchange = raw["Exchange"].str.strip().map(EXCHANGES).fillna("Other") if "Exchange" in raw.columns else "Nasdaq"
    df = pd.DataFrame({"Symbol": raw[sym].str.strip(), "Name": raw["Security Name"].str.strip(), "Exchange": exchange,
                       "ETF": raw["ETF"].str.strip(), "Test": raw["Test Issue"].str.strip()})
    keep = ((df["ETF"] != "Y") & (df["Test"] != "Y") & df["Symbol"].str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")
            & ~df["Name"].str.contains(NOT_A_STOCK))
    df = df[keep]
    return pd.DataFrame({"Ticker": df["Symbol"].str.replace(".", "-", regex=False),     # BRK.B -> BRK-B
                         "Name": df["Name"].map(clean_name), "Exchange": df["Exchange"]}).reset_index(drop=True)


def load_universe(fetch=_fetch) -> pd.DataFrame:
    frames = []
    for name, urls in SYMBOL_FILES.items():
        error = None
        for url in urls:
            try:
                frames.append(parse_directory(fetch(url)))
                break
            except Exception as e:  # noqa: BLE001  (try the FTP copy)
                error = e
        else:
            raise ValueError(f"couldn't download Nasdaq's list of US stocks ({name}): {error}")
    universe = pd.concat(frames).drop_duplicates("Ticker").sort_values("Ticker").reset_index(drop=True)
    if len(universe) < 1000:
        raise ValueError(f"Nasdaq's list of US stocks looks incomplete ({len(universe)} stocks).")
    return universe


def download(tickers: list[str], period: str = "1y", threads: int = 8) -> dict[str, pd.DataFrame]:
    """Daily bars for a batch of tickers from Yahoo Finance. Tickers Yahoo returns nothing for are left out."""
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=False, actions=False, group_by="ticker",
                      threads=threads, progress=False)
    if raw is None or raw.empty:
        return {}
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({tickers[0]: raw}, axis=1)
    if not set(tickers) & set(raw.columns.get_level_values(0)):
        raw = raw.swaplevel(axis=1)                    # (field, ticker) instead of (ticker, field)
    idx = pd.DatetimeIndex(raw.index)
    raw.index = idx.tz_localize(None) if idx.tz is not None else idx
    out = {}
    for t in set(tickers) & set(raw.columns.get_level_values(0)):
        df = raw[t]
        if not {"Open", "High", "Low", "Close"} <= set(df.columns):
            continue
        df = df[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]]
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).astype("float64")
        if len(df):
            out[t] = df
    return out


def download_all(tickers: list[str], progress=None, batch: int = BATCH):
    """Every ticker in batches, then one more try for those that came back empty (Yahoo sometimes throttles).
    Returns the prices by ticker and the tickers that never came back."""
    data, n = {}, len(tickers)
    chunks = [tickers[i:i + batch] for i in range(0, n, batch)]
    for k, chunk in enumerate(chunks):
        try:
            data.update(download(chunk))
        except Exception:  # noqa: BLE001  (its tickers are retried below)
            pass
        if progress:
            progress(0.9 * (k + 1) / len(chunks), f"Downloaded {min((k + 1) * batch, n):,} of {n:,} stocks")
        time.sleep(PAUSE)
    missing = [t for t in tickers if t not in data]
    if missing:
        time.sleep(RETRY_PAUSE)
        step = max(batch // 2, 1)
        for i in range(0, len(missing), step):
            try:
                data.update(download(missing[i:i + step]))
            except Exception:  # noqa: BLE001
                pass
            if progress:
                progress(0.9 + 0.1 * min(1.0, (i + step) / len(missing)), f"Trying {len(missing):,} stocks again")
    return data, [t for t in tickers if t not in data]


def latest_day(data: dict) -> pd.Timestamp:
    """The most common last date, i.e. the latest trading day. Stocks whose prices stop earlier aren't trading."""
    return pd.Series([df.index[-1] for df in data.values()]).mode().iloc[0]


def nmacd_tails(data: dict, min_volume: float, min_price: float, fast: int = 13, slow: int = 21, trigger: int = 9,
                normalize: int = 50, ma_type: int = 1, tail: int = 10):
    """For every stock that is still trading and passes the liquidity filter (average volume over the last 20
    days, last close): its last `tail` bars of NMACD and trigger, with the bar before to spot crosses.
    Returns (tails, liquid stocks, latest trading day)."""
    cols = ["Ticker", "Date", "Close", "NMACD", "Trigger", "Prev NMACD", "Prev trigger", "Bars ago"]
    if not data:
        return pd.DataFrame(columns=cols), pd.DataFrame(), None
    latest = latest_day(data)
    tails, liquid = [], []
    for t, df in data.items():
        if df.index[-1] != latest or len(df) < MIN_BARS or "Volume" not in df.columns:
            continue
        close, volume = df["Close"], df["Volume"].tail(20)
        avg_volume = volume.mean()
        if not (avg_volume >= min_volume and close.iloc[-1] >= min_price):
            continue
        nm = nmacd.compute(close, fast, slow, trigger, normalize, ma_type)
        part = pd.DataFrame({"Ticker": t, "Date": close.index, "Close": close.to_numpy(),
                             "NMACD": nm["NMACD"].to_numpy(), "Trigger": nm["Trigger"].to_numpy(),
                             "Prev NMACD": nm["NMACD"].shift(1).to_numpy(),
                             "Prev trigger": nm["Trigger"].shift(1).to_numpy()}).tail(tail)
        part["Bars ago"] = np.arange(len(part))[::-1]
        tails.append(part)
        liquid.append({"Ticker": t, "Last close": close.iloc[-1], "Avg volume": avg_volume,
                       "Avg dollar volume": (close.tail(20) * volume).mean()})
    tails = pd.concat(tails, ignore_index=True) if tails else pd.DataFrame(columns=cols)
    return tails, pd.DataFrame(liquid).set_index("Ticker") if liquid else pd.DataFrame(), latest


def find_signals(tails: pd.DataFrame, level: float, days: int) -> pd.DataFrame:
    """The most recent NMACD buy signal of each stock within the last `days` bars (0 = the latest bar)."""
    t = tails[tails["Bars ago"] < days]
    hit = (t["NMACD"] > t["Trigger"]) & (t["Prev NMACD"] <= t["Prev trigger"]) & (t["NMACD"] <= level)
    return t[hit].sort_values("Bars ago").drop_duplicates("Ticker")


def atr_at(df: pd.DataFrame, when, source: str = "close", length: int = 5, mult: float = 7.6):
    """ATR Trend Bands trend ('Up' / 'Down') and band position (0 = lower band, 1 = upper band) on the bar `when`."""
    row = atr_bands.compute(df, source, int(length), float(mult)).loc[when]
    return row["Trend"], (row["Close"] - row["Lower"]) / row["Width"]
