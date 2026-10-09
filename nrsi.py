"""Normalized RSI: Wilder's RSI of the close, rescaled to -1..+1 against its own range over the last
`normalize` bars, the same way the Normalized MACD rescales its momentum value.

    rsi   = ta.rsi(close, length)
    NRSI  = (rsi - lowest(rsi, n)) / (highest(rsi, n) - lowest(rsi, n) + 0.000001) * 2 - 1

NRSI is -1 when the RSI is at its lowest of the last n bars and +1 at its highest. With normalize < 2
there's no rescaling, just a change of units: NRSI = (rsi - 50) / 50, so RSI 30 = -0.4 and RSI 70 = +0.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine's ta.rma: starts at the first full window of values, seeded with their simple average."""
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(np.isfinite(x))
    if len(ok) == 0 or len(x) - ok[0] < n:
        return out
    s = ok[0] + n - 1
    out[s] = np.mean(x[ok[0]:s + 1])
    for i in range(s + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Pine's ta.rsi: 100 when there were no down moves, 0 when there were no up moves."""
    change = close.diff().to_numpy(dtype=float)
    up = rma(np.where(np.isfinite(change), np.maximum(change, 0), np.nan), length)
    down = rma(np.where(np.isfinite(change), np.maximum(-change, 0), np.nan), length)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(down == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / down)))
    value[~(np.isfinite(up) & np.isfinite(down))] = np.nan
    return pd.Series(value, index=close.index)


def compute(close: pd.Series, length: int = 14, normalize: int = 50) -> pd.DataFrame:
    r = rsi(close, length)
    if normalize < 2:
        norm = (r - 50) / 50
    else:
        lo, hi = r.rolling(normalize).min(), r.rolling(normalize).max()
        norm = (r - lo) / (hi - lo + 0.000001) * 2 - 1
    return pd.DataFrame({"RSI": r, "NRSI": norm})
