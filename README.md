# ATR Trend Bands + NMACD + Normalized RSI backtest, with an NMACD screener

A Streamlit website with two pages.

**Backtest** tests long-only rules on one stock, mixing three indicators. Every rule can be switched on
or off in the sidebar:

- **Buy rules:** the ATR Trend Bands trend (down, up, any, or "buy when it turns green"), the close near
  the lower band, the NMACD crossing above its trigger at or below a chosen level, and the normalized RSI
  crossing up through its lower bound. The signals switched on can be required on the same bar or within
  a few bars of each other.
- **Sell rules:** NMACD hits +1, the ATR Trend Bands turn green, they turn red, or the normalized RSI
  falls back below its upper bound, whichever comes first. A stop loss and a time limit are optional.
- **Buy signals while already holding** are either ignored (one position at a time) or each one buys
  more (a chosen % of the account, up to a chosen number of buys).
- **Test period:** pick the dates to test and chart; the indicators still use the earlier data.

**NMACD screener** scans every US-listed stock (Nasdaq, NYSE, NYSE American, NYSE Arca, Cboe and IEX;
no ETFs, warrants, units, rights or preferred shares) with enough trading volume for an NMACD buy signal
in the last few trading days: the line crossing above its trigger at or below a chosen level, the same
signal as on the Backtest page. Tick a stock to open its full backtest with the rules set on the
Backtest page. The list of stocks comes from Nasdaq's daily symbol directory and the prices from Yahoo
Finance. The first scan of the day downloads a year of prices for about 5,000 stocks and takes a few
minutes; later scans that day reuse it.

Backtest prices: the Netflix file in `data/` (Nasdaq export, Oct 2016 to Oct 2026), any uploaded CSV of
daily prices (Nasdaq, Yahoo or TradingView export), or any Yahoo Finance ticker with bars from 1 minute
to 1 month.

## Put it online

Upload everything in this folder, including `data/` and `.streamlit/`, to a GitHub repository, then
create an app on https://share.streamlit.io with the main file `app.py`.

## Run it on a computer with Python

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py`: the entry point; sets up the two pages and keeps your settings when you switch between them
- `backtest_page.py`: the Backtest page
- `screener_page.py`: the NMACD screener page
- `scanner.py`: the list of US stocks, the bulk download from Yahoo and the signal search
- `backtest.py`: the rules, the trade simulation and the statistics
- `nmacd.py`: the Normalized MACD, line by line from the TradingView script
- `nrsi.py`: the normalized RSI (Wilder's RSI, rescaled to -1..+1 the same way as the NMACD)
- `atr_bands.py`: the ATR Trend Bands replica (same as the ATR Trend Bands app)
- `data_io.py`: reads the CSV formats and Yahoo Finance at the chosen bar size
