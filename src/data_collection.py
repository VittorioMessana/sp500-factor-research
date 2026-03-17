# data_collection.py
# Vittorio Messana, 2026
#
# downloading all the price data I need for the factor analysis
# same setup as the regime model project but this time
# I'm focused on cross-sectional factor returns not time series regimes

import yfinance as yf
import pandas as pd
import numpy as np
import os

START_DATE = "2010-01-01"
END_DATE   = "2025-12-31"

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSM", "AVGO", "ORCL", "ASML",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "SPGI", "MCO", "ICE",
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "TGT", "HD",
    "XOM", "CVX", "CAT", "DE", "HON", "UPS", "BA", "MMM", "GE", "LMT"
]

print("downloading prices...")
prices = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
print(f"got {prices.shape[0]} days, {prices.shape[1]} stocks")

print("downloading SPY...")
spy         = yf.download("SPY", start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
spy_returns = spy.pct_change().dropna()

print("computing returns...")
returns = prices.pct_change().dropna()

# removing stocks with too much missing data
missing_pct  = returns.isnull().sum() / len(returns)
good_tickers = missing_pct[missing_pct < 0.20].index
returns      = returns[good_tickers].fillna(0)
prices       = prices[good_tickers]

common_dates = returns.index.intersection(spy_returns.index)
returns      = returns.loc[common_dates]
spy_returns  = spy_returns.loc[common_dates]
prices       = prices.loc[common_dates]

print(f"{len(common_dates)} trading days")
print(f"from {common_dates[0].date()} to {common_dates[-1].date()}")
print(f"{len(good_tickers)} stocks after cleaning")

os.makedirs("data", exist_ok=True)
returns.to_csv("data/stock_returns.csv")
spy_returns.to_csv("data/spy_returns.csv")
prices.to_csv("data/stock_prices.csv")

print("done")
