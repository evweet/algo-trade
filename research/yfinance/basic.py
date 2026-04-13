import pandas as pd
import yfinance as yf
import copy

tickers = ['MSFT', 'AAPL']
ticker_ohlc_download = {}

# Download the data
for ticker in tickers:
    ticker_ohlc_download[ticker] = yf.download(ticker, period='1y', interval='1mo', multi_level_index=False)
    ticker_ohlc_download[ticker].dropna(inplace=True, how='all')

ticker_ohlc = copy.deepcopy(ticker_ohlc_download)
return_df = pd.DataFrame()

# Calculate the monthly return
for ticker in tickers:
    print("calculating monthly return for ",ticker)
    ticker_ohlc[ticker]["mon_ret"] = ticker_ohlc[ticker]["Close"].pct_change()
    return_df[ticker] = ticker_ohlc[ticker]["mon_ret"]
return_df.dropna(inplace=True)

# Calculate the return for the whole year
acc_return = pd.DataFrame()
for ticker in tickers:
    acc_return[ticker] = (1 + return_df[ticker]).cumprod()