# sentiment_loader.py
import pandas as pd
from a_fetch_data import load_sp500_ticker_list, fetch_news_for_ticker
from c_aggregate import aggregate_sentiments

def build_sentiment_df(start_date, end_date, max_tickers=10):
    tickers = load_sp500_ticker_list()[:max_tickers]
    all_data = []

    for ticker in tickers:
        news = fetch_news_for_ticker(ticker, start_date, end_date)
        if not news:
            continue
        agg_df = aggregate_sentiments(news)
        agg_df['ticker'] = ticker
        all_data.append(agg_df)

    if not all_data:
        return pd.DataFrame(columns=['date', 'score', 'positive', 'negative', 'ticker'])

    final_df = pd.concat(all_data)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df = final_df.sort_values(['ticker', 'date'])

    return final_df

# Example usage
# df = build_sentiment_df('2025-07-01', '2025-07-11')
# print(df.head())


import yfinance as yf

def get_price_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', threads=True)
    records = []
    for ticker in tickers:
        if (ticker,) in data.columns:
            df = data[ticker]
        else:
            df = data.loc[:, pd.IndexSlice[:, ticker]]
            df.columns = df.columns.droplevel(1)
            df = df.copy()
        df['ticker'] = ticker
        df['date'] = df.index
        records.append(df.reset_index(drop=True))
    return pd.concat(records)


