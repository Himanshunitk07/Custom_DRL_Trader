import pandas as pd
from a_fetch_data import load_sp500_ticker_list, fetch_news_for_ticker
from c_aggregate import aggregate_sentiments
from concurrent.futures import ThreadPoolExecutor

def build_sentiment_df(start_date, end_date, max_tickers=50, workers=8):
    tickers = load_sp500_ticker_list()[:max_tickers]

    def process_ticker(ticker):
        try:
            news = fetch_news_for_ticker(ticker, start_date, end_date)
            if not news:
                return None
            agg_df = aggregate_sentiments(news, workers=workers)
            agg_df['ticker'] = ticker
            return agg_df
        except Exception as e:
            print(f"[ERROR] Ticker {ticker}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_ticker, tickers))

    results = [r for r in results if r is not None]
    if not results:
        return pd.DataFrame(columns=['date', 'score', 'positive', 'negative', 'ticker'])

    final_df = pd.concat(results)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df = final_df.sort_values(['ticker', 'date'])

    return final_df
