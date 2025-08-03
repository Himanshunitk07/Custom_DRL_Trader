from Shorttermenv import fetch_intraday, ShortTermDynamicTrader
from d_sentiment_loader import build_sentiment_df
from e_normalize import normalize_sentiment
from f_sentiment_module import SentimentModule
import pandas as pd

def build_sentiment_env(tickers, interval='5m', period='7d', min_rows=100, window_size=5):
    # Step 1: Fetch data
    data = fetch_intraday(tickers, interval=interval, period=period, min_rows=min_rows)
    if not data:
        raise ValueError("No valid intraday data fetched.")

    # Step 2: Date range from first ticker
    first_df = list(data.values())[0]
    start_date = first_df.index[0].date().isoformat()
    end_date = first_df.index[-1].date().isoformat()

    # Step 3: Sentiment
    sent_df = build_sentiment_df(start_date, end_date, len(tickers))
    if sent_df.empty:
        raise ValueError("No sentiment data available.")
    sent_df = normalize_sentiment(sent_df)

    # Step 4: Sentiment Module
    trading_dates = sorted(sent_df['date'].unique())
    sentiment_module = SentimentModule(sent_df, tickers, trading_dates)

    # Step 5: Build env
    env = ShortTermDynamicTrader(
        sentiment_module=sentiment_module,
        data_dict=data,
        window_size=window_size
    )

    return env
