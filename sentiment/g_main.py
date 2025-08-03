# # main.py
# from d_sentiment_loader import build_sentiment_df,get_price_data
# from e_normalize import normalize_sentiment
# import pandas as pd

# df = build_sentiment_df('2025-07-01', '2025-07-11', max_tickers=5)
# if not df.empty:
#     df = normalize_sentiment(df)
#     print(df.head())
# else:
#     print("No sentiment data was generated.")


# price_df = get_price_data(df['ticker'].unique(), '2025-07-01', '2025-07-11')

# merged = pd.merge(df, price_df, on=['date', 'ticker'], how='left')
# print(merged[['date', 'ticker', 'score', 'Open', 'Close']].head())
# main.py
from d_sentiment_loader import build_sentiment_df, get_price_data
from e_normalize import normalize_sentiment
from f_sentiment_module import SentimentModule

import pandas as pd
from datetime import datetime

# Define date range
start_date = '2025-07-01'
end_date = '2025-07-11'

# Step 1: Build sentiment
df = build_sentiment_df(start_date, end_date, max_tickers=5)

if df is not None and not df.empty:
    df = normalize_sentiment(df)
    print("✅ Normalized Sentiment Data:")
    print(df.head())

    # Step 2: Get price data
    tickers = df['ticker'].unique().tolist()
    price_df = get_price_data(tickers, start_date, end_date)

    # Step 3: Merge price + sentiment
    merged = pd.merge(df, price_df, on=['date', 'ticker'], how='left')
    print("\n✅ Merged Data (Sentiment + Price):")
    print(merged[['date', 'ticker', 'score', 'Open', 'Close']].head())

    # Step 4: Build sentiment module
    trading_dates = sorted(df['date'].unique())
    sentiment_module = SentimentModule(df, tickers, trading_dates)

    # Step 5: Simulate use in DRL environment
    print("\n Sample Sentiment Features from Module:")
    print(f"Dates: {trading_dates[0]} to {trading_dates[-1]}")
    print(f"Tickers: {tickers}")

    t_idx = 0  # first day
    ticker_idx = 0  # first ticker
    features = sentiment_module.get(t_idx, ticker_idx)
    print(f"\nSentiment features on day 0 for ticker 0: {features}")

else:
    print(" No sentiment data was generated.")
