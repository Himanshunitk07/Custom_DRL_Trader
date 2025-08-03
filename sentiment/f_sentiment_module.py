# sentiment_module.py
import numpy as np

class SentimentModule:
    def __init__(self, sentiment_df, tickers, trading_dates):
        self.tickers = tickers
        self.dates = trading_dates
        self.tensor = self.build_tensor(sentiment_df)

    def build_tensor(self, df):
        T = len(self.dates)
        N = len(self.tickers)
        F = 3  # ['score', 'positive', 'negative']

        tensor = np.zeros((T, N, F))
        date_map = {d: i for i, d in enumerate(self.dates)}
        ticker_map = {t: i for i, t in enumerate(self.tickers)}

        for _, row in df.iterrows():
            if row['date'] in date_map and row['ticker'] in ticker_map:
                t = date_map[row['date']]
                a = ticker_map[row['ticker']]
                tensor[t, a, :] = [row['score'], row['positive'], row['negative']]
        return tensor

    def get(self, t, asset_idx):
        return self.tensor[t, asset_idx]
