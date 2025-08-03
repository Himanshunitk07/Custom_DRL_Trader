import numpy as np
import pandas as pd

class SentimentModule:
    def __init__(self, sentiment_df, tickers, trading_timestamps):
        self.tickers = tickers
        self.timestamps = trading_timestamps
        self.date_list = [ts.date() for ts in trading_timestamps]
        self.unique_dates = sorted(set(self.date_list))

        self.date_index_map = {idx: self.unique_dates.index(date) for idx, date in enumerate(self.date_list)}
        self.ticker_map = {t: i for i, t in enumerate(tickers)}
        self.tensor = self._build_tensor(sentiment_df)

    def _build_tensor(self, df):
        D = len(self.unique_dates)
        N = len(self.tickers)
        F = 3  # score, positive, negative

        tensor = np.zeros((D, N, F))

        # Pre-compute date-to-index map once
        date_to_index = {d: i for i, d in enumerate(self.unique_dates)}

        for _, row in df.iterrows():
            d = row['date'].date()
            t = row['ticker']
            if t in self.ticker_map and d in date_to_index:
                d_idx = date_to_index[d]
                a_idx = self.ticker_map[t]
                tensor[d_idx, a_idx] = [row['score'], row['positive'], row['negative']]

        return tensor

    def get(self, t_step, asset_idx):
        d_idx = self.date_index_map.get(t_step, None)
        if d_idx is None:
            return np.zeros(3)
        return self.tensor[d_idx, asset_idx]
