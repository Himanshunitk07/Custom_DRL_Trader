import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands

class IntradayFeatureExtractor:
    def __init__(self, indicators=None, window_size=5):
        self.window_size = window_size
        self.indicators = indicators or ['Open', 'Close', 'rsi', 'macd', 'obv', 'bb_upper', 'bb_lower','minute_norm']

    def extract(self, df, current_step):
        window = df.iloc[current_step : current_step + self.window_size]
        return window[self.indicators].to_numpy()

class ShortTermDynamicTrader(gym.Env):
    def __init__(self, sentiment_module, data_dict, window_size=5, initial_balance=100000, transaction_cost_pct=0.001, slippage_pct=0.0005):
        super().__init__()
        self.data = data_dict
        self.asset_names = list(data_dict.keys())
        self.n_assets = len(self.asset_names)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        ref_asset = self.asset_names[0]
        self.trading_timestamps = self.data[ref_asset].index.to_list()
        self.sentiment_module = sentiment_module
        self.extractor = IntradayFeatureExtractor(window_size=window_size)
        self.features = self.extractor.indicators
        self.n_features = len(self.features)
        self.reset()

        obs_len = self.n_assets * self.n_features * self.window_size + self.n_assets + 1 + self.n_assets * 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.position = np.zeros(self.n_assets)
        self.trades = []
        self.net_worths = [self.initial_balance]  
        return self._get_obs().astype(np.float32), {}



    def step(self, action):
        action = np.clip(action, -1, 1)

        current_prices = np.array([
            (self.data[asset].iloc[self.current_step]['Close'] + self.data[asset].iloc[self.current_step]['Open']) / 2
            for asset in self.asset_names
        ])

        for i in range(self.n_assets):
            price = current_prices[i]
            slippage = price * self.slippage_pct

            if action[i] > 0:
                alloc_cash = action[i] * self.balance
                units = int(alloc_cash / (price + slippage))
                if units > 0:
                    trade_cost = units * (price + slippage)
                    fee = trade_cost * self.transaction_cost_pct
                    total_cost = trade_cost + fee
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        self.position[i] += units
                        self.trades.append((self.current_step, i, 'buy', price + slippage, units, fee))

            elif action[i] < 0:
                units = int(-action[i] * self.position[i])
                if units > 0:
                    revenue = units * (price - slippage)
                    fee = revenue * self.transaction_cost_pct
                    total_gain = revenue - fee
                    self.balance += total_gain
                    self.position[i] -= units
                    self.trades.append((self.current_step, i, 'sell', price - slippage, units, fee))

        net_worth = self.balance + np.sum(self.position * current_prices)
        reward = float((net_worth - self.prev_net_worth) / self.prev_net_worth)
        reward = np.clip(reward, -0.05, 0.05)
        self.prev_net_worth = net_worth
        self.current_step += 1
        self.net_worths.append(net_worth)

        min_len = min(len(df) for df in self.data.values())
        terminated = bool(
            self.current_step + self.window_size >= min_len
            or (self.balance <= 0 and np.sum(self.position) == 0)
        )
        truncated = False  # No time-limit-based cutoff used here

        obs = self._get_obs().astype(np.float32)
        info = {
            "net_worth": net_worth,
            "balance": self.balance,
            "positions": self.position.copy(),
            "trades": list(self.trades),
            "net_worths": self.net_worths,
        }

        return obs, reward, terminated, truncated, info



    def _get_obs(self):
        windows = [self.extractor.extract(self.data[asset], self.current_step) for asset in self.asset_names]
        sentiments = [self.sentiment_module.get(self.current_step, idx) for idx in range(self.n_assets)]

        obs_tensor = np.stack(windows, axis=1).flatten()
        sentiment_tensor = np.concatenate(sentiments)
        agent_state = np.concatenate((self.position.flatten(), [float(self.balance)]))
        return np.concatenate([obs_tensor, sentiment_tensor, agent_state])

    def render(self, mode='human'):
        current_prices = [
            float((self.data[asset].iloc[self.current_step]['Close'] + self.data[asset].iloc[self.current_step]['Open']) / 2)
            for asset in self.asset_names
        ]
        net_worth = self.balance + np.sum(self.position * np.array(current_prices))
        print(f"\nStep: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Net Worth: ${net_worth:.2f}")
        for i, asset in enumerate(self.asset_names):
            print(f"{asset}: {self.position[i]} units at ${current_prices[i]:.2f}")
        print("Last Trade:", self.trades[-1] if self.trades else "No trades yet.")

def add_indicators(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(0)
        except Exception:
            return None
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        return None
    close = df['Close']
    volume = df['Volume']
    if isinstance(close, pd.DataFrame) or isinstance(volume, pd.DataFrame):
        return None
    try:
        df['rsi'] = RSIIndicator(close=close, window=14).rsi()
        df['macd'] = MACD(close=close).macd_diff()
        df['obv'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
    except:
        return None
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['minute_norm'] = (df['minute_of_day'] - 570) / (960 - 570)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def fetch_intraday(tickers, interval='5m', period='7d', min_rows=100):
    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, interval=interval, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            df = add_indicators(df)
            if df is None or df.shape[0] < min_rows:
                continue
            data_dict[ticker] = df
        except:
            continue

    if not data_dict:
        return {}

    ref_ticker = min(data_dict.items(), key=lambda x: len(x[1]))[0]
    ref_index = data_dict[ref_ticker].index
    for k in data_dict:
        data_dict[k] = data_dict[k].reindex(ref_index).ffill().bfill()
    return data_dict


'''def fetch_intraday(tickers, interval='5m', period='5d', min_rows=50):

    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, interval=interval, period=period, progress=False)
            if df.empty:
                print(f"[SKIPPED] {ticker}: No data returned.")
                continue
            df = add_indicators(df)
            if df.shape[0] < min_rows:
                print(f"[SKIPPED] {ticker}: Not enough rows after indicators ({df.shape[0]} rows).")
                continue
            data_dict[ticker] = df
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    if data_dict:
        min_len = min(len(df) for df in data_dict.values())
        for k in data_dict:
            data_dict[k] = data_dict[k].iloc[-min_len:]

    return data_dict'''
