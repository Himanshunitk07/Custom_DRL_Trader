# trading_env.py

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
        self.indicators = indicators or ["Open", "Close", "rsi", "macd", "obv", "bb_upper", "bb_lower", "minute_norm"]

    def extract(self, df, current_step):
        window = df.iloc[current_step: current_step + self.window_size]
        return window[self.indicators].to_numpy()

class ShortTermDynamicTrader(gym.Env):
    def __init__(
        self,
        data_dict,
        window_size=5,
        initial_balance=100000,
        transaction_cost_pct=0.0001,
        slippage_pct=0.0005,
        max_alloc_per_asset=0.2,
        min_price_filter=1.0
    ):
        super().__init__()
        self.data = data_dict
        self.asset_names = list(data_dict.keys())
        self.n_assets = len(self.asset_names)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_alloc_per_asset = max_alloc_per_asset
        self.min_price_filter = min_price_filter

        self.extractor = IntradayFeatureExtractor(window_size=window_size)
        self.features = self.extractor.indicators
        self.n_features = len(self.features)
        obs_len = self.n_assets * self.n_features * self.window_size + self.n_assets + 1

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

        self.seed()
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
       # Inside ShortTermDynamicTrader.reset()
        max_start = min(len(df_) for df_ in self.data.values()) - self.window_size - 1
    
        self.current_step = self.np_random.integers(0, max_start)
        self.balance = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.position = np.zeros(self.n_assets)
        self.trades = []
        self.net_worths = [self.initial_balance]
        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        assert len(action) == self.n_assets
        action = np.clip(action, -1, 1)

        current_prices = []
        for asset in self.asset_names:
            df = self.data[asset]
            close = df.iloc[self.current_step]["Close"]
            open_price = df.iloc[self.current_step]["Open"]
            if isinstance(close, pd.Series):
                close = close.item()
            if isinstance(open_price, pd.Series):
                open_price = open_price.item()
            price = (float(close) + float(open_price)) / 2
            current_prices.append(price)
        current_prices = np.array(current_prices)

        for i in range(self.n_assets):
            if i >= len(current_prices):
                continue
            price = current_prices[i]
            if price < self.min_price_filter:
                continue

            slippage = price * self.slippage_pct
            scaled_action = (action[i] + 1) / 2
            alloc_cash = scaled_action * self.balance * self.max_alloc_per_asset
            units = int(alloc_cash / (price + slippage))

            max_affordable_units = int(self.balance // (price + slippage))
            units = min(units, max_affordable_units)

            if action[i] > 0 and units >= 1:
                trade_cost = units * (price + slippage)
                fee = trade_cost * self.transaction_cost_pct
                total_cost = trade_cost + fee
                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.position[i] += units
                    self.trades.append((self.current_step, i, "buy", price + slippage, units, fee))

            elif action[i] < 0:
                sell_units = int(-action[i] * self.position[i])
                if sell_units >= 1:
                    revenue = sell_units * (price - slippage)
                    fee = revenue * self.transaction_cost_pct
                    total_gain = revenue - fee
                    self.balance += total_gain
                    self.position[i] -= sell_units
                    self.trades.append((self.current_step, i, "sell", price - slippage, sell_units, fee))

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
        truncated = False

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
        windows = []
        for asset in self.asset_names:
            df = self.data[asset]
            if self.current_step + self.window_size > len(df):
                raise ValueError(f"Not enough data to extract window for {asset} at step {self.current_step}")
            window = self.extractor.extract(df, self.current_step)
            windows.append(window)

        obs_tensor = np.stack(windows, axis=1)
        obs_flat = obs_tensor.flatten()
        agent_state = np.concatenate((self.position.flatten(), np.array([float(self.balance)])))
        return np.concatenate([obs_flat, agent_state])

    def render(self):
        current_prices = []
        for asset in self.asset_names:
            df = self.data[asset]
            close = df.iloc[self.current_step]["Close"]
            open_price = df.iloc[self.current_step]["Open"]
            if isinstance(close, pd.Series):
                close = close.item()
            if isinstance(open_price, pd.Series):
                open_price = open_price.item()
            price = (float(close) + float(open_price)) / 2
            current_prices.append(price)

        net_worth = self.balance + np.sum(self.position * np.array(current_prices))
        print(f"\nStep: {self.current_step}")
        print(f"Balance: ${float(np.squeeze(self.balance)):.2f}")
        print(f"Net Worth: ${float(net_worth):.2f}")
        for i, asset in enumerate(self.asset_names):
            print(f"{asset}: {self.position[i]} units at ${current_prices[i]:.2f}")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        pass


def add_indicators(df):
    df = df.copy()
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    df["rsi"] = RSIIndicator(close=close, window=14).rsi()
    df["macd"] = MACD(close=close).macd_diff()
    df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["minute_norm"] = (df["minute_of_day"] - 570) / (960 - 570)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def fetch_intraday(tickers, interval="5m", period="60d", min_rows=200):
    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
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

    if not data_dict:
        print("No valid data fetched. Returning empty dict.")
        return {}

    all_indices = [set(df.index) for df in data_dict.values()]
    common_index = sorted(list(set.intersection(*all_indices)))
    if len(common_index) < min_rows:
        print("Not enough common timestamps across tickers.")
        return {}

    common_index = common_index[-min_rows:]
    for k in data_dict:
        data_dict[k] = data_dict[k].loc[common_index]

    return data_dict
