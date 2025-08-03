import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI in Finance – Smart Portfolio Optimization",
    page_icon="P",
    layout="wide",
)

# --- Data Loading ---
@st.cache_data
def load_all_data():
    """
    Loads all 8 CSV files into pandas DataFrames.
    Uses caching to improve performance.
    """
    data_paths = {
        # Agent Data
        'training_summary': 'dashboard_data/training_summary.csv',
        'test_summary': 'dashboard_data/test_summary.csv',
        'trade_log': 'dashboard_data/trade_log.csv',
        # Sentiment Data
        'raw_headlines': 'sentiment_dashboard_data/01_raw_headlines_with_sentiment.csv',
        'agg_sentiment': 'sentiment_dashboard_data/02_daily_aggregated_sentiment.csv',
        'norm_sentiment': 'sentiment_dashboard_data/03_normalized_daily_sentiment.csv',
        'price_data': 'sentiment_dashboard_data/04_stock_price_data.csv',
        'merged_data': 'sentiment_dashboard_data/05_final_merged_data.csv',
    }
    
    dataframes = {}
    for name, path in data_paths.items():
        if os.path.exists(path):
            dataframes[name] = pd.read_csv(path)
        else:
            st.error(f"File not found: {path}. Please make sure all CSV files are in the correct directories.")
            return None
            
    # Data type conversions for plotting
    if 'merged_data' in dataframes:
        dataframes['merged_data']['date'] = pd.to_datetime(dataframes['merged_data']['date'])
    if 'trade_log' in dataframes:
         dataframes['trade_log']['timestamp'] = pd.to_datetime(dataframes['trade_log']['timestamp'])

    return dataframes

data = load_all_data()

if data is None:
    st.stop()

# --- Sidebar ---
st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio("Go to", ["Introduction", "Sentiment & Price Analysis", "DRL Agent Performance", "Sentiment Signal Analysis", "Trade Log Simulation"])

st.sidebar.markdown("---")
st.sidebar.header("Filters")

available_tickers = sorted(data['merged_data']['ticker'].unique())
selected_tickers = st.sidebar.multiselect(
    "Select Ticker(s)",
    options=available_tickers,
    default=available_tickers[:1] if available_tickers else None
)

if selected_tickers:
    merged_df_filtered = data['merged_data'][data['merged_data']['ticker'].isin(selected_tickers)]
else:
    merged_df_filtered = data['merged_data']

st.title("AI in Finance – Smart Portfolio Optimization using DRL")


# ==============================================================================
# PAGE 1: INTRODUCTION
# ==============================================================================
if page == "Introduction":
    st.header("Project Overview")
    st.markdown("""
    This project applies **Deep Reinforcement Learning (DRL)** to the world of finance for intelligent portfolio optimization and stock trading. 
    It integrates market price signals with financial news sentiment to train agents that can learn profitable strategies over time.
    """)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Project Goals")
        st.markdown("""
        - **Build** a simulated stock market environment using OpenAI Gym.
        - **Train** a DRL agent (e.g., PPO, DDPG) to learn portfolio strategies.
        - **Enhance** decision-making with financial news sentiment using FinBERT.
        - **Visualize** training performance and portfolio metrics using this Streamlit dashboard.
        """)
        st.subheader("Core Models & Technology")
        st.markdown("""
        - **DRL Algorithms**: Stable-Baselines3 (PPO, DDPG)
        - **Sentiment Analysis**: FinBERT
        - **Data Source**: yfinance
        - **Dashboard**: Streamlit & Plotly
        - **Core Libraries**: Python, Pandas, PyTorch
        """)
    with col2:
        st.subheader("Key Resources")
        st.markdown("""
        - **FinRL – DRL in Finance**: A comprehensive library for financial reinforcement learning.
        - **FinBERT Paper**: The research behind the sentiment model used.
        - **Streamlit Docs**: Documentation for the dashboarding framework.
        """)
        st.subheader("Team & Contributions")
        with st.expander("Click to see team member contributions"):
            st.markdown("""
            - **Praneet**: DRL agent implementation (PPO/DDPG)
            - **Chinmay**: Environment design & reward function
            - **Naman**: Sentiment module using FinBERT
            - **Himanshu**: Streamlit dashboard & integration
            """)

# ==============================================================================
# PAGE 2: SENTIMENT & PRICE ANALYSIS
# ==============================================================================
elif page == "Sentiment & Price Analysis":
    st.header("Sentiment & Price Correlation Analysis")
    col1, col2, col3, col4 = st.columns(4)
    final_net_worth = data['training_summary']['final_net_worth'].iloc[-1]
    avg_test_return = data['test_summary']['portfolio_return'].mean()
    win_rate = (data['test_summary']['portfolio_return'] > 0).mean() * 100
    col1.metric("Final Training Net Worth", f"${final_net_worth:,.2f}")
    col2.metric("Avg. Test Return", f"{avg_test_return:.2f}%")
    col3.metric("Test Win Rate", f"{win_rate:.1f}%")
    col4.metric("Tickers Analyzed", len(available_tickers))
    st.markdown("---")
    st.subheader("Normalized Sentiment vs. Stock Price")
    if not selected_tickers:
        st.warning("Please select at least one ticker from the sidebar.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for ticker in selected_tickers:
            ticker_df = merged_df_filtered[merged_df_filtered['ticker'] == ticker]
            fig.add_trace(go.Scatter(x=ticker_df['date'], y=ticker_df['Close'], name=f"{ticker} Price"), secondary_y=False)
            fig.add_trace(go.Scatter(x=ticker_df['date'], y=ticker_df['score'], name=f"{ticker} Sentiment", line=dict(dash='dot')), secondary_y=True)
        fig.update_layout(title_text="Stock Price and Sentiment Score Over Time", xaxis_title="Date", legend_title="Legend")
        fig.update_yaxes(title_text="<b>Stock Price ($)</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Normalized Sentiment Score</b>", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 3: DRL AGENT PERFORMANCE
# ==============================================================================
elif page == "DRL Agent Performance":
    st.header("Deep Reinforcement Learning Agent Performance")
    tab1, tab2 = st.tabs(["Training Performance", "Testing Performance"])
    with tab1:
        st.subheader("Agent Training Results")
        col1, col2 = st.columns(2)
        fig1 = go.Figure(go.Scatter(x=data['training_summary'].index, y=data['training_summary']['final_net_worth'], mode='lines', name='Net Worth'))
        fig1.update_layout(title="Net Worth per Training Episode", xaxis_title="Episode", yaxis_title="Final Net Worth ($)")
        col1.plotly_chart(fig1, use_container_width=True)
        fig2 = go.Figure(go.Scatter(x=data['training_summary'].index, y=data['training_summary']['portfolio_return_pct'], mode='lines', name='Return', line=dict(color='green')))
        fig2.update_layout(title="Portfolio Return per Training Episode", xaxis_title="Episode", yaxis_title="Return (%)")
        col2.plotly_chart(fig2, use_container_width=True)
    with tab2:
        st.subheader("Agent Testing Summary")
        fig3 = go.Figure(go.Bar(x=data['test_summary']['episode'], y=data['test_summary']['portfolio_return'], name='Test Return'))
        fig3.update_layout(title="Portfolio Return by Test Episode", xaxis_title="Test Episode", yaxis_title="Return (%)")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(data['test_summary'])

# ==============================================================================
# PAGE 4: SENTIMENT SIGNAL ANALYSIS
# ==============================================================================
elif page == "Sentiment Signal Analysis":
    st.header("Sentiment Signal Deep Dive")
    if not selected_tickers:
        st.warning("Please select a ticker from the sidebar.")
    else:
        ticker = selected_tickers[0]
        st.info(f"Showing detailed sentiment for: **{ticker}**")
        agg_sentiment_df = data['agg_sentiment'][data['agg_sentiment']['ticker'] == ticker]
        if not agg_sentiment_df.empty:
            selected_date = st.select_slider(
                "Select a Date to Inspect Headlines",
                options=sorted(agg_sentiment_df['date'].unique()),
                value=agg_sentiment_df['date'].unique()[0]
            )
            daily_sentiment = agg_sentiment_df[agg_sentiment_df['date'] == selected_date]
            col1, col2 = st.columns(2)
            score = daily_sentiment['score'].iloc[0] if not daily_sentiment.empty else 0
            count = daily_sentiment['headline_count'].iloc[0] if not daily_sentiment.empty else 0
            col1.metric(f"Aggregated Score on {selected_date}", f"{score:.3f}")
            col2.metric("Headlines Found", count)
            st.subheader(f"Headlines for {ticker} on {selected_date}")
            headlines_on_date = data['raw_headlines'][(data['raw_headlines']['date'] == selected_date) & (data['raw_headlines']['ticker'] == ticker)]
            st.dataframe(headlines_on_date[['headline', 'score', 'positive', 'negative']])
        else:
            st.warning(f"No sentiment data available for ticker: {ticker}")

# ==============================================================================
# PAGE 5: TRADE LOG SIMULATION
# ==============================================================================
elif page == "Trade Log Simulation":
    st.header("Agent's Trading Behavior Simulation")
    trade_log_df = data['trade_log']
    max_steps = len(trade_log_df) - 1
    if max_steps < 0:
        st.warning("The trade log is empty. Cannot display simulation.")
    else:
        step = st.slider("Scrub Through Trading Day", 0, max_steps, 0)
        current_step_data = trade_log_df.iloc[step]
        col1, col2, col3 = st.columns(3)
        timestamp_val = current_step_data['timestamp']
        display_time = "N/A" if pd.isna(timestamp_val) else str(timestamp_val.time())
        col1.metric("Timestamp", display_time)
        col2.metric("Net Worth", f"${current_step_data['net_worth']:,.2f}")
        col3.metric("Cash Balance", f"${current_step_data['cash_balance']:,.2f}")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Portfolio Composition")
            asset_value_cols = [col for col in trade_log_df.columns if 'value_' in col]
            asset_values = current_step_data[asset_value_cols]
            asset_labels = [col.replace('value_', '') for col in asset_value_cols]
            portfolio_labels = asset_labels + ['Cash']
            portfolio_values = list(asset_values) + [current_step_data['cash_balance']]
            fig_pie = go.Figure(data=[go.Pie(labels=portfolio_labels, values=portfolio_values, hole=.3)])
            fig_pie.update_layout(title_text="Current Allocation")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.subheader("Agent's Action Signal")
            action_cols = [col for col in trade_log_df.columns if 'action_' in col]
            actions = current_step_data[action_cols]
            action_labels = [col.replace('action_', '') for col in action_cols]
            fig_bar = go.Figure(data=[go.Bar(x=action_labels, y=actions)])
            fig_bar.update_layout(title_text="Allocation Signals (-1: Sell, 1: Buy)", yaxis_title="Action Signal")
            st.plotly_chart(fig_bar, use_container_width=True)
        st.subheader("Net Worth Trajectory During Simulation")
        fig_line = go.Figure(go.Scatter(x=trade_log_df['step'], y=trade_log_df['net_worth'], mode='lines'))
        fig_line.add_vline(x=step, line_width=2, line_dash="dash", line_color="red")
        fig_line.update_layout(xaxis_title="Step", yaxis_title="Net Worth ($)")
        st.plotly_chart(fig_line, use_container_width=True)
