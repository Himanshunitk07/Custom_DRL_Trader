import os
import pandas as pd
from datetime import datetime

# Import all necessary functions from your existing modules
from a_fetch_data import load_sp500_ticker_list, fetch_news_for_ticker
from b_sentimental_model import analyze_sentiment
from d_sentiment_loader import get_price_data
from e_normalize import normalize_sentiment

# --- Configuration ---
START_DATE = '2025-07-20'
END_DATE = '2025-08-02'
MAX_TICKERS = 5 # Set the number of tickers to process
OUTPUT_DIR = "sentiment_dashboard_data"

def generate_all_csvs():
    """
    Orchestrates the entire data processing pipeline and saves the output
    of each major step into a separate CSV file for dashboarding.
    """
    print("Starting CSV generation process...")

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # --- Step 1: Fetch Raw News and Analyze Sentiment for Each Headline ---
    print(f"\n[1/5] Fetching news for {MAX_TICKERS} tickers from {START_DATE} to {END_DATE}...")
    tickers = load_sp500_ticker_list()[:MAX_TICKERS]
    all_headlines = []

    for ticker in tickers:
        print(f"  -> Fetching for {ticker}...")
        news_items = fetch_news_for_ticker(ticker, START_DATE, END_DATE)
        if news_items:
            # Add ticker information to each headline
            for item in news_items:
                item['ticker'] = ticker
            all_headlines.extend(news_items)

    if not all_headlines:
        print("No news found for the selected tickers and date range. Exiting.")
        return

    headlines_df = pd.DataFrame(all_headlines)
    print("  -> Analyzing sentiment for each headline...")
    sentiment_data = headlines_df['headline'].apply(analyze_sentiment)
    sentiment_df = pd.json_normalize(sentiment_data)
    
    # Combine headlines with their sentiment scores
    raw_sentiment_df = pd.concat([headlines_df, sentiment_df], axis=1)
    raw_sentiment_df.to_csv(f"{OUTPUT_DIR}/01_raw_headlines_with_sentiment.csv", index=False)
    print(f"Saved raw headlines and sentiments.")

    # --- Step 2: Aggregate Sentiment by Day and Ticker ---
    print("\n[2/5] Aggregating sentiment scores by day...")
    daily_agg_df = raw_sentiment_df.groupby(['date', 'ticker']).agg(
        score=('score', 'mean'),
        positive=('positive', 'sum'),
        negative=('negative', 'sum'),
        headline_count=('headline', 'count')
    ).reset_index()
    daily_agg_df.to_csv(f"{OUTPUT_DIR}/02_daily_aggregated_sentiment.csv", index=False)
    print("Saved daily aggregated sentiment.")

    # --- Step 3: Normalize the Aggregated Sentiment ---
    print("\n[3/5] Normalizing aggregated sentiment data...")
    normalized_df = normalize_sentiment(daily_agg_df.copy())
    normalized_df.to_csv(f"{OUTPUT_DIR}/03_normalized_daily_sentiment.csv", index=False)
    print("Saved normalized sentiment data.")

    # --- Step 4: Fetch Stock Price Data ---
    print("\n[4/5] Fetching stock price data...")
    price_df = get_price_data(tickers, START_DATE, END_DATE)
    price_df['date'] = pd.to_datetime(price_df['date']).dt.date.astype(str)
    price_df.to_csv(f"{OUTPUT_DIR}/04_stock_price_data.csv", index=False)
    print("Saved stock price data.")

    # --- Step 5: Merge Normalized Sentiment and Price Data ---
    print("\n[5/5] Merging sentiment and price data...")
    # Ensure date columns are of the same type (string) for merging
    normalized_df['date'] = pd.to_datetime(normalized_df['date']).dt.date.astype(str)
    
    final_merged_df = pd.merge(normalized_df, price_df, on=['date', 'ticker'], how='left')
    final_merged_df.to_csv(f"{OUTPUT_DIR}/05_final_merged_data.csv", index=False)
    print("Saved final merged data.")
    
    print("\nProcess complete! All CSV files are in the 'sentiment_dashboard_data' folder.")
    print("\nFinal Merged Data Head:")
    print(final_merged_df[['date', 'ticker', 'score', 'Close', 'Volume']].head())


if __name__ == "__main__":
    # Ensure you have the required packages installed:
    # pip install pandas feedparser transformers torch yfinance scikit-learn
    generate_all_csvs()
