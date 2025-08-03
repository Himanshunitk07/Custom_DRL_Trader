# pip install feedparser

import pandas as pd
import feedparser
from datetime import datetime
from datetime import datetime

def load_sp500_ticker_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df["Symbol"].tolist()

def fetch_news_for_ticker(ticker, start_date, end_date):

    # Convert to date objects
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    feed_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(feed_url)

    news = []
    for entry in feed.entries:
        pub_date = datetime(*entry.published_parsed[:6]).date()
        if start_date <= pub_date <= end_date:
            news.append({
                "date": pub_date.isoformat(),
                "headline": entry.title
            })
    return news
