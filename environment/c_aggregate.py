import pandas as pd
from b_sentimental_model import analyze_sentiment_batch

def aggregate_sentiments(news_list, workers=8):
    df = pd.DataFrame(news_list)
    df['sentiment'] = analyze_sentiment_batch(df['headline'].tolist(), max_workers=workers)
    df['score'] = [s['score'] for s in df['sentiment']]
    df['positive'] = [s['positive'] for s in df['sentiment']]
    df['negative'] = [s['negative'] for s in df['sentiment']]
    grouped = df.groupby('date').agg({
        'score': 'mean',
        'positive': 'sum',
        'negative': 'sum',
    }).reset_index()

    return grouped
