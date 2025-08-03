# aggregate.py
import pandas as pd
from b_sentimental_model import analyze_sentiment 

def aggregate_sentiments(news_list):
    # news_list = [{'date': ..., 'headline': ...}, ...]
    df = pd.DataFrame(news_list)
    df['sentiment'] = df['headline'].apply(analyze_sentiment)
    
    # Extract individual scores
    df['score'] = df['sentiment'].apply(lambda x: x['score'])
    df['positive'] = df['sentiment'].apply(lambda x: x['positive'])
    df['negative'] = df['sentiment'].apply(lambda x: x['negative'])

    # Group by date
    grouped = df.groupby('date').agg({
        'score': 'mean',
        'positive': 'sum',
        'negative': 'sum',
    }).reset_index()

    return grouped
