def normalize_sentiment(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['score', 'positive', 'negative']] = scaler.fit_transform(df[['score', 'positive', 'negative']])
    return df