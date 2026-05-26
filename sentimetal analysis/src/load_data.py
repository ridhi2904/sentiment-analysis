import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path, encoding='latin-1', header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = df[['sentiment', 'text']]
    df['sentiment'] = df['sentiment'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
    return df