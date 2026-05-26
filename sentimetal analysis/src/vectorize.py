from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(corpus, max_features=5000):
    """
    Converts cleaned text into TF-IDF vectors.

    Parameters:
        corpus (list or Series): List of cleaned tweets
        max_features (int): Max number of features to keep

    Returns:
        X (sparse matrix): TF-IDF feature matrix
        vectorizer (TfidfVectorizer): Fitted vectorizer object
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer