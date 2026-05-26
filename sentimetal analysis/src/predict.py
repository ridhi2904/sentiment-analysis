from src.preprocess import clean_tweet

def predict_sentiment(tweet, model, vectorizer):
    """
    Predicts sentiment of a single tweet.

    Parameters:
        tweet (str): Raw tweet text
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer

    Returns:
        str: Predicted sentiment label
    """
    cleaned = clean_tweet(tweet)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]