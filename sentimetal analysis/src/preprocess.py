import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_tweet(text):
    # Remove URLs, mentions, hashtags, punctuation
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", text.lower())
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)