from src.load_data import load_dataset
from src.preprocess import clean_tweet
from src.vectorize import vectorize_text
from sklearn.model_selection import train_test_split
from src.train_model import train_logistic_regression, evaluate_model
from src.predict import predict_sentiment


# Load and clean
df = load_dataset(r'D:\downloads\sentiment analysis\sentiment.csv')
df['clean_text'] = df['text'].apply(clean_tweet)

# Vectorize
X, vectorizer = vectorize_text(df['clean_text'])

# Prepare labels
y = df['sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_logistic_regression(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test)

sample_tweet = "I love how easy this project is turning out!"
predicted = predict_sentiment(sample_tweet, model, vectorizer)
print(f"Tweet: {sample_tweet}\nPredicted Sentiment: {predicted}")
