# Tweet Sentiment Analysis with Logistic Regression

This project builds a modular sentiment analysis pipeline using the Sentiment140 dataset. It processes tweets, vectorizes them with TF-IDF, trains a Logistic Regression model, and predicts sentiment on new inputs.

## Project Structure
sentiment-analysis/ │ ├── data/                  # CSV dataset (sentiment.csv) ├── src/ │   ├── load_data.py       # Load and format dataset │   ├── preprocess.py      # Clean tweets using NLTK │   ├── vectorize.py       # TF-IDF vectorization │   ├── train_model.py     # Train & evaluate model │   └── predict.py         # Predict sentiment on new tweets ├── main.py                # Pipeline entry point ├── requirements.txt       # Dependencies └── README.md              # Project overview

## How to Run (Windows)

1. **Navigate to project folder**  
   `cd D:\downloads\sentiment analysis`

2. **Create and activate virtual environment**  
   ```bash
   python -m venv venv
   venv\Scripts\activate

- Install dependencies
pip install -r requirements.txt
- Run the pipeline
python main.py

##SAMPLE OUTPUT
Tweet: I love how easy this project is turning out!
Predicted Sentiment: positive

Model Performance- Classifier: Logistic Regression
- Vectorizer: TF-IDF (max_features=5000)
- Accuracy: ~85% on test set

 Tools Used- Python, pandas, scikit-learn, NLTK
- VS Code on Windows
- Sentiment140 dataset

