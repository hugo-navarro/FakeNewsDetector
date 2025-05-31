import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessing import load_and_prepare_data, preprocess_dataframe

def train_and_save_model():
    # Load and preprocess data
    df = load_and_prepare_data()
    df = preprocess_dataframe(df)

    X = df['text']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_tfidf)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(clf, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print('Model and vectorizer saved.')

if __name__ == '__main__':
    train_and_save_model()
