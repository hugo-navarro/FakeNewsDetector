import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessing import preprocess_dataframe

PT_DATA_PATH = 'FakeTrueBr_corpus.csv'
MODEL_PATH = 'fake_news_model_pt.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer_pt.pkl'

def load_and_prepare_pt_data():
    df = pd.read_csv(PT_DATA_PATH)
    # Extract fake and true news texts
    fake_texts = df[['fake']].copy()
    fake_texts['label'] = 0
    fake_texts = fake_texts.rename(columns={'fake': 'text'})

    true_texts = df[['true']].copy()
    true_texts['label'] = 1
    true_texts = true_texts.rename(columns={'true': 'text'})

    # Combine and shuffle
    all_texts = pd.concat([fake_texts, true_texts], ignore_index=True)
    all_texts = all_texts.sample(frac=1, random_state=42).reset_index(drop=True)
    return all_texts

def train_and_save_pt_model():
    df = load_and_prepare_pt_data()
    df = preprocess_dataframe(df)
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words=None, max_df=0.7)  # No stopwords for PT by default
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print('Portuguese model and vectorizer saved.')

if __name__ == '__main__':
    train_and_save_pt_model()
