import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from data_preprocessing import load_and_prepare_data

# ====== Configurações por idioma ======
LANG_SETTINGS = {
    'en': {
        'use_title': False,
        'model_out': 'fake_news_model.pkl',
        'vectorizer_out': 'tfidf_vectorizer.pkl'
    },
    'pt': {
        'use_title': True,
        'model_out': 'fake_news_model_pt.pkl',
        'vectorizer_out': 'tfidf_vectorizer_pt.pkl'
    }
}

for language in ['en', 'pt']:
    print(f"\nTreinando modelo de regressão logística para: {language.upper()}")

    # ====== Carregamento e preprocessamento ======
    df = load_and_prepare_data(language=language, use_title=LANG_SETTINGS[language]['use_title'], balance_classes=True)
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # ====== Vetorização ======
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ====== Modelo ======
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    # ====== Avaliação ======
    y_pred = model.predict(X_test_vec)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ====== Salvando ======
    joblib.dump(model, LANG_SETTINGS[language]['model_out'])
    joblib.dump(vectorizer, LANG_SETTINGS[language]['vectorizer_out'])
    print(f"Modelos salvos para {language}: {LANG_SETTINGS[language]['model_out']}, {LANG_SETTINGS[language]['vectorizer_out']}")
