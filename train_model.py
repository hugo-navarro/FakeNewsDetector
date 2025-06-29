import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
    print(f"\nTreinando modelo de regressão logística com validação cruzada para: {language.upper()}")

    # ====== Carregamento e preprocessamento ======
    df = load_and_prepare_data(language=language, use_title=LANG_SETTINGS[language]['use_title'], balance_classes=True)
    X = df['text']
    y = df['label']

    # ====== Vetorização ======
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)

    # ====== Modelo ======
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    # ====== Validação cruzada (Stratified K-Fold) ======
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_vec, y, cv=skf)

    # ====== Avaliação ======
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Fake', 'Real']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # ====== Matriz de confusão relativa ======
    conf_matrix = confusion_matrix(y, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='.4f', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {language.upper()} (Normalized)")
    plt.savefig(f'matrix_logreg_{language}_relative.png')
    plt.show()

    # ====== Treinamento final com todos os dados ======
    model.fit(X_vec, y)

    # ====== Salvando ======
    joblib.dump(model, LANG_SETTINGS[language]['model_out'])
    joblib.dump(vectorizer, LANG_SETTINGS[language]['vectorizer_out'])
    print(f"Modelos salvos para {language}: {LANG_SETTINGS[language]['model_out']}, {LANG_SETTINGS[language]['vectorizer_out']}")
