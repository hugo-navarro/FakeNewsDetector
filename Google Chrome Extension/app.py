from flask import Flask, request, jsonify
import joblib
import pandas as pd
from data_preprocessing import clean_text, download_nltk_resources

# Inicializa app Flask
app = Flask(__name__)

# Baixa recursos necessários do NLTK
download_nltk_resources()

# Carrega modelos e vetorizadores
models = {
    'English': {
        'model': joblib.load('fake_news_model.pkl'),
        'vectorizer': joblib.load('tfidf_vectorizer.pkl'),
        'lang_code': 'en'
    },
    'Portuguese': {
        'model': joblib.load('fake_news_model_pt.pkl'),
        'vectorizer': joblib.load('tfidf_vectorizer_pt.pkl'),
        'lang_code': 'pt'
    }
}

@app.route('/validate', methods=['POST'])
def validate():
    data = request.get_json()
    text = data.get('texto', '')
    language = data.get('language', 'English')

    if not text or len(text.strip()) < 20:
        return jsonify({'resposta': 'Please insert a longer news text.'})

    if language not in models:
        return jsonify({'resposta': 'Unsupported language selected.'})

    # Limpeza e vetorização
    lang_info = models[language]
    cleaned_text = clean_text(text, language=lang_info['lang_code'])
    vec_text = lang_info['vectorizer'].transform([cleaned_text])

    # Predição
    prediction = lang_info['model'].predict(vec_text)[0]
    resposta = 'This news is predicted to be TRUE.' if prediction == 1 else 'This news is predicted to be FAKE.'

    return jsonify({'resposta': resposta})

if __name__ == '__main__':
    app.run(debug=True)
