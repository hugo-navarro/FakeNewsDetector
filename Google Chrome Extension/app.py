from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

def load_model(language='English'):
    if language == 'Portuguese':
        model = joblib.load('fake_news_model_pt.pkl')
        vectorizer = joblib.load('tfidf_vectorizer_pt.pkl')
    else:
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

def predict_news(news_text, model, vectorizer):
    # Preprocess input
    text_df = pd.DataFrame({'text': [news_text]})
    # Use the same cleaning as in training
    from data_preprocessing import preprocess_dataframe
    text_df = preprocess_dataframe(text_df)
    X = vectorizer.transform(text_df['text'])
    prediction = model.predict(X)[0]
    return prediction

@app.route('/validate', methods=['POST'])
def validate():
    data = request.get_json()
    language = data.get("language", '')
    news_text = data.get('texto', '')

    answer = 'Please enter some text.'

    if news_text.strip() != '':
        model, vectorizer = load_model(language)
        prediction = predict_news(news_text, model, vectorizer)
        if prediction == 0:
            answer = 'This news is predicted to be FAKE.'
        else:
            answer = 'This news is predicted to be TRUE.'
    return jsonify({'resposta': answer})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
