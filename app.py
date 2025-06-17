import streamlit as st
import joblib
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Define dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Função para carregar modelo de regressão ----------
def load_logistic_model(language='English'):
    if language == 'Portuguese':
        model = joblib.load('fake_news_model_pt.pkl')
        vectorizer = joblib.load('tfidf_vectorizer_pt.pkl')
    else:
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# ---------- Função para predição usando regressão ----------
def predict_with_logistic(news_text, model, vectorizer):
    text_df = pd.DataFrame({'text': [news_text]})
    from data_preprocessing import preprocess_dataframe
    text_df = preprocess_dataframe(text_df)
    X = vectorizer.transform(text_df['text'])
    prediction = model.predict(X)[0]
    return prediction

# ---------- Função para carregar modelo BERT ----------
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('bert_finetuned_fake_news.pt', map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

# ---------- Função para predição usando BERT ----------
def predict_with_bert(news_text, model, tokenizer):
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# ---------- Interface Streamlit ----------
def main():
    st.title('Fake News Detector')
    st.write('Classify news articles as Fake or Real using Logistic Regression or BERT.')

    language = st.selectbox('Select Language', ['English', 'Portuguese'])
    model_choice = st.selectbox('Select Model', ['Logistic Regression', 'BERT'])

    news_text = st.text_area('News Article Text', height=200)

    if st.button('Classify'):
        if news_text.strip() == '':
            st.warning('Please enter some text.')
        else:
            if model_choice == 'Logistic Regression':
                model, vectorizer = load_logistic_model(language)
                prediction = predict_with_logistic(news_text, model, vectorizer)
            else:
                model, tokenizer = load_bert_model()
                prediction = predict_with_bert(news_text, model, tokenizer)

            if prediction == 0:
                st.error('This news is predicted to be FAKE.')
            else:
                st.success('This news is predicted to be TRUE.')

if __name__ == '__main__':
    main()
