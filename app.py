import streamlit as st
import joblib
import pandas as pd

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

def main():
    st.title('Fake News Detector')
    st.write('Enter a news article below to check if it is fake or true.')

    language = st.selectbox('Select language', ['English', 'Portuguese'])
    news_text = st.text_area('News Article Text', height=200)
    if st.button('Classify'):
        if news_text.strip() == '':
            st.warning('Please enter some text.')
        else:
            model, vectorizer = load_model(language)
            prediction = predict_news(news_text, model, vectorizer)
            if prediction == 0:
                st.error('This news is predicted to be FAKE.')
            else:
                st.success('This news is predicted to be TRUE.')

if __name__ == '__main__':
    main()
