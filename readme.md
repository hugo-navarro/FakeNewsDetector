# Fake News Detector

This project uses machine learning to classify news articles as fake or true. It provides a simple web interface using Streamlit.

## Requirements

- Python 3.7+
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Training the Model

1. Place your dataset files (`Fake.csv` and `True.csv`) in the project directory.
2. Run the training script for Regression Model:
   ```
   python train_model.py
   ```
   This will train the model and save `fake_news_model.pkl` and `tfidf_vectorizer.pkl` for english, and `fake_news_model_pt.pkl` and `tfidf_vectorizer_pt.pkl` for portuguese the project directory.
3. Run the training script for BERT/Transformer Model:
   ```
   python transformer.py
   ```
   This will train the model and save `bert_finetuned_fake_news` (for english) and `bertimbau_finetuned_fake_news` (for portuguese) in the project directory.

## Running the Web App

1. After training, start the Streamlit server:
   ```
   streamlit run app.py
   ```
2. Open the provided local URL in your browser (usually http://localhost:8501).
3. Paste or type a news article in the text box and click "Classify" to see the prediction.

## Adding the Chrome Extension to your browser

1. Go to the "My Extensions" section on Google Chrome settings.
2. Turn on the "Developer mode".
3. Click on "Load Unpacked".
4. Then select the directory "Google Chrome Extension" from this project.
5. Now you should have the extension on your browser.
6. But for it to work you need to go to the "Google Chrome Extension" directory and run the app.py:
   ```
   python app.py
   ```
8. Now you can open the extension and paste or type a news article in the text box and click "Check" to see the prediction.

## Project Structure

```
FakeNewsDetector/
├── Fake.csv
├── True.csv
├── data_preprocessing.py
├── train_model.py
├── app.py
├── requirements.txt
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
└── README.md
```

## Notes

- You can retrain the model anytime by running `python train_model.py` and `python transformer.py` again.
- Make sure `Fake.csv` and `True.csv` have a `text` column containing the news content.


### Using the Web App

- The Streamlit app now includes a language selector (English/Portuguese).
- Select your desired language before classifying news articles.
- The app will use the appropriate model and preprocessing pipeline for the selected language.

## Updated Project Structure

```
FakeNewsDetector/
├── Fake.csv
├── True.csv
├── FakeTrueBr_corpus.csv
├── data_preprocessing.py
├── train_model.py
├── transformer.py
├── app.py
├── requirements.txt
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── fake_news_model_pt.pkl
├── tfidf_vectorizer_pt.pkl
└── README.md
```

## Notes

- You can retrain the English or Portuguese models anytime by running their respective training scripts.
- For English, ensure `Fake.csv` and `True.csv` have a `text` column.
- For Portuguese, `FakeTrueBr_corpus.csv` should have `fake` and `true` columns containing the news content.
- Please desconsider the archive train_model_pt. This is a legacy code.
