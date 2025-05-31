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
2. Run the training script:
   ```
   python train_model.py
   ```
   This will train the model and save `fake_news_model.pkl` and `tfidf_vectorizer.pkl` in the project directory.

## Running the Web App

1. After training, start the Streamlit server:
   ```
   streamlit run app.py
   ```
2. Open the provided local URL in your browser (usually http://localhost:8501).
3. Paste or type a news article in the text box and click "Classify" to see the prediction.

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

- You can retrain the model anytime by running `python train_model.py` again.
- Make sure `Fake.csv` and `True.csv` have a `text` column containing the news content.