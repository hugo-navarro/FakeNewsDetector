import pandas as pd
import string

def load_and_prepare_data(fake_path='Fake.csv', true_path='True.csv'):
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def clean_text(text):
    # Lowercase, remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_dataframe(df, text_column='text'):
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

if __name__ == '__main__':
    df = load_and_prepare_data()
    df = preprocess_dataframe(df)
    print(df.head())
