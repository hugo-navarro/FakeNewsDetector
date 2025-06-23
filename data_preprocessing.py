import pandas as pd
import string
import nltk
import nlpaug.augmenter.word as naw
from nltk.corpus import stopwords
from sklearn.utils import resample

def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text, language='en'):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())  # Remove espaços múltiplos
    text = text.replace('\n', ' ').replace('\r', ' ')  # Remove quebras de linha
    
    if language == 'pt':
        stops = set(stopwords.words('portuguese'))
    else:
        stops = set(stopwords.words('english'))
    
    text = ' '.join([word for word in text.split() if word not in stops])
    return text

def load_and_prepare_data(language='en', fake_path='Fake.csv', true_path='True.csv', 
                         use_title=True, do_augment=False, augment_frac=0.1, balance_classes=False):
    
    download_nltk_resources()
    print("\n=== Carregando dados ===")

    if language == 'pt':
        # Carrega dataset em português
        df_pt = pd.read_csv('FakeTrueBr_corpus.csv')
        
        # Prepara os dados no formato esperado
        fake_df = pd.DataFrame({
            'text': df_pt['title_fake'] + " " + df_pt['fake'],
            'label': 0
        })
        true_df = pd.DataFrame({
            'text': df_pt['true'],
            'label': 1
        })
    else:
        # Dataset em inglês padrão
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        fake_df['label'] = 0
        true_df['label'] = 1

        if use_title:
            fake_df['text'] = fake_df['title'].astype(str) + " " + fake_df['text'].astype(str)
            true_df['text'] = true_df['title'].astype(str) + " " + true_df['text'].astype(str)
            print("→ Usando título + texto.")
        else:
            fake_df['text'] = fake_df['text'].astype(str)
            true_df['text'] = true_df['text'].astype(str)
            print("→ Usando apenas o texto (sem título).")

    df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)
    df['text'] = df['text'].apply(lambda x: clean_text(x, language))

    # Remover duplicatas
    before = len(df)
    df = df.drop_duplicates(subset='text')
    after = len(df)
    print(f"→ Removidas {before - after} duplicatas ({before} → {after}).")

    # Restante do código mantido igual
    print("Distribuição antes do augmentation:\n", df['label'].value_counts())

    if do_augment:
        print(f"\n=== Aplicando Data Augmentation em {augment_frac*100:.1f}% das amostras ===")
        aug = naw.SynonymAug(aug_src='wordnet')
        samples_per_class = int(len(df) * augment_frac / 2)

        augmented_rows = []
        for label in [0, 1]:
            class_df = df[df['label'] == label].sample(n=samples_per_class, random_state=42)
            for _, row in class_df.iterrows():
                try:
                    new_text = aug.augment(row['text'])
                    augmented_rows.append({'text': clean_text(new_text, language), 'label': label})
                except:
                    pass

        aug_df = pd.DataFrame(augmented_rows)
        print("→ Total de amostras aumentadas:", len(aug_df))
        df = pd.concat([df, aug_df], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if balance_classes:
        df_fake = df[df['label'] == 0]
        df_real = df[df['label'] == 1]
        min_samples = min(len(df_fake), len(df_real))
        df_fake = resample(df_fake, n_samples=min_samples, random_state=42)
        df_real = resample(df_real, n_samples=min_samples, random_state=42)
        df = pd.concat([df_fake, df_real])
        print("\nDataset balanceado:")
        print(df['label'].value_counts())

    return df

if __name__ == '__main__':
    # Exemplo de uso para português
    df_pt = load_and_prepare_data(
        language='pt',
        use_title=True,
        balance_classes=True
    )
    print("\nExemplo de amostras PT:", df_pt.head())

    # Exemplo de uso para inglês
    df_en = load_and_prepare_data(
        language='en',
        use_title=False,
        balance_classes=True
    )
    print("\nExemplo de amostras EN:", df_en.head())