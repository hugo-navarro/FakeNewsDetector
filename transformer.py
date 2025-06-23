import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup, BertTokenizer, BertConfig
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data_preprocessing import load_and_prepare_data
from collections import Counter
import os

# ======= Configurações por idioma =======
LANG_SETTINGS = {
    'en': {
        'model_name': 'prajjwal1/bert-tiny',
        'output_file': 'bert_finetuned_fake_news.pt',
        'params': {
            'batch_size': 64,
            'learning_rate': 2e-5,
            'weight_decay': 0.1,
            'max_length': 256,
            'epochs': 6,
        }
    },
    'pt': {
        'model_name': 'neuralmind/bert-base-portuguese-cased',
        'output_file': 'bertimbau_finetuned_fake_news.pt',
        'params': {
            'batch_size': 32,
            'learning_rate': 3e-5,
            'weight_decay': 0.1,
            'max_length': 128,
            'epochs': 8,
        }
    }
}

# ======= Treinar para cada linguagem =======
for language in ['en', 'pt']:
    print(f"\n=== Treinando modelo para linguagem: {language.upper()} ===")

    model_name = LANG_SETTINGS[language]['model_name']
    output_file = LANG_SETTINGS[language]['output_file']
    params = LANG_SETTINGS[language]['params']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    df = load_and_prepare_data(
        language=language,
        use_title=False,
        do_augment=False,
        balance_classes=True
    )

    X = df['text'].tolist()
    y = df['label'].tolist()

    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.3, stratify=y_temp, random_state=42)

    def avg_length(texts):
        return np.mean([len(text) for text in texts])

    print("\n=== Verificação Pós-Split ===")
    print("Comprimento médio:")
    print(f"Treino: {avg_length(X_train):.1f} chars")
    print(f"Validação: {avg_length(X_valid):.1f} chars")
    print(f"Teste: {avg_length(X_test):.1f} chars")

    print("Distribuição das classes no treino:", Counter(y_train))
    print("Distribuição das classes no val:", Counter(y_valid))
    print("Distribuição das classes no test:", Counter(y_test))

    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(texts):
        return tokenizer(texts, truncation=True, padding=True, max_length=params['max_length'], return_tensors="pt")

    print("Tokenizando os dados...")
    train_encodings = tokenize(X_train)
    print("Tokenização do treino completa.")
    valid_encodings = tokenize(X_valid)
    print("Tokenização de validação completa.")
    test_encodings = tokenize(X_test)
    print("Tokenização de teste completa.")

    class FakeNewsDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.tensor(labels).long()
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

    train_dataset = FakeNewsDataset(train_encodings, y_train)
    valid_dataset = FakeNewsDataset(valid_encodings, y_valid)
    test_dataset = FakeNewsDataset(test_encodings, y_test)

    from torch.utils.data import WeightedRandomSampler

    class_counts = np.bincount(y_train)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = class_weights[y_train]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        sampler=sampler,
        shuffle=False,
        num_workers=2
    )

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

    config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob=0.6, attention_probs_dropout_prob=0.6, classifier_dropout=0.5, layer_norm_eps=1e-6)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    total_steps = len(train_loader) * params['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        total_loss, correct = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()

        return total_loss / len(train_loader), correct / len(train_dataset)

    def evaluate():
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

                total_loss += loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()

        return total_loss / len(valid_loader), correct / len(valid_dataset)

    print("Iniciando treino...")
    for epoch in range(params['epochs']):
        print(f"Epoch {epoch+1}/{params['epochs']}")
        train_loss, train_acc = train()
        val_loss, val_acc = evaluate()

        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), output_file)
    print(f"Modelo salvo em: {output_file}")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            _, preds = torch.max(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {language.upper()}")
    plt.savefig(f'matrix_{language}.png')
    plt.show()
