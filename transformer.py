import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_preprocessing import load_and_prepare_data, preprocess_dataframe

# ======= Configurações =======
params = {
    'bert_version': "bert-base-uncased",
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'max_length': 128,
    'epochs': 3,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ======= Carregar e preparar os dados =======
df = load_and_prepare_data()
df = preprocess_dataframe(df)

X = df['text'].tolist()
y = df['label'].tolist()

# Split simples usando sklearn
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ======= Tokenização =======
tokenizer = BertTokenizerFast.from_pretrained(params['bert_version'])

def tokenize(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=params['max_length'], return_tensors="pt")

train_encodings = tokenize(X_train)
valid_encodings = tokenize(X_valid)
test_encodings = tokenize(X_test)

# ======= Dataset Customizado =======
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

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

# ======= Modelo BERT para Classificação =======
model = BertForSequenceClassification.from_pretrained(params['bert_version'], num_labels=2)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

total_steps = len(train_loader) * params['epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = torch.nn.CrossEntropyLoss()

# ======= Funções de treino e validação =======
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

# ======= Treinamento =======
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(params['epochs']):
    train_loss, train_acc = train()
    val_loss, val_acc = evaluate()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{params['epochs']}")
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Valid Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# ======= Teste final =======
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ======= Relatório Final =======
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

# Salvar o modelo BERT fine-tuned
torch.save(model.state_dict(), 'bert_finetuned_fake_news.pt')


# Matriz de Confusão
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
