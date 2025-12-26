"""
Fine-tuned DistilBERT
Legal Complaint Classification (WITH fine-tuning)
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.preprocess import clean_text


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "crime_data.csv")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "datasets", "label_map.json")


# =========================
# Load label map
# =========================
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

num_labels = len(label_map["label_to_name"])
print(f"Number of classes: {num_labels}")


# =========================
# Load dataset
# =========================
# Read CSV by splitting from the right to handle commas in text
texts_list = []
labels_list = []
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().rsplit(',', 1)
        if len(parts) == 2:
            texts_list.append(clean_text(parts[0]))
            labels_list.append(int(parts[1]))

texts = pd.Series(texts_list).values
labels = pd.Series(labels_list).values

print(f"Loaded {len(texts)} samples")


# =========================
# Train / Validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# =========================
# Dataset class
# =========================
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =========================
# Tokenizer & Model
# =========================
MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# 🔓 UNFREEZE DISTILBERT (FINE-TUNING)
for param in model.parameters():
    param.requires_grad = True

model.to(device)


# =========================
# DataLoaders
# =========================
BATCH_SIZE = 16

train_dataset = ComplaintDataset(X_train, y_train, tokenizer)
val_dataset = ComplaintDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =========================
# Optimizer & Loss
# =========================
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,              # ✅ Correct LR for fine-tuning
    weight_decay=0.01
)

loss_fn = torch.nn.CrossEntropyLoss()


# =========================
# Training loop
# =========================
EPOCHS = 4

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().tolist())
            true_labels.extend(batch["labels"].tolist())

    acc = accuracy_score(true_labels, preds)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print("-" * 40)


# =========================
# Final evaluation
# =========================
label_names = [label_map["label_to_name"][str(i)] for i in range(num_labels)]

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=label_names))

print("\nTraining completed (WITH fine-tuning DistilBERT).")


# =========================
# Save model and tokenizer
# =========================
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_model")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print(f"\nSaving model to {MODEL_SAVE_PATH}...")
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("✅ Model and tokenizer saved successfully!")
