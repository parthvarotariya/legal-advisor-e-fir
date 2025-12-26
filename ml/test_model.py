"""
Test the trained DistilBERT model on the test dataset
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
TEST_PATH = os.path.join(BASE_DIR, "datasets", "test.csv")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "datasets", "label_map.json")


# =========================
# Load label map
# =========================
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

num_labels = len(label_map["label_to_name"])
print(f"Number of classes: {num_labels}")


# =========================
# Load test dataset
# =========================
df_test = pd.read_csv(TEST_PATH, header=None, names=["text", "label"])
df_test["text"] = df_test["text"].apply(clean_text)

X_test = df_test["text"].values
y_test = df_test["label"].values

print(f"Test samples: {len(X_test)}")


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
# Load Model (will retrain since we haven't saved it)
# =========================
MODEL_NAME = "distilbert-base-uncased"

print("\nRetraining model for testing...")
print("(Note: Model weights were not saved. Training from scratch.)")

# Load training data
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join(BASE_DIR, "datasets", "crime_data.csv")
df = pd.read_csv(DATASET_PATH, header=None, names=["text", "label"], skiprows=1)
df["text"] = df["text"].apply(clean_text)

texts = df["text"].values
labels = df["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

for param in model.parameters():
    param.requires_grad = True

model.to(device)

# Quick training
BATCH_SIZE = 16
EPOCHS = 4

train_dataset = ComplaintDataset(X_train, y_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )
        
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

print("\n" + "="*60)
print("TESTING ON HELD-OUT TEST SET")
print("="*60)


# =========================
# Test evaluation
# =========================
test_dataset = ComplaintDataset(X_test, y_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        predictions = torch.argmax(outputs.logits, dim=1)
        preds.extend(predictions.cpu().tolist())
        true_labels.extend(batch["labels"].tolist())

test_acc = accuracy_score(true_labels, preds)

print(f"\n🎯 TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Correct predictions: {sum([1 for t, p in zip(true_labels, preds) if t == p])}/{len(true_labels)}")

label_names = [label_map["label_to_name"][str(i)] for i in range(num_labels)]

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT (TEST SET)")
print("="*60)
print(classification_report(true_labels, preds, target_names=label_names, zero_division=0))

# Show misclassifications
print("\n" + "="*60)
print("MISCLASSIFICATIONS")
print("="*60)
for i, (true_label, pred_label) in enumerate(zip(true_labels, preds)):
    if true_label != pred_label:
        true_name = label_map["label_to_name"][str(true_label)]
        pred_name = label_map["label_to_name"][str(pred_label)]
        print(f"\nSample {i+1}:")
        print(f"  Text: {X_test[i][:100]}...")
        print(f"  True: {true_name} (label {true_label})")
        print(f"  Predicted: {pred_name} (label {pred_label})")

if sum([1 for t, p in zip(true_labels, preds) if t != p]) == 0:
    print("\n✅ No misclassifications! Perfect test accuracy!")
