"""
Evaluate the saved trained model on test dataset (NO RETRAINING)
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
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_model")


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
# Read CSV by splitting from the right to handle commas in text
texts_list = []
labels_list = []
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().rsplit(',', 1)
        if len(parts) == 2:
            texts_list.append(clean_text(parts[0]))
            labels_list.append(int(parts[1]))

X_test = pd.Series(texts_list).values
y_test = pd.Series(labels_list).values

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
# Load saved model
# =========================
if not os.path.exists(MODEL_SAVE_PATH):
    print(f"\n❌ ERROR: Model not found at {MODEL_SAVE_PATH}")
    print("Please run train_distilbert.py first to train and save the model.")
    exit(1)

print(f"\nLoading saved model from {MODEL_SAVE_PATH}...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
model.to(device)
print("✅ Model loaded successfully!")


print("\n" + "="*60)
print("TESTING ON HELD-OUT TEST SET")
print("="*60)


# =========================
# Test evaluation
# =========================
BATCH_SIZE = 16
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
misclassifications = 0
for i, (true_label, pred_label) in enumerate(zip(true_labels, preds)):
    if true_label != pred_label:
        misclassifications += 1
        true_name = label_map["label_to_name"][str(true_label)]
        pred_name = label_map["label_to_name"][str(pred_label)]
        print(f"\nSample {i+1}:")
        print(f"  Text: {X_test[i][:100]}...")
        print(f"  True: {true_name} (label {true_label})")
        print(f"  Predicted: {pred_name} (label {pred_label})")

if misclassifications == 0:
    print("\n✅ No misclassifications! Perfect test accuracy!")
else:
    print(f"\nTotal misclassifications: {misclassifications}/{len(true_labels)}")
