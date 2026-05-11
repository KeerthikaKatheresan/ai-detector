"""
train_bert.py  —  Fine-tune DistilBERT for AI text detection
Expected accuracy: 95-97%
Run once: python train_bert.py

Install first:
pip install transformers torch datasets scikit-learn
"""

import pickle
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 256       # token limit per text
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
MAX_SAMPLES = 8000      # per class (8000 human + 8000 AI = 16000 total)
SAVE_PATH   = "bert_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Using device: {device}")
if device.type == "cpu":
    print("  (No GPU detected — training on CPU, will take ~20-30 mins)")
    print("  (With GPU it takes ~5 mins)")

print("=" * 55)
print("  AI Detector — DistilBERT Fine-tuning")
print("=" * 55)


# ── 1. Load & prepare data ───────────────────────────
print("\n[1/5] Loading dataset...")

try:
    dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")
    texts, labels = [], []
    for item in dataset:
        text   = item.get("text", "")
        source = item.get("source", "")
        if text and len(text.split()) >= 20:
            labels.append(0 if source == "human" else 1)
            texts.append(text)
except Exception:
    dataset = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)
    texts, labels = [], []
    for item in dataset['train']:
        for ans in item.get('human_answers', []):
            if ans and len(ans.split()) >= 20:
                texts.append(ans); labels.append(0)
        for ans in item.get('chatgpt_answers', []):
            if ans and len(ans.split()) >= 20:
                texts.append(ans); labels.append(1)

print(f"  Loaded: {len(texts):,} | Human: {labels.count(0):,} | AI: {labels.count(1):,}")


# ── 2. Balance ───────────────────────────────────────
print("\n[2/5] Balancing & splitting...")

human_texts = [t for t, l in zip(texts, labels) if l == 0]
ai_texts    = [t for t, l in zip(texts, labels) if l == 1]

n = min(MAX_SAMPLES, len(human_texts), len(ai_texts))
np.random.seed(42)
h_idx = np.random.choice(len(human_texts), n, replace=False)
a_idx = np.random.choice(len(ai_texts),    n, replace=False)

all_texts  = [human_texts[i] for i in h_idx] + [ai_texts[i] for i in a_idx]
all_labels = [0] * n + [1] * n

X_train, X_test, y_train, y_test = train_test_split(
    all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")


# ── 3. Tokenize ──────────────────────────────────────
print("\n[3/5] Tokenizing...")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

train_dataset = TextDataset(X_train, y_train)
test_dataset  = TextDataset(X_test,  y_test)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

print(f"  Tokenized. Max length: {MAX_LENGTH} tokens")


# ── 4. Train ─────────────────────────────────────────
print(f"\n[4/5] Fine-tuning DistilBERT ({EPOCHS} epochs)...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch   = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg:.4f}")

    avg_loss = total_loss / len(train_loader)

    # Validation after each epoch
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(batch['labels'].numpy())

    epoch_acc = accuracy_score(all_true, all_preds)
    print(f"\n  ✅ Epoch {epoch+1} done | Loss: {avg_loss:.4f} | Val Accuracy: {epoch_acc*100:.2f}%\n")


# ── 5. Final evaluation ──────────────────────────────
print("\n[5/5] Final Evaluation...")

cm = confusion_matrix(all_true, all_preds)
print(f"\n  Final Accuracy: {epoch_acc*100:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"               Predicted Human   Predicted AI")
print(f"  Actual Human       {cm[0][0]:>5}          {cm[0][1]:>5}")
print(f"  Actual AI          {cm[1][0]:>5}          {cm[1][1]:>5}")
print(f"\n{classification_report(all_true, all_preds, target_names=['Human', 'AI'])}")


# ── Save model ───────────────────────────────────────
import os
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n  Model saved → {SAVE_PATH}/")
print("  Run: streamlit run app_bert.py")
print("=" * 55)
