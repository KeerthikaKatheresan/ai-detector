"""
train_model.py  —  Improved version
Uses TF-IDF n-grams + handcrafted features for much better accuracy
Run once: python train_model.py
"""

import pickle
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from scipy.sparse import hstack
from feature_extractor import extract_features

print("=" * 55)
print("  AI Writing Detector — Improved Model Training")
print("=" * 55)

# ── 1. Load dataset ──────────────────────────────────
print("\n[1/5] Loading dataset...")

try:
    dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")
    texts, labels = [], []
    for item in dataset:
        text = item.get("text", "")
        source = item.get("source", "")
        if text and len(text.split()) >= 20:
            label = 0 if source == "human" else 1
            texts.append(text)
            labels.append(label)
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

print(f"  Total: {len(texts):,} | Human: {labels.count(0):,} | AI: {labels.count(1):,}")

# ── 2. Balance dataset ───────────────────────────────
print("\n[2/5] Balancing dataset...")

human_texts = [t for t, l in zip(texts, labels) if l == 0]
ai_texts    = [t for t, l in zip(texts, labels) if l == 1]

n = min(15000, len(human_texts), len(ai_texts))
np.random.seed(42)
h_idx = np.random.choice(len(human_texts), n, replace=False)
a_idx = np.random.choice(len(ai_texts),    n, replace=False)

balanced_texts  = [human_texts[i] for i in h_idx] + [ai_texts[i] for i in a_idx]
balanced_labels = [0] * n + [1] * n

combined = list(zip(balanced_texts, balanced_labels))
np.random.shuffle(combined)
balanced_texts, balanced_labels = zip(*combined)
balanced_texts  = list(balanced_texts)
balanced_labels = list(balanced_labels)

print(f"  Balanced: {len(balanced_texts):,} samples ({n:,} human + {n:,} AI)")

# ── 3. Handcrafted features ──────────────────────────
print("\n[3/5] Extracting handcrafted features...")
X_hand = np.array([extract_features(t) for t in balanced_texts])
y      = np.array(balanced_labels)
print(f"  Shape: {X_hand.shape}")

# ── 4. TF-IDF features ───────────────────────────────
print("\n[4/5] Building TF-IDF n-gram features...")

word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    analyzer='word'
)
char_tfidf = TfidfVectorizer(
    ngram_range=(3, 5),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    analyzer='char_wb'
)

X_word = word_tfidf.fit_transform(balanced_texts)
X_char = char_tfidf.fit_transform(balanced_texts)

scaler = StandardScaler()
X_hand_scaled  = scaler.fit_transform(X_hand)
X_hand_sparse  = sp.csr_matrix(X_hand_scaled)
X_combined     = hstack([X_word, X_char, X_hand_sparse])

print(f"  Combined feature matrix: {X_combined.shape}")

# ── 5. Train & evaluate ──────────────────────────────
print("\n[5/5] Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy: {acc * 100:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"               Predicted Human   Predicted AI")
print(f"  Actual Human       {cm[0][0]:>5}          {cm[0][1]:>5}")
print(f"  Actual AI          {cm[1][0]:>5}          {cm[1][1]:>5}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Human', 'AI'])}")

# ── Save bundle ──────────────────────────────────────
bundle = {
    'model':      model,
    'word_tfidf': word_tfidf,
    'char_tfidf': char_tfidf,
    'scaler':     scaler,
}
with open('model.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("  Saved → model.pkl")
print("\n  Run: streamlit run app.py")
print("=" * 55)
