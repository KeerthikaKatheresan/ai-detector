import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from feature_extractor import extract_features

# ── Config ───────────────────────────────────────────
MODEL_PATH = "bert_model/bert_model"

MAX_LENGTH = 256
CHUNK_SIZE = 200

TEMPERATURE = 2.0

BERT_WEIGHT = 0.75
FEATURE_WEIGHT = 0.25


# ── Load model ───────────────────────────────────────
def load_model(model_path=MODEL_PATH):

    import os

    if not os.path.exists(model_path):
        return None, None, "Model not found"

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

        model = DistilBertForSequenceClassification.from_pretrained(
            model_path
        )

        model.eval()

        return model, tokenizer, None

    except Exception as e:
        return None, None, str(e)


# ── Chunk text ───────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE):

    words = text.split()

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# ── BERT probability ─────────────────────────────────
def get_bert_score(text, model, tokenizer, device):

    chunks = chunk_text(text)

    scores = []

    for chunk in chunks:

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            logits = model(**inputs).logits

            scaled_logits = logits / TEMPERATURE

            probs = torch.softmax(
                scaled_logits,
                dim=1
            )[0].cpu().numpy()

            scores.append(float(probs[1]))

    return sum(scores) / len(scores)


# ── Heuristic score ──────────────────────────────────
def get_feature_score(text):

    f = extract_features(text)

    ai_score = 0.0
    human_score = 0.0

    # AI signals
    ai_score += min(f[3] / 50, 0.20)
    ai_score += min(f[4] * 0.15, 0.30)
    ai_score += min(f[5] * 0.10, 0.20)

    # NEW NGRAM SIGNALS
    ai_score += min(f[9] * 0.40, 0.15)
    ai_score += min(f[10] * 0.50, 0.15)

    # Human signals
    human_score += min(f[0], 0.20)
    human_score += min(f[6] * 0.10, 0.20)
    human_score += min(f[7] * 0.10, 0.20)

    raw = 0.5 + (ai_score - human_score)

    return max(0.05, min(0.95, raw))


# ── Final prediction ─────────────────────────────────
def predict(text, model, tokenizer, device):

    bert_score = get_bert_score(
        text,
        model,
        tokenizer,
        device
    )

    feat_score = get_feature_score(text)

    final = (
        (BERT_WEIGHT * bert_score)
        +
        (FEATURE_WEIGHT * feat_score)
    )

    ai_prob = round(final * 100, 1)

    return {
        "ai_prob": ai_prob,
        "human_prob": round(100 - ai_prob, 1),

        "bert_score": round(bert_score * 100, 1),

        "feat_score": round(feat_score * 100, 1)
    }


def get_verdict(ai_prob):

    if ai_prob >= 70:
        return (
            "🤖 Likely AI-Generated",
            "ai-box",
            "#e74c3c"
        )

    elif ai_prob >= 50:
        return (
            "🔍 Possibly AI-Generated",
            "mixed-box",
            "#f39c12"
        )

    elif ai_prob >= 35:
        return (
            "❓ Mixed Signals",
            "mixed-box",
            "#f39c12"
        )

    else:
        return (
            "✍️ Likely Human-Written",
            "human-box",
            "#27ae60"
        )