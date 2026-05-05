"""
app.py  —  AI Writing Detector (Streamlit)
Run: streamlit run app.py
"""

import streamlit as st
import pickle
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack
from feature_extractor import (
    extract_features, FEATURE_NAMES,
    burstiness, entropy, type_token_ratio,
    avg_sentence_length, ai_phrase_count,
    adverb_count, contraction_count, first_person_count,
    get_sentences, get_words
)

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="AI Writing Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
  .main { max-width: 780px; }
  .verdict-box {
      padding: 20px 24px;
      border-radius: 12px;
      margin: 16px 0;
  }
  .ai-box    { background: #fff0f0; border: 1.5px solid #e74c3c; }
  .human-box { background: #f0fff4; border: 1.5px solid #27ae60; }
  .mixed-box { background: #fffbf0; border: 1.5px solid #f39c12; }
  .verdict-title { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .verdict-sub   { font-size: 14px; color: #555; }
  .feat-row {
      display: flex; align-items: center;
      margin-bottom: 10px; gap: 10px;
  }
  .feat-label { width: 260px; font-size: 13px; color: #444; flex-shrink: 0; }
  .feat-bar-bg {
      flex: 1; height: 8px; background: #eee;
      border-radius: 4px; overflow: hidden;
  }
  .feat-bar-fill { height: 100%; border-radius: 4px; }
  .feat-val { width: 48px; font-size: 12px; color: #888; text-align: right; }
</style>
""", unsafe_allow_html=True)


# ── Load model bundle ────────────────────────────────
@st.cache_resource
def load_bundle():
    if not os.path.exists('model.pkl'):
        return None
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)


bundle = load_bundle()


# ── Predict using TF-IDF + handcrafted features ──────
def predict(text):
    model      = bundle['model']
    word_tfidf = bundle['word_tfidf']
    char_tfidf = bundle['char_tfidf']
    scaler     = bundle['scaler']

    X_word = word_tfidf.transform([text])
    X_char = char_tfidf.transform([text])

    X_hand        = np.array([extract_features(text)])
    X_hand_scaled = scaler.transform(X_hand)
    X_hand_sparse = sp.csr_matrix(X_hand_scaled)

    X_combined = hstack([X_word, X_char, X_hand_sparse])
    proba      = model.predict_proba(X_combined)[0]
    return round(proba[1] * 100, 1), round(proba[0] * 100, 1)


# ── Helper: verdict ──────────────────────────────────
def get_verdict(ai_prob):
    if ai_prob >= 70:
        return "🤖 Likely AI-Generated", "ai-box", "#e74c3c"
    elif ai_prob >= 50:
        return "🔍 Possibly AI-Generated", "mixed-box", "#f39c12"
    elif ai_prob >= 35:
        return "❓ Uncertain — Mixed Signals", "mixed-box", "#f39c12"
    else:
        return "✍️ Likely Human-Written", "human-box", "#27ae60"


# ── Helper: feature bar ──────────────────────────────
def feature_bar(label, value, max_val, color):
    pct = min(100, int((value / max_val) * 100)) if max_val > 0 else 0
    st.markdown(f"""
    <div class="feat-row">
      <div class="feat-label">{label}</div>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width:{pct}%; background:{color};"></div>
      </div>
      <div class="feat-val">{round(value, 2)}</div>
    </div>
    """, unsafe_allow_html=True)


# ── UI ───────────────────────────────────────────────
st.title("🔍 AI Writing Detector")
st.markdown("Paste or type any text below. The model predicts whether it was written by a human or an AI.")

if bundle is None:
    st.warning("⚠️ **Model not trained yet.** Run `python train_model.py` first, then restart this app.")
    st.code("python train_model.py", language="bash")
    st.stop()

# ── Text input ───────────────────────────────────────
text = st.text_area(
    label="Enter text to analyze",
    placeholder="Start typing or paste your text here...\n\nTip: longer texts (50+ words) give more accurate results.",
    height=200,
    label_visibility="collapsed"
)

word_count = len(text.split()) if text.strip() else 0
st.caption(f"{word_count} words")

# ── Real-time prediction ─────────────────────────────
if text.strip() and word_count >= 15:

    ai_prob, human_prob = predict(text)
    verdict_text, box_class, bar_color = get_verdict(ai_prob)

    # ── Verdict card ─────────────────────────────────
    st.markdown(f"""
    <div class="verdict-box {box_class}">
      <div class="verdict-title">{verdict_text}</div>
      <div class="verdict-sub">AI probability: <b>{ai_prob}%</b> &nbsp;·&nbsp; Human probability: <b>{human_prob}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(ai_prob), text=f"AI likelihood: {ai_prob}%")

    st.divider()

    # ── Feature breakdown ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Human signals")
        st.caption("Higher = more human-like")
        feature_bar("Burstiness",           burstiness(text),        1.5,  "#27ae60")
        feature_bar("Shannon Entropy",      entropy(text),           6.0,  "#27ae60")
        feature_bar("Type-Token Ratio",     type_token_ratio(text),  1.0,  "#27ae60")
        feature_bar("Contractions",         contraction_count(text), 8.0,  "#27ae60")
        feature_bar("First-Person Usage",   first_person_count(text),10.0, "#27ae60")

    with col2:
        st.markdown("#### 🤖 AI signals")
        st.caption("Higher = more AI-like")
        feature_bar("AI Phrases",           ai_phrase_count(text),   8.0,  "#e74c3c")
        feature_bar("Hedge Adverbs",        adverb_count(text),      6.0,  "#e74c3c")
        feature_bar("Avg Sentence Length",  avg_sentence_length(text), 35.0, "#e74c3c")

    st.divider()

    # ── Text statistics ───────────────────────────────
    st.markdown("#### 📝 Text statistics")
    sentences = get_sentences(text)
    words     = get_words(text)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words",            word_count)
    c2.metric("Sentences",        len(sentences))
    c3.metric("Unique words",     len(set(words)))
    c4.metric("Avg sent. length", f"{avg_sentence_length(text):.1f}w")

    # ── Flagged phrases ───────────────────────────────
    from feature_extractor import AI_PHRASES
    found_phrases = [p for p in AI_PHRASES if p in text.lower()]
    if found_phrases:
        st.divider()
        st.markdown("#### 🚩 Flagged AI phrases")
        for phrase in found_phrases:
            st.markdown(f"- `{phrase}`")

elif text.strip() and word_count < 15:
    st.info(f"⏳ Enter at least **15 words** for a prediction. ({word_count}/15 so far)")

else:
    st.markdown("""
    ---
    **How it works:**
    - Trained on a real human vs AI text dataset from HuggingFace
    - Uses **TF-IDF word + character n-grams** (strongest signal) + 9 handcrafted linguistic features
    - **Logistic Regression** classifier — ~90%+ accuracy
    - Updates prediction as you type
    """)

st.markdown("---")
st.caption("ML-based analysis. Not a definitive classifier — use as a signal, not a verdict.")
