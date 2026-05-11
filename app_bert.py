"""
app_bert.py  —  UI only (Streamlit)
All AI/ML logic lives in predictor.py
All feature math lives in feature_extractor.py
"""

import streamlit as st
import torch
from predictor import load_model, predict, get_verdict
from feature_extractor import (
    burstiness, entropy, type_token_ratio,
    avg_sentence_length, ai_phrase_count,
    adverb_count, contraction_count, first_person_count,
    get_sentences, get_words, AI_PHRASES
)

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Writing Detector",
    page_icon="🔍",
    layout="centered"
)

# ─────────────────────────────────────────────────────
# STYLES  — unsafe_allow_html=True is REQUIRED here
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-box {
    padding: 24px;
    border-radius: 16px;
    margin: 18px 0;
}
.ai-box    { background: #fff0f0; border: 2px solid #e74c3c; }
.human-box { background: #f0fff4; border: 2px solid #27ae60; }
.mixed-box { background: #fffaf0; border: 2px solid #f39c12; }
.feat-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
}
.feat-label {
    width: 220px;
    font-size: 13px;
    color: #444;
    flex-shrink: 0;
}
.feat-bar-bg {
    flex: 1;
    height: 8px;
    background: #eee;
    border-radius: 4px;
    overflow: hidden;
}
.feat-bar-fill {
    height: 100%;
    border-radius: 4px;
}
.feat-val {
    width: 45px;
    font-size: 12px;
    color: #777;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

model, tokenizer, load_error = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model:
    model.to(device)

# ─────────────────────────────────────────────────────
# FEATURE BAR HELPER
# ─────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
st.title("🔍 AI Writing Detector")
st.markdown("Powered by **fine-tuned DistilBERT** with heuristic + n-gram analysis.")

# ─────────────────────────────────────────────────────
# MODEL ERROR
# ─────────────────────────────────────────────────────
if load_error:
    st.error(f"⚠️ {load_error}")
    st.code("python train_bert.py", language="bash")
    st.stop()

# ─────────────────────────────────────────────────────
# TEXT INPUT
# ─────────────────────────────────────────────────────
text = st.text_area(
    label="Enter text",
    placeholder="Paste any text here...",
    height=220,
    label_visibility="collapsed"
)

word_count = len(text.split()) if text.strip() else 0
st.caption(f"{word_count} words")

# ─────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────
if text.strip() and word_count >= 10:

    with st.spinner("Analyzing text..."):
        result = predict(text, model, tokenizer, device)

    ai_prob    = result["ai_prob"]
    human_prob = result["human_prob"]
    bert_score = result["bert_score"]
    feat_score = result["feat_score"]

    verdict_text, box_class, _ = get_verdict(ai_prob)

    # ── Verdict card ─────────────────────────────────
    st.markdown(f"""
<div class="verdict-box {box_class}">
    <div style="font-size:32px; font-weight:800; margin-bottom:8px;">{verdict_text}</div>
    <div style="font-size:28px; font-weight:700; margin-bottom:6px; color:#111;">{ai_prob}% AI Probability</div>
    <div style="font-size:16px; color:#555; margin-bottom:14px;">Human Probability: <b>{human_prob}%</b></div>
    <hr style="opacity:0.2; margin:12px 0;">
    <div style="display:flex; flex-wrap:wrap; gap:18px; font-size:14px; color:#444; margin-top:8px;">
        <div>🧠 <b>BERT Score:</b> {bert_score}%</div>
        <div>📐 <b>Feature Score:</b> {feat_score}%</div>
        <div>🎯 <b>Confidence:</b> {max(ai_prob, human_prob)}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.progress(int(ai_prob), text=f"AI Likelihood: {ai_prob}%")

    st.divider()

    # ── Feature signals ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Human Signals")
        st.caption("Higher = more human-like")
        feature_bar("Burstiness",         burstiness(text),         1.5,  "#27ae60")
        feature_bar("Shannon Entropy",    entropy(text),            6.0,  "#27ae60")
        feature_bar("Type-Token Ratio",   type_token_ratio(text),   1.0,  "#27ae60")
        feature_bar("Contractions",       contraction_count(text),  8.0,  "#27ae60")
        feature_bar("First-Person Usage", first_person_count(text), 10.0, "#27ae60")

    with col2:
        st.markdown("#### 🤖 AI Signals")
        st.caption("Higher = more AI-like")
        feature_bar("AI Phrases",          ai_phrase_count(text),    8.0,  "#e74c3c")
        feature_bar("Hedge Adverbs",       adverb_count(text),       6.0,  "#e74c3c")
        feature_bar("Avg Sentence Length", avg_sentence_length(text),35.0, "#e74c3c")

    st.divider()

    # ── Text statistics ───────────────────────────────
    st.markdown("#### 📝 Text Statistics")
    sentences = get_sentences(text)
    words     = get_words(text)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words",        word_count)
    c2.metric("Sentences",    len(sentences))
    c3.metric("Unique Words", len(set(words)))
    c4.metric("Avg Sentence", f"{avg_sentence_length(text):.1f}w")

    # ── Flagged phrases ───────────────────────────────
    found = [p for p in AI_PHRASES if p in text.lower()]
    if found:
        st.divider()
        st.markdown("#### 🚩 Flagged AI Phrases")
        for p in found:
            st.markdown(f"- `{p}`")

# ─────────────────────────────────────────────────────
# TOO SHORT
# ─────────────────────────────────────────────────────
elif text.strip() and word_count < 10:
    st.info(f"Enter at least 10 words. ({word_count}/10 so far)")

# ─────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────
else:
    st.markdown("""
---
### 📌 Model Information
- **Model:** DistilBERT
- **Detection:** Transformer + Heuristics + N-Grams
- **Training:** Human vs AI dataset
- **Expected Accuracy:** ~95%

Run training:
```bash
python train_bert.py
```
""")

# ─────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────
st.markdown("---")
st.caption("Transformer-based AI text detection. Use as an intelligent signal — not absolute proof.")