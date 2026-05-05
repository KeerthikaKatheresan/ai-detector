"""
app_transformer.py  —  AI Writing Detector (Transformer version)
Uses roberta-base-openai-detector for much better accuracy on all text types
Run: streamlit run app_transformer.py

Install extra dependency first:
pip install transformers torch
"""

import streamlit as st
from feature_extractor import (
    burstiness, entropy, type_token_ratio,
    avg_sentence_length, ai_phrase_count,
    adverb_count, contraction_count, first_person_count,
    get_sentences, get_words, AI_PHRASES
)

st.set_page_config(page_title="AI Writing Detector", page_icon="🔍", layout="centered")

st.markdown("""
<style>
  .verdict-box { padding: 20px 24px; border-radius: 12px; margin: 16px 0; }
  .ai-box    { background: #fff0f0; border: 1.5px solid #e74c3c; }
  .human-box { background: #f0fff4; border: 1.5px solid #27ae60; }
  .mixed-box { background: #fffbf0; border: 1.5px solid #f39c12; }
  .verdict-title { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .verdict-sub   { font-size: 14px; color: #555; }
  .feat-row { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
  .feat-label { width: 260px; font-size: 13px; color: #444; flex-shrink: 0; }
  .feat-bar-bg { flex: 1; height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
  .feat-bar-fill { height: 100%; border-radius: 4px; }
  .feat-val { width: 48px; font-size: 12px; color: #888; text-align: right; }
</style>
""", unsafe_allow_html=True)


# ── Load transformer model ───────────────────────────
@st.cache_resource
def load_transformer():
    try:
        from transformers import pipeline as hf_pipeline
        detector = hf_pipeline(
            "text-classification",
            model="roberta-base-openai-detector",
            device=-1  # CPU; change to 0 if you have GPU
        )
        return detector, None
    except Exception as e:
        return None, str(e)


detector, load_error = load_transformer()


# ── Helpers ──────────────────────────────────────────
def chunk_text(text, max_tokens=480):
    """Split long text into chunks (RoBERTa has 512 token limit)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks


def predict_transformer(text):
    """
    Returns (ai_prob, human_prob) as percentages.
    Averages across chunks for long texts.
    """
    chunks = chunk_text(text)
    ai_scores = []

    for chunk in chunks:
        result = detector(chunk)[0]
        label  = result['label'].upper()
        score  = result['score']
        # Model labels: LABEL_1 = AI/Fake, LABEL_0 = Human/Real
        if label in ('LABEL_1', 'FAKE'):
            ai_scores.append(score)
        else:
            ai_scores.append(1 - score)

    ai_prob    = round(sum(ai_scores) / len(ai_scores) * 100, 1)
    human_prob = round(100 - ai_prob, 1)
    return ai_prob, human_prob


def get_verdict(ai_prob):
    if ai_prob >= 70:
        return "🤖 Likely AI-Generated", "ai-box", "#e74c3c"
    elif ai_prob >= 50:
        return "🔍 Possibly AI-Generated", "mixed-box", "#f39c12"
    elif ai_prob >= 35:
        return "❓ Uncertain — Mixed Signals", "mixed-box", "#f39c12"
    else:
        return "✍️ Likely Human-Written", "human-box", "#27ae60"


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
st.markdown("Powered by **RoBERTa** (OpenAI detector) — accurate on emails, essays, articles, and more.")

if load_error:
    st.error(f"Failed to load transformer model: {load_error}")
    st.code("pip install transformers torch", language="bash")
    st.stop()

# ── Text input ───────────────────────────────────────
text = st.text_area(
    label="Enter text",
    placeholder="Paste any text — email, essay, article, chat message...",
    height=220,
    label_visibility="collapsed"
)

word_count = len(text.split()) if text.strip() else 0
st.caption(f"{word_count} words")

if text.strip() and word_count >= 10:

    with st.spinner("Analyzing..."):
        ai_prob, human_prob = predict_transformer(text)

    verdict_text, box_class, _ = get_verdict(ai_prob)

    # ── Verdict ───────────────────────────────────────
    st.markdown(f"""
    <div class="verdict-box {box_class}">
      <div class="verdict-title">{verdict_text}</div>
      <div class="verdict-sub">
        AI probability: <b>{ai_prob}%</b> &nbsp;·&nbsp; Human probability: <b>{human_prob}%</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(ai_prob), text=f"AI likelihood: {ai_prob}%")

    st.divider()

    # ── Linguistic breakdown ──────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Human signals")
        st.caption("Higher = more human-like")
        feature_bar("Burstiness",         burstiness(text),         1.5,  "#27ae60")
        feature_bar("Shannon Entropy",    entropy(text),            6.0,  "#27ae60")
        feature_bar("Type-Token Ratio",   type_token_ratio(text),   1.0,  "#27ae60")
        feature_bar("Contractions",       contraction_count(text),  8.0,  "#27ae60")
        feature_bar("First-Person Usage", first_person_count(text), 10.0, "#27ae60")

    with col2:
        st.markdown("#### 🤖 AI signals")
        st.caption("Higher = more AI-like")
        feature_bar("AI Phrases",          ai_phrase_count(text),    8.0,  "#e74c3c")
        feature_bar("Hedge Adverbs",       adverb_count(text),       6.0,  "#e74c3c")
        feature_bar("Avg Sentence Length", avg_sentence_length(text),35.0, "#e74c3c")

    st.divider()

    # ── Stats ─────────────────────────────────────────
    st.markdown("#### 📝 Text statistics")
    sentences = get_sentences(text)
    words     = get_words(text)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words",            word_count)
    c2.metric("Sentences",        len(sentences))
    c3.metric("Unique words",     len(set(words)))
    c4.metric("Avg sent. length", f"{avg_sentence_length(text):.1f}w")

    # ── Flagged phrases ───────────────────────────────
    found = [p for p in AI_PHRASES if p in text.lower()]
    if found:
        st.divider()
        st.markdown("#### 🚩 Flagged AI phrases")
        for p in found:
            st.markdown(f"- `{p}`")

elif text.strip() and word_count < 10:
    st.info(f"Enter at least 10 words. ({word_count}/10 so far)")
else:
    st.markdown("""
    ---
    **Model:** `roberta-base-openai-detector` (OpenAI / HuggingFace)  
    Works accurately on: emails, essays, articles, formal writing, casual text  
    No training needed — downloads once (~500MB), cached after first run
    """)

st.markdown("---")
st.caption("Transformer-based detection. More accurate than rule-based approaches but not infallible.")
