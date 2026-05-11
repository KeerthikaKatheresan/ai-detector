import re
import math
from collections import Counter

# ── AI phrases ───────────────────────────────────────
AI_PHRASES = [
    'it is important to note', 'in conclusion', 'furthermore',
    'it is worth noting', 'plays a crucial role', 'additionally',
    'notably', 'significantly', 'in today\'s', 'delve into',
    'it is imperative', 'has revolutionized', 'in the realm of',
    'it is essential', 'it is worth mentioning', 'a testament to',
    'can be seen as', 'it\'s worth noting', 'in summary',
    'to summarize', 'in essence', 'it goes without saying'
]

AI_ADVERBS = [
    'significantly', 'notably', 'fundamentally', 'substantially',
    'essentially', 'importantly', 'crucially', 'subsequently',
    'comprehensively', 'overwhelmingly', 'undeniably', 'remarkably'
]

CONTRACTIONS = [
    "don't", "can't", "won't", "i'm", "i've", "i'll",
    "isn't", "aren't", "wasn't", "weren't", "it's",
    "that's", "there's", "they're", "we're", "didn't", "couldn't"
]

FIRST_PERSON = [
    ' i ', " i'm", " i've", " i'll", " i'd", ' my ', ' me '
]


# ── Basic text helpers ───────────────────────────────
def get_sentences(text):
    return [
        s.strip()
        for s in re.split(r'[.!?]+', text)
        if len(s.strip()) > 5
    ]


def get_words(text):
    return re.findall(r"\b[a-z']+\b", text.lower())


# ── Statistical features ─────────────────────────────
def burstiness(text):
    sentences = get_sentences(text)

    if len(sentences) < 2:
        return 0.0

    lengths = [len(s.split()) for s in sentences]

    mean = sum(lengths) / len(lengths)

    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)

    std = math.sqrt(variance)

    return round(std / mean, 4) if mean > 0 else 0.0


def entropy(text):
    words = get_words(text)

    if len(words) < 5:
        return 0.0

    freq = Counter(words)

    total = len(words)

    return round(
        -sum((c / total) * math.log2(c / total) for c in freq.values()),
        4
    )


def type_token_ratio(text):
    words = get_words(text)

    if not words:
        return 0.0

    return round(len(set(words)) / len(words), 4)


def avg_sentence_length(text):
    sentences = get_sentences(text)

    if not sentences:
        return 0.0

    return round(
        sum(len(s.split()) for s in sentences) / len(sentences),
        2
    )


# ── AI phrase features ───────────────────────────────
def ai_phrase_count(text):
    lower = text.lower()
    return sum(1 for p in AI_PHRASES if p in lower)


def adverb_count(text):
    words = get_words(text)
    return sum(1 for w in words if w in AI_ADVERBS)


def contraction_count(text):
    lower = text.lower()
    return sum(1 for c in CONTRACTIONS if c in lower)


def first_person_count(text):
    lower = ' ' + text.lower() + ' '
    return sum(lower.count(x) for x in FIRST_PERSON)


# ── N-GRAM FEATURES ──────────────────────────────────
def get_ngrams(words, n):
    return [
        tuple(words[i:i+n])
        for i in range(len(words) - n + 1)
    ]


def repeated_bigram_ratio(text):
    words = get_words(text)

    if len(words) < 2:
        return 0.0

    bigrams = get_ngrams(words, 2)

    counts = Counter(bigrams)

    repeated = sum(1 for c in counts.values() if c > 1)

    return round(repeated / len(counts), 4)


def repeated_trigram_ratio(text):
    words = get_words(text)

    if len(words) < 3:
        return 0.0

    trigrams = get_ngrams(words, 3)

    counts = Counter(trigrams)

    repeated = sum(1 for c in counts.values() if c > 1)

    return round(repeated / len(counts), 4)


# ── Final feature vector ─────────────────────────────
def extract_features(text):

    return [

        # Human-like
        burstiness(text),
        entropy(text),
        type_token_ratio(text),

        # AI-like
        avg_sentence_length(text),
        ai_phrase_count(text),
        adverb_count(text),

        # Human-like
        contraction_count(text),
        first_person_count(text),

        # Normalizer
        len(text.split()),

        # NEW NGRAM FEATURES
        repeated_bigram_ratio(text),
        repeated_trigram_ratio(text),
    ]


FEATURE_NAMES = [
    'Burstiness',
    'Entropy',
    'Type Token Ratio',
    'Average Sentence Length',
    'AI Phrase Count',
    'AI Adverb Count',
    'Contraction Count',
    'First Person Count',
    'Word Count',
    'Repeated Bigram Ratio',
    'Repeated Trigram Ratio',
]