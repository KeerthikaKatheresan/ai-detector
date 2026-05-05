import re
import math
from collections import Counter

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

FIRST_PERSON = [' i ', " i'm", " i've", " i'll", " i'd", ' my ', ' me ']


def get_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]


def get_words(text):
    return re.findall(r"\b[a-z']+\b", text.lower())


def burstiness(text):
    sentences = get_sentences(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    m = sum(lengths) / len(lengths)
    variance = sum((l - m) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    return round(std / m, 4) if m > 0 else 0.0


def entropy(text):
    words = get_words(text)
    if len(words) < 5:
        return 0.0
    freq = Counter(words)
    total = len(words)
    return round(-sum((c / total) * math.log2(c / total) for c in freq.values()), 4)


def type_token_ratio(text):
    words = get_words(text)
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def avg_sentence_length(text):
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    return round(sum(len(s.split()) for s in sentences) / len(sentences), 2)


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
    return sum(lower.count(m) for m in FIRST_PERSON)


def extract_features(text):
    """
    Returns a 9-element feature vector.
    Feature order must match training exactly.
    """
    return [
        burstiness(text),           # f0: high = human (varied sentence length)
        entropy(text),              # f1: high = human (diverse vocabulary)
        type_token_ratio(text),     # f2: high = human (less repetition)
        avg_sentence_length(text),  # f3: high = AI (long uniform sentences)
        ai_phrase_count(text),      # f4: high = AI (filler/transition phrases)
        adverb_count(text),         # f5: high = AI (hedge adverbs)
        contraction_count(text),    # f6: high = human (natural register)
        first_person_count(text),   # f7: high = human (personal voice)
        len(text.split()),          # f8: word count (normalizing feature)
    ]


FEATURE_NAMES = [
    'Burstiness (sentence length variation)',
    'Shannon Entropy (vocabulary diversity)',
    'Type-Token Ratio (word uniqueness)',
    'Avg Sentence Length',
    'AI Phrase Count',
    'Hedge Adverb Count',
    'Contraction Count',
    'First-Person Count',
    'Total Word Count',
]
