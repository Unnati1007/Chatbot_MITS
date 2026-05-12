# backend/nlp/preprocess.py

import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

# ---------- GLOBAL OBJECTS ----------
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# =====================================================
# 4A. HEAVY PREPROCESSING (RULE-BASED MATCHING)
# =====================================================
def preprocess_for_rules(text: str) -> str:
    """
    Heavy normalization for keyword / rule matching.
    Used in rules.py
    """

    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Normalize unicode (é → e, etc.)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords + lemmatize
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS
    ]

    # Rejoin
    text = " ".join(cleaned_tokens)

    # Remove duplicate spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# 4B. LIGHT PREPROCESSING (EMBEDDINGS - MiniLM)
# =====================================================
def preprocess_for_embeddings(text: str) -> str:
    """
    Light normalization for sentence embeddings.
    Used in build_embeddings.py & matcher.py
    """

    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    # Replace weird punctuation with space (keep sentence structure)
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove duplicate spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
