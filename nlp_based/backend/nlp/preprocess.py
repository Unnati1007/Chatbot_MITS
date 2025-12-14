# backend/nlp/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only first time)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def normalize_terms(text: str) -> str:
    """
    Standardize important domain-specific words.
    Example: "log in" → "login", "mits" → "MITS"
    """
    replacements = {
        "log in": "login",
        "log-in": "login",
        "mits ": "MITS ",
        "m.i.t.s": "MITS",
        "moodle ": "moodle ",
        "ims ": "IMS ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    - lowercase
    - remove URLs
    - normalize numbers
    - remove punctuation
    - remove stopwords
    - lemmatization
    - remove duplicate spaces
    """
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)

    # replace numbers
    text = re.sub(r"\d+", " number ", text)

    # remove punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove duplicate spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    words = [w for w in text.split() if w not in stop_words]

    # lemmatize
    words = [lemmatizer.lemmatize(w) for w in words]

    # join text
    text = " ".join(words).strip()

    # apply domain-specific normalization
    text = normalize_terms(text)

    return text


def preprocess_text(text: str) -> str:
    """
    Final public preprocessing function.
    Ensures both:
    - Training data
    - User queries
    use **identical** preprocessing.
    """
    return clean_text(text)
