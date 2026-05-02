import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove punctuation & numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # remove stopwords
    tokens = [
        word for word in text.split()
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(tokens)
