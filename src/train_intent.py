import pandas as pd
import pickle

from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, f1_score

# --------------------------------------------------
# 1️⃣ Load dataset
# --------------------------------------------------
df = pd.read_csv("data/sample.csv")

# Keep only customer messages
df = df[df["inbound"] == True]
df = df[["text"]].dropna()

# --------------------------------------------------
# 2️⃣ Intent keyword rules (same as before)
# --------------------------------------------------
complaint_keywords = [
    "error", "issue", "problem", "not working", "broken",
    "delay", "late", "failed", "crash", "disappointed"
]

request_keywords = [
    "help", "please", "support", "fix", "resolve",
    "assist", "can you", "could you"
]

inquiry_keywords = [
    "how", "what", "why", "when", "where",
    "is it", "does", "do i", "can i"
]

def detect_intent(text, keywords):
    text = text.lower()
    return int(any(k in text for k in keywords))

# --------------------------------------------------
# 3️⃣ Create multi-label intent columns
# --------------------------------------------------
df["complaint"] = df["text"].apply(
    lambda x: detect_intent(x, complaint_keywords)
)
df["request"] = df["text"].apply(
    lambda x: detect_intent(x, request_keywords)
)
df["inquiry"] = df["text"].apply(
    lambda x: detect_intent(x, inquiry_keywords)
)

# --------------------------------------------------
# 4️⃣ Text preprocessing
# --------------------------------------------------
df["clean_text"] = df["text"].apply(clean_text)

# --------------------------------------------------
# 5️⃣ Prepare X and y
# --------------------------------------------------
X = df["clean_text"]
y = df[["complaint", "request", "inquiry"]]

# --------------------------------------------------
# 6️⃣ Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 7️⃣ TF-IDF Vectorization
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --------------------------------------------------
# 8️⃣ Multi-label Intent Model
# --------------------------------------------------
model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)

model.fit(X_train_tfidf, y_train)

# --------------------------------------------------
# 9️⃣ Evaluation
# --------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("\nIntent Model Evaluation")
print("-----------------------")
print("Hamming Loss :", hamming_loss(y_test, y_pred))
print("Micro F1     :", f1_score(y_test, y_pred, average="micro"))
print("Macro F1     :", f1_score(y_test, y_pred, average="macro"))

# --------------------------------------------------
# 🔟 Save model & vectorizer
# --------------------------------------------------
pickle.dump(model, open("intent_model.pkl", "wb"))
pickle.dump(vectorizer, open("intent_tfidf.pkl", "wb"))

print("\nIntent model and vectorizer saved successfully!")
