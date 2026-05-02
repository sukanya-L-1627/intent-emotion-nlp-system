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
df = pd.read_csv("data/go_emotions.csv")

# --------------------------------------------------
# 2️⃣ Drop unnecessary columns
# --------------------------------------------------
df = df.drop(columns=["id", "example_very_unclear"])

# --------------------------------------------------
# 3️⃣ Label engineering (final emotions)
# --------------------------------------------------
df["anger_final"] = df[["anger", "annoyance"]].max(axis=1)
df["frustration_final"] = df[["disappointment", "disapproval"]].max(axis=1)
df["sadness_final"] = df[["sadness", "grief"]].max(axis=1)
df["joy_final"] = df[["joy", "amusement", "excitement"]].max(axis=1)
df["satisfaction_final"] = df[["gratitude", "approval"]].max(axis=1)
df["neutral_final"] = df["neutral"]

final_columns = [
    "text",
    "anger_final",
    "frustration_final",
    "sadness_final",
    "joy_final",
    "satisfaction_final",
    "neutral_final"
]

df = df[final_columns]

# --------------------------------------------------
# 4️⃣ Text preprocessing (FAST)
# --------------------------------------------------
df["clean_text"] = df["text"].apply(clean_text)

# --------------------------------------------------
# 5️⃣ Prepare X and y
# --------------------------------------------------
X = df["clean_text"]

y = df[
    [
        "anger_final",
        "frustration_final",
        "sadness_final",
        "joy_final",
        "satisfaction_final",
        "neutral_final",
    ]
]

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
    max_features=5000,
    min_df=5
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --------------------------------------------------
# 8️⃣ Multi-label Model (One-vs-Rest)
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
# Custom threshold prediction
y_proba = model.predict_proba(X_test_tfidf)

threshold = 0.4
y_pred = (y_proba >= threshold).astype(int)

print("\nEvaluation Metrics")
print("------------------")
print("Hamming Loss :", hamming_loss(y_test, y_pred))
print("Micro F1     :", f1_score(y_test, y_pred, average="micro"))
print("Macro F1     :", f1_score(y_test, y_pred, average="macro"))

# --------------------------------------------------
# 🔟 Save model & vectorizer
# --------------------------------------------------
pickle.dump(model, open("emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
