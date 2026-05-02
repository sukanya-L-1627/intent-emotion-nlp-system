import pickle
from preprocess import clean_text

# -----------------------------------------
# 1️⃣ Load saved models and vectorizers
# -----------------------------------------

# Emotion model
emotion_model = pickle.load(open("emotion_model.pkl", "rb"))
emotion_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Intent model
intent_model = pickle.load(open("intent_model.pkl", "rb"))
intent_vectorizer = pickle.load(open("intent_tfidf.pkl", "rb"))

# Label names (IMPORTANT)
emotion_labels = [
    "anger",
    "frustration",
    "sadness",
    "joy",
    "satisfaction",
    "neutral"
]

intent_labels = [
    "complaint",
    "request",
    "inquiry"
]

# -----------------------------------------
# 2️⃣ Prediction function
# -----------------------------------------

def predict_intent_emotion(text, threshold=0.4):
    # Preprocess
    clean = clean_text(text)

    # -------- Emotion prediction --------
    X_emotion = emotion_vectorizer.transform([clean])
    emotion_probs = emotion_model.predict_proba(X_emotion)
    emotion_pred = (emotion_probs >= threshold).astype(int)[0]

    detected_emotions = [
        emotion_labels[i]
        for i, val in enumerate(emotion_pred)
        if val == 1
    ]
# -------- Intent prediction --------
    X_intent = intent_vectorizer.transform([clean])
    intent_probs = intent_model.predict_proba(X_intent)

    intent_pred = (intent_probs >= 0.3).astype(int)[0]

    detected_intents = [
    intent_labels[i]
    for i, val in enumerate(intent_pred)
    if val == 1
]


    return detected_intents, detected_emotions

# -----------------------------------------
# 3️⃣ Test with sample inputs
# -----------------------------------------

if __name__ == "__main__":
    samples = [
        "My order arrived late and I'm very frustrated. Please help me.",
        "Thanks a lot, the issue is resolved!",
        "How do I reset my password?"
    ]

    for text in samples:
        intents, emotions = predict_intent_emotion(text)
        print("\nText:", text)
        print("Detected Intents :", intents)
        print("Detected Emotions:", emotions)
