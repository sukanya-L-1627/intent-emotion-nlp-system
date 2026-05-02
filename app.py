import streamlit as st
import pickle
from src.preprocess import clean_text

# =================================================
# Helper: Inquiry Detection (Rule-based fallback)
# =================================================
def is_inquiry_question(text):
    question_words = [
        "how", "what", "why", "when", "where",
        "can i", "do i", "does", "is it", "are"
    ]
    text = text.lower()
    return text.strip().endswith("?") or any(q in text for q in question_words)

# =================================================
# Page Configuration
# =================================================
st.set_page_config(
    page_title="Intent & Emotion Intelligence System",
    page_icon="🧠",
    layout="wide"
)

# =================================================
# Load Models & Vectorizers
# =================================================
emotion_model = pickle.load(open("emotion_model.pkl", "rb"))
emotion_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

intent_model = pickle.load(open("intent_model.pkl", "rb"))
intent_vectorizer = pickle.load(open("intent_tfidf.pkl", "rb"))

# =================================================
# Labels
# =================================================
emotion_labels = [
    "Anger",
    "Frustration",
    "Sadness",
    "Joy",
    "Satisfaction",
    "Neutral"
]

intent_labels = [
    "Complaint",
    "Request",
    "Inquiry"
]

# =================================================
# Internal Threshold
# =================================================
THRESHOLD = 0.5

# =================================================
# Initialize results (IMPORTANT for Streamlit reruns)
# =================================================
intent_results = []
emotion_results = []

# =================================================
# Sidebar
# =================================================
with st.sidebar:
    st.title("🧠 NLP Intelligence System")

    st.markdown("""
    **Advanced NLP Project**

    ✔ Multi-Label Intent Detection  
    ✔ Multi-Label Emotion Detection  
    ✔ Classical NLP (TF-IDF + ML)  
    ✔ Hybrid NLP (ML + Rules)  
    ✔ No LLM / No APIs  
    ✔ Real-World Datasets  
    """)

    st.divider()

    st.markdown("""
    **How it works**
    - Text is preprocessed
    - Intent & Emotion models run independently
    - Results are combined at inference time
    """)

# =================================================
# Main UI
# =================================================
st.title("📊 Intent & Emotion Analysis Dashboard")

st.markdown(
    "Analyze **what the user wants** and **how the user feels** from a single text input."
)

user_text = st.text_area(
    "Enter customer message:",
    height=130,
    placeholder="Example: My order arrived late and I'm very frustrated. Please help me."
)

analyze_btn = st.button("🔍 Analyze Text", use_container_width=True)

# =================================================
# Prediction Logic
# =================================================
if analyze_btn and user_text.strip():

    clean = clean_text(user_text)

    # -------- Emotion Prediction --------
    X_emotion = emotion_vectorizer.transform([clean])
    emotion_probs = emotion_model.predict_proba(X_emotion)[0]

    emotion_results = [
        (emotion_labels[i], round(emotion_probs[i], 3))
        for i in range(len(emotion_labels))
        if emotion_probs[i] >= THRESHOLD
    ]

    # -------- Intent Prediction (ML) --------
    X_intent = intent_vectorizer.transform([clean])
    intent_probs = intent_model.predict_proba(X_intent)[0]

    intent_results = [
        (intent_labels[i], round(intent_probs[i], 3))
        for i in range(len(intent_labels))
        if intent_probs[i] >= THRESHOLD
    ]

    # -------- Inquiry Fallback (RULE) --------
    if not intent_results and is_inquiry_question(user_text):
        intent_results = [("Inquiry", 1.0)]

# =================================================
# Results Display (INTENT + EMOTION TOGETHER)
# =================================================
if analyze_btn:

    if intent_results or emotion_results:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Detected Intents")
            if intent_results:
                for intent, score in intent_results:
                    st.success(f"{intent} | Confidence: {score}")
            else:
                st.info("No intent detected")

        with col2:
            st.subheader("💬 Detected Emotions")
            if emotion_results:
                for emotion, score in emotion_results:
                    st.info(f"{emotion} | Confidence: {score}")
            else:
                st.info("No emotion detected")

        # -------- Interpretation --------
        st.divider()
        st.subheader("🧠 Model Interpretation")

        summary = []

        if intent_results:
            summary.append(
                f"The user intent indicates **{', '.join([i[0] for i in intent_results])}**."
            )

        if emotion_results:
            summary.append(
                f"The emotional tone reflects **{', '.join([e[0] for e in emotion_results])}**."
            )

        st.markdown(" ".join(summary))

    else:
        st.warning("No strong intent or emotion detected.")

# =================================================
# Footer
# =================================================
st.divider()
st.caption(
    "Built with Classical NLP • Hybrid Intent Detection • TF-IDF • Multi-Label Learning • Streamlit"
)
