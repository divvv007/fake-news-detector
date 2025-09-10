import joblib
import re

# --- Load saved model and vectorizer ---
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# --- Preprocess text ---
def clean_text(text: str) -> str:
    """Lowercase, remove special chars/numbers, and strip spaces."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters and spaces
    return text.strip()

# --- Predict function ---
def predict_news(text: str):
    """Predict whether news is FAKE or REAL with confidence score."""
    processed_text = clean_text(text)
    
    if not processed_text:  # handle empty input
        return "UNKNOWN", 0.0

    vector = vectorizer.transform([processed_text])
    
    prediction = model.predict(vector)[0]
    confidence = float(max(model.predict_proba(vector)[0]))  # ensure float
    
    # Map model's numeric labels to FAKE/REAL dynamically
    label_map = dict(zip(model.classes_, ["FAKE", "REAL"]))
    label = label_map.get(prediction, str(prediction))
    
    return label, confidence
