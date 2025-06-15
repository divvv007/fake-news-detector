import streamlit as st
from src.predict import predict_news
import time

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

# --- Modern Custom CSS ---
st.markdown("""
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .stApp {
            background: linear-gradient(to right, #a1c4fd, #c2e9fb);  /* Fresh Blue Gradient */
            font-family: 'Segoe UI', sans-serif;
            padding-top: 100px;  /* Space for navbar */
        }
        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 70px;
            background: #0d6efd;
            color: white;
            text-align: center;
            line-height: 70px;
            font-size: 28px;
            font-weight: 600;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 999;
        }
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            color: #222;
            text-align: center;
        }
        .sub-text {
            text-align: center;
            color: #333;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .result-box {
           background: linear-gradient(145deg, #b2fefa, #0ed2f7);
            color: #042a2b;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.12);
            margin-top: 20px;
        }
        .confidence {
            font-weight: bold;
            color: #003f5c;
        }
    </style>
""", unsafe_allow_html=True)

# --- Navbar (Sticky Header) ---
st.markdown('<div class="navbar">📰 Fake News Detector</div>', unsafe_allow_html=True)

# --- App Body Content ---
st.markdown("<h2 class='main-title'>🧠 Detect if News is Fake or Real</h2>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Paste any news headline or article to verify its authenticity using machine learning.</p>", unsafe_allow_html=True)

# --- Input ---
text = st.text_area("📝 Paste the news title and/or article:", height=200, placeholder="Type or paste your news content here...")

# --- Button ---
if st.button("🔍 Check Now"):
    if text.strip():
        with st.spinner("Analyzing the news..."):
            time.sleep(1)  # Optional delay
            label, confidence = predict_news(text)

        # --- Styled Result Box ---
        st.markdown(f"""
        <div class="result-box">
            <h3>📢 Prediction Result</h3>
            <p>The news is likely: <strong style="color: {'#ff3e4d' if label == 'FAKE' else '#008000'}">{label}</strong></p>
            <p class="confidence">Confidence: {round(confidence * 100, 2)}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some news content before checking.")
