import pandas as pd
import re
import pickle
import os
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return " ".join([word for word in text.split() if word not in stop_words])

# Load data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

# Balance dataset
min_len = min(len(fake), len(true))
fake = fake.sample(n=min_len, random_state=42)
true = true.sample(n=min_len, random_state=42)

df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# Combine title + text and clean
df["combined"] = (df["title"] + " " + df["text"]).apply(clean_text)

# Features and labels
X = df["combined"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(tfidf_train, y_train)

# Evaluate
y_pred = model.predict(tfidf_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
print("✅ Model and vectorizer saved.")
