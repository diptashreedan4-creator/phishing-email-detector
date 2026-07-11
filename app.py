import streamlit as st
import joblib
import re
import string
import numpy as np
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
rf_model = joblib.load('rf_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ AI-Driven Phishing Email Detector")
st.write("Paste an email below to check if it's **Safe** or **Phishing**, using NLP + Machine Learning.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def extract_metadata(text):
    has_url = int(bool(re.search(r'http\S+|www\S+', text)))
    num_exclamations = text.count('!')
    email_length = len(text)
    num_words = len(text.split())
    return np.array([[has_url, num_exclamations, email_length, num_words]])

email_input = st.text_area("Email text", height=250, placeholder="Paste the email content here...")

if st.button("Analyze Email"):
    if email_input.strip() == "":
        st.warning("Please paste an email first.")
    else:
        cleaned = clean_text(email_input)
        tfidf_features = tfidf.transform([cleaned])
        metadata_features = extract_metadata(email_input)
        final_features = hstack([tfidf_features, metadata_features])

        prediction = rf_model.predict(final_features)[0]
        probability = rf_model.predict_proba(final_features)[0]

        if prediction == 1:
            st.error(f"⚠️ This looks like a **PHISHING** email. (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ This looks like a **SAFE** email. (Confidence: {probability[0]*100:.1f}%)")

        with st.expander("See details"):
            st.write("**Has URL:**", bool(metadata_features[0][0]))
            st.write("**Exclamation marks:**", int(metadata_features[0][1]))
            st.write("**Email length (chars):**", int(metadata_features[0][2]))
            st.write("**Word count:**", int(metadata_features[0][3]))

st.markdown("---")
st.caption("Built as part of an AI-Driven Phishing Email Detection project using NLP. Model: Random Forest Classifier.")
