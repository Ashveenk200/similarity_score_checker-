import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Text Similarity App", layout="centered")

# Load Tailwind CSS via CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Load trained models
try:
    with open('best_knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'best_knn_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {str(e)}")
    st.stop()

# Initialize scaler
scaler = StandardScaler(with_mean=False)

# Preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def get_word_count(text):
    return len(text.split())

# Similarity function
def get_similarity(text1, text2, method="KNN Model"):
    try:
        text1_processed = preprocess(text1)
        text2_processed = preprocess(text2)

        vec1 = tfidf.transform([text1_processed])
        vec2 = tfidf.transform([text2_processed])

        if method == "KNN Model":
            X = abs(vec1 - vec2)
            if X.nnz == 0:
                return 1.0
            X_scaled = scaler.fit_transform(X)
            similarity = knn_model.predict(X_scaled)[0]
        else:
            similarity = cosine_similarity(vec1, vec2)[0][0]

        return round(similarity, 2)
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return None

# Header UI
st.markdown("""
    <div class="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-4 rounded-lg shadow-md text-center">
        <h1 class="text-2xl font-bold">Text Similarity Calculator</h1>
        <p class="mt-1 text-sm">Enter two texts and choose a method to calculate their similarity score.</p>
    </div>
""", unsafe_allow_html=True)

# Input Form
with st.form(key='text_similarity_form'):
    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("First Text (Statement 1)", height=100, placeholder="Type your first text here...")
    with col2:
        text2 = st.text_area("Second Text (Statement 2)", height=100, placeholder="Type your second text here...")
    method = st.radio("Select Similarity Method:", ("KNN Model", "Cosine Similarity"), horizontal=True)
    submit_button = st.form_submit_button("Calculate Similarity")

# Process
if submit_button:
    if not text1 or not text2:
        st.error("Please enter both texts to calculate similarity.")
    else:
        similarity_score = get_similarity(text1, text2, method)
        if similarity_score is not None:
            word_count1 = get_word_count(text1)
            word_count2 = get_word_count(text2)

            # Styled result
            st.markdown(f"""
                <div class="mt-4 p-4 bg-gray-100 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold text-gray-800 mb-2">Results</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                        <div class="p-3 bg-white rounded-lg shadow-sm">
                            <h3 class="text-base font-medium text-gray-700">Statement 1</h3>
                            <p class="mt-1 text-sm text-gray-600">{text1}</p>
                            <p class="mt-1 text-sm text-gray-600"><strong>Word Count:</strong> {word_count1}</p>
                        </div>
                        <div class="p-3 bg-white rounded-lg shadow-sm">
                            <h3 class="text-base font-medium text-gray-700">Statement 2</h3>
                            <p class="mt-1 text-sm text-gray-600">{text2}</p>
                            <p class="mt-1 text-sm text-gray-600"><strong>Word Count:</strong> {word_count2}</p>
                        </div>
                    </div>
                    <div class="mt-4 p-3 bg-blue-50 rounded-lg text-center">
                        <h3 class="text-base font-semibold text-blue-800">Similarity Score ({method})</h3>
                        <p class="mt-1 text-blue-600 text-lg">{similarity_score:.2f}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # JSON Result
            json_result = {
                "text1": text1,
                "text2": text2,
                "similarity_score": similarity_score
            }

            st.markdown("---")
            st.subheader("JSON Output")
            st.json(json_result)
