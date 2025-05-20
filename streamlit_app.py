import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="AI-Powered Text Similarity Analyzer", layout="wide")

# Load Tailwind CSS and Font Awesome for icons via CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hover-tooltip:hover .tooltip-text { display: block; }
        .tooltip-text { display: none; position: absolute; background-color: #1f2937; color: white; padding: 8px; border-radius: 4px; font-size: 12px; z-index: 10; }
        .animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model and TF-IDF vectorizer
try:
    with open('best_knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 Model or vectorizer file not found. Please ensure 'best_knn_model.pkl' and 'tfidf_vectorizer.pkl' are available.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error loading model or vectorizer: {str(e)}")
    st.stop()

# Initialize StandardScaler
scaler = StandardScaler(with_mean=False)

# Preprocessing function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

# Function to calculate word count
def get_word_count(text):
    return len(text.split())

# Function to calculate similarity
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
        st.error(f"🚨 Error calculating similarity: {str(e)}")
        return None

# Header with engaging title and description
st.markdown("""
    <div class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-8 rounded-xl shadow-2xl text-center">
        <h1 class="text-4xl font-extrabold mb-2">AI-Powered Text Similarity Analyzer</h1>
        <p class="text-lg">Harness advanced AI to compare texts with precision. Choose between our custom KNN model or Cosine Similarity for cutting-edge analysis.</p>
    </div>
""", unsafe_allow_html=True)

# Info section about methods
with st.expander("🔍 Learn About Similarity Methods", expanded=False):
    st.markdown("""
        <div class="p-4 bg-gray-50 rounded-lg">
            <h3 class="text-lg font-semibold text-gray-800">How It Works</h3>
            <p class="text-sm text-gray-600">
                <strong>KNN Model:</strong> Our custom K-Nearest Neighbors model leverages TF-IDF features to predict text similarity with high accuracy, trained on diverse datasets for robust performance.<br>
                <strong>Cosine Similarity:</strong> A classic metric that measures the cosine of the angle between two TF-IDF vectors, ideal for capturing semantic similarity in texts.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Input form with modern design
with st.form(key='text_similarity_form'):
    st.markdown('<div class="p-6 bg-white rounded-xl shadow-lg">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        text1 = st.text_area(
            "📝 Text 1",
            height=150,
            placeholder="Enter your first text here...",
            help="Type or paste the first text to compare."
        )
        st.markdown(f'<p class="text-sm text-gray-500">Word Count: <span class="font-semibold">{get_word_count(text1)}</span></p>', unsafe_allow_html=True)
    
    with col2:
        text2 = st.text_area(
            "📝 Text 2",
            height=150,
            placeholder="Enter your second text here...",
            help="Type or paste the second text to compare."
        )
        st.markdown(f'<p class="text-sm text-gray-500">Word Count: <span class="font-semibold">{get_word_count(text2)}</span></p>', unsafe_allow_html=True)
    
    # Method selection with tooltip
    st.markdown("""
        <div class="hover-tooltip relative">
            <span class="text-sm text-gray-600">Select Similarity Method:</span>
            <span class="tooltip-text">Choose KNN for AI-driven predictions or Cosine for traditional similarity metrics.</span>
        </div>
    """, unsafe_allow_html=True)
    method = st.radio("", ["KNN Model", "Cosine Similarity"], horizontal=True, label_visibility="collapsed")
    
    submit_button = st.form_submit_button(
        "🚀 Calculate Similarity",
        use_container_width=True,
        type="primary"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Process submission
if submit_button:
    if not text1.strip() or not text2.strip():
        st.error("⚠️ Please enter both texts to calculate similarity.")
    else:
        with st.spinner("Analyzing texts..."):
            similarity_score = get_similarity(text1, text2, method=method)
        
        if similarity_score is not None:
            # Display results with animation
            st.markdown("""
                <div class="mt-6 p-6 bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl shadow-lg animate-pulse">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-chart-line mr-2 text-indigo-600"></i> Analysis Results
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="p-4 bg-white rounded-lg shadow-md">
                            <h3 class="text-lg font-semibold text-gray-700">Text 1</h3>
                            <p class="mt-2 text-sm text-gray-600 line-clamp-3">{}</p>
                            <p class="mt-2 text-sm text-gray-600"><strong>Word Count:</strong> {}</p>
                        </div>
                        <div class="p-4 bg-white rounded-lg shadow-md">
                            <h3 class="text-lg font-semibold text-gray-700">Text 2</h3>
                            <p class="mt-2 text-sm text-gray-600 line-clamp-3">{}</p>
                            <p class="mt-2 text-sm text-gray-600"><strong>Word Count:</strong> {}</p>
                        </div>
                    </div>
                    <div class="mt-6 p-4 bg-indigo-50 rounded-lg text-center">
                        <h3 class="text-lg font-semibold text-indigo-800">Similarity Score ({})</h3>
                        <p class="mt-2 text-indigo-600 text-3xl font-bold">{:.2f}</p>
                    </div>
                </div>
            """.format(text1, get_word_count(text1), text2, get_word_count(text2), method, similarity_score), unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="mt-8 p-4 bg-gray-800 text-white text-center rounded-lg">
        <p class="text-sm">Powered by <strong>xAI</strong> | Built with expertise in AI-driven text analysis</p>
    </div>
""", unsafe_allow_html=True)