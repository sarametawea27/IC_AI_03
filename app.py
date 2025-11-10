# app.py
import streamlit as st
import pandas as pd
import numpy as np
import ast, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------- Load your Kaggle-prepared CSVs ----------------
MOVIES_CSV = 'tmdb_5000_movies.csv'
CREDITS_CSV = 'tmdb_5000_credits.csv'

# Load & preprocess your data here
# You can copy the full preprocessing and TF-IDF code from your Kaggle notebook
# This should include creating tfidf_matrix and cosine_sim
# Example placeholders:
# df = ...  # merged movies + credits dataframe
# tfidf_matrix = ...  # TF-IDF vectorized "soup"
# cosine_sim = ...    # cosine similarity matrix

# ---------------- Streamlit UI ----------------
st.title("🎬 Movie Recommendation System")

st.write("Enter a movie title and get the top recommendations!")

# Input box for the movie title
movie_input = st.text_input("Movie title:")

# Slider to select number of recommendations
top_n = st.slider("Number of recommendations:", min_value=1, max_value=15, value=5)

# Button to trigger recommendations
if st.button("Recommend"):
    if movie_input:
        try:
            # Call the recommendation function
            recommendations = recommend_by_title(movie_input, top_n=top_n)
            
            # If no recommendations found
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                # Display each recommended movie
                for i, row in recommendations.iterrows():
                    st.subheader(f"{i+1}. {row['title']} ({row['release_date']})")
                    st.write(f"Rating: {row['vote_average']} | Votes: {row['vote_count']}")
                    st.write(row['overview'])
                    st.write("---")
        except Exception as e:
            st.error(f"Error: {e}")
