# app.py
import streamlit as st
import pandas as pd
from movie_recommender import load_recommender  # or import pipeline builder
import pickle

@st.cache_resource
def load_model(path='recommender.pkl'):
    return load_recommender(path)

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

# Load recommender (make sure recommender.pkl exists; or call build pipeline)
recommender = load_model('recommender.pkl')

# Sidebar controls
st.sidebar.header("Find recommendations")
title_input = st.sidebar.text_input("Movie title (partial allowed)", "The Dark Knight")
top_k = st.sidebar.slider("Number of recommendations", 5, 30, 10)
use_cluster = st.sidebar.checkbox("Boost same-cluster results", value=True)
cluster_boost = st.sidebar.slider("Cluster boost factor", 0.0, 1.0, 0.2)

if st.sidebar.button("Recommend"):
    try:
        results = recommender.recommend_by_title(title_input, top_n=top_k, use_cluster_boost=use_cluster, cluster_boost=cluster_boost)
        st.subheader(f"Recommendations for: {title_input}")
        for i, row in results.iterrows():
            st.markdown(f"**{i+1}. {row['title']}** — score: {row['score']:.3f} — rating: {row.get('vote_average', 'N/A')} ({row.get('vote_count', 0)} votes)")
            st.write(row['overview'][:400] + ('...' if len(row['overview'])>400 else ''))
            st.markdown("---")
    except Exception as e:
        st.error(str(e))

# Option: recommend by liked movies (simple user profile)
st.sidebar.header("Or: Give me movies I like")
liked = st.sidebar.text_area("Enter a few movie titles you like, separated by commas", "Inception, Interstellar")
if st.sidebar.button("Recommend for profile"):
    liked_list = [t.strip() for t in liked.split(',') if t.strip()]
    # build a simple user vector by averaging TF-IDF vectors of liked movies
    idxs = []
    for t in liked_list:
        idx = recommender.get_index_from_title(t)
        if idx is not None:
            idxs.append(idx)
    if len(idxs) == 0:
        st.error("No liked titles found in the dataset.")
    else:
        import numpy as np
        user_vec = recommender.tfidf_matrix[idxs].mean(axis=0)
        recs = recommender.recommend_by_vector(user_vec, top_n=top_k)
        st.subheader("Recommendations based on your liked movies")
        for i, row in recs.iterrows():
            st.markdown(f"**{i+1}. {row['title']}** — score: {row['score']:.3f}")
            st.write(row['overview'][:300] + ('...' if len(row['overview'])>300 else ''))
            st.markdown("---")
