"""
Streamlit Web App — Hybrid Movie Recommendation System.

Interactive demo with:
- Movie search and similar movie recommendations
- User-based personalized recommendations
- Algorithm picker (Content-Based / Collaborative / Hybrid)
- Explainability panel
- Model comparison dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_movies, load_ratings, load_tags, get_dataset_stats
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender, compare_algorithms
from src.hybrid import HybridRecommender

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .movie-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    .movie-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }

    .movie-genres {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #252540);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    .stSidebar {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data & Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load all datasets."""
    data_dir = os.path.join(PROJECT_ROOT, "data", "ml-latest-small")
    if not os.path.exists(data_dir):
        msg_placeholder = st.empty()
        with msg_placeholder.container():
            st.info("⚠️ Dataset not found! Downloading automatically...")
            from data.download_data import download_dataset
            with st.spinner("Downloading dataset... (This only happens once)"):
                download_dataset()
        msg_placeholder.empty()
    movies = load_movies(data_dir)
    ratings = load_ratings(data_dir)
    tags = load_tags(data_dir)
    return movies, ratings, tags


@st.cache_resource
def build_content_engine(movies_df, tags_df):
    """Build and cache the content-based recommender."""
    engine = ContentBasedRecommender()
    engine.fit(movies_df, tags_df)
    return engine


@st.cache_resource
def build_collab_engine(ratings_df, algo_name):
    """Build and cache a collaborative filtering recommender."""
    engine = CollaborativeRecommender(algo_name)
    engine.fit(ratings_df)
    return engine


@st.cache_resource
def build_hybrid_engine(ratings_df, movies_df, tags_df, algo_name):
    """Build and cache the hybrid recommender."""
    engine = HybridRecommender(alpha=0.3, collab_algo=algo_name)
    engine.fit(ratings_df, movies_df, tags_df)
    return engine


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
movies, ratings, tags = load_data()
stats = get_dataset_stats(movies, ratings, tags)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    page = st.radio(
        "Navigate",
        ["🎬 Movie Recommendations", "📊 Model Comparison", "📈 Dataset Explorer", "📖 Algorithm Guide"],
        index=0,
    )

    st.markdown("---")

    algo_choice = st.selectbox(
        "Collaborative Algorithm",
        ["svd", "knn", "nmf"],
        format_func=lambda x: {"svd": "SVD", "knn": "KNN Basic", "nmf": "NMF"}[x],
        index=0,
    )

    alpha = st.slider(
        "Hybrid Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="α=0: pure collaborative, α=1: pure content-based",
    )

    n_recs = st.slider("Number of Recommendations", 5, 20, 10)

    st.markdown("---")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{stats['n_ratings']:,}</div>
            <div class="metric-label">Total Ratings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Users", f"{stats['n_users']:,}")
    with col2:
        st.metric("Movies", f"{stats['n_movies']:,}")

    st.markdown("---")
    st.caption("Built with Streamlit • MovieLens Dataset • Free & Open Source")


# ─────────────────────────────────────────────
# Page: Movie Recommendations
# ─────────────────────────────────────────────
if page == "🎬 Movie Recommendations":
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🎬 Movie Recommender</h1>
            <p>Hybrid recommendation system combining content-based and collaborative filtering</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["🔍 Find Similar Movies", "👤 Personalized Recommendations"])

    # ── Tab 1: Similar Movies ──
    with tab1:
        st.markdown('<div class="section-header">Search for a Movie</div>', unsafe_allow_html=True)

        search_query = st.text_input(
            "Type a movie name",
            placeholder="e.g., Toy Story, The Matrix, Inception...",
            key="movie_search",
        )

        if search_query:
            # Fuzzy search
            matches = movies[
                movies["title"].str.contains(search_query, case=False, na=False)
            ].head(20)

            if len(matches) == 0:
                st.warning("No movies found. Try a different search term.")
            else:
                selected_movie = st.selectbox(
                    "Select a movie",
                    matches["movieId"].tolist(),
                    format_func=lambda mid: movies[movies["movieId"] == mid]["title"].values[0],
                    key="movie_select",
                )

                if selected_movie:
                    movie_info = movies[movies["movieId"] == selected_movie].iloc[0]
                    st.markdown(f"**Selected:** {movie_info['title']} — *{movie_info['genres']}*")

                    with st.spinner("Finding similar movies..."):
                        content_engine = build_content_engine(movies, tags)
                        similar = content_engine.recommend_similar(selected_movie, n=n_recs)

                    if len(similar) > 0:
                        st.markdown(
                            '<div class="section-header">Similar Movies</div>',
                            unsafe_allow_html=True,
                        )
                        for _, row in similar.iterrows():
                            st.markdown(
                                f"""
                                <div class="movie-card">
                                    <div class="movie-title">{row['title']}</div>
                                    <div class="movie-genres">{row['genres']}</div>
                                    <span class="score-badge">
                                        Similarity: {row['similarity_score']:.2%}
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No similar movies found for this selection.")

    # ── Tab 2: Personalized ──
    with tab2:
        st.markdown(
            '<div class="section-header">Personalized Recommendations</div>',
            unsafe_allow_html=True,
        )

        # User selection
        user_ids = sorted(ratings["userId"].unique())
        selected_user = st.selectbox(
            "Select a User ID",
            user_ids,
            index=0,
            key="user_select",
        )

        # Show user's rating history
        user_ratings = ratings[ratings["userId"] == selected_user].merge(
            movies[["movieId", "title", "genres"]], on="movieId"
        )

        with st.expander(f"📋 User {selected_user}'s Rating History ({len(user_ratings)} ratings)"):
            display_cols = ["title", "genres", "rating"]
            st.dataframe(
                user_ratings[display_cols]
                .sort_values("rating", ascending=False)
                .head(20)
                .reset_index(drop=True),
                use_container_width=True,
            )

        # Algorithm selection
        rec_method = st.radio(
            "Recommendation Method",
            ["Hybrid", "Content-Based Only", "Collaborative Only"],
            horizontal=True,
            key="rec_method",
        )

        if st.button("🚀 Get Recommendations", type="primary", key="get_recs"):
            with st.spinner("Generating recommendations..."):
                if rec_method == "Hybrid":
                    hybrid_engine = build_hybrid_engine(
                        ratings, movies, tags, algo_choice
                    )
                    recs = hybrid_engine.recommend(
                        selected_user, n=n_recs, alpha=alpha
                    )
                    score_col = "hybrid_score"

                elif rec_method == "Content-Based Only":
                    content_engine = build_content_engine(movies, tags)
                    recs = content_engine.recommend_for_user(
                        selected_user, ratings, n=n_recs
                    )
                    score_col = "content_score"

                else:
                    collab_engine = build_collab_engine(ratings, algo_choice)
                    recs = collab_engine.recommend_for_user(
                        selected_user, ratings, movies, n=n_recs
                    )
                    score_col = "predicted_rating"

            if len(recs) > 0:
                st.markdown(
                    f'<div class="section-header">Top {len(recs)} Recommendations</div>',
                    unsafe_allow_html=True,
                )

                for rank, (_, row) in enumerate(recs.iterrows(), 1):
                    score = row[score_col]
                    score_label = {
                        "hybrid_score": "Hybrid Score",
                        "content_score": "Content Score",
                        "predicted_rating": "Predicted Rating",
                    }.get(score_col, "Score")

                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div class="movie-title">#{rank} {row['title']}</div>
                                    <div class="movie-genres">{row['genres']}</div>
                                </div>
                                <span class="score-badge">{score_label}: {score:.3f}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Show explanation for hybrid
                if rec_method == "Hybrid" and len(recs) > 0:
                    with st.expander("🔍 Why these recommendations?"):
                        hybrid_engine = build_hybrid_engine(
                            ratings, movies, tags, algo_choice
                        )
                        top_movie_id = recs.iloc[0]["movieId"]
                        explanation = hybrid_engine.explain(
                            selected_user, int(top_movie_id)
                        )
                        st.json(explanation)
            else:
                st.info("No recommendations found for this user.")


# ─────────────────────────────────────────────
# Page: Model Comparison
# ─────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown(
        """
        <div class="main-header">
            <h1>📊 Model Comparison</h1>
            <p>Cross-validation comparison of KNN, SVD, and NMF algorithms</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔬 Run Cross-Validation (5-fold)", type="primary"):
        with st.spinner("Running cross-validation... This may take a minute."):
            comparison = compare_algorithms(ratings, cv=5)

        st.markdown(
            '<div class="section-header">Results</div>',
            unsafe_allow_html=True,
        )

        # Display table
        st.dataframe(
            comparison.style.highlight_min(
                subset=["rmse_mean", "mae_mean"], color="#2d5a3d"
            ),
            use_container_width=True,
        )

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0e1117")

        colors = ["#667eea", "#764ba2", "#f093fb"]

        # RMSE comparison
        axes[0].bar(
            comparison["algorithm"],
            comparison["rmse_mean"],
            yerr=comparison["rmse_std"],
            color=colors,
            capsize=5,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[0].set_title("RMSE (lower is better)", color="white", fontweight="bold")
        axes[0].set_facecolor("#0e1117")
        axes[0].tick_params(colors="white")
        axes[0].spines["bottom"].set_color("white")
        axes[0].spines["left"].set_color("white")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # MAE comparison
        axes[1].bar(
            comparison["algorithm"],
            comparison["mae_mean"],
            yerr=comparison["mae_std"],
            color=colors,
            capsize=5,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].set_title("MAE (lower is better)", color="white", fontweight="bold")
        axes[1].set_facecolor("#0e1117")
        axes[1].tick_params(colors="white")
        axes[1].spines["bottom"].set_color("white")
        axes[1].spines["left"].set_color("white")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
            > **Key Insights:**
            > - **SVD** typically achieves the lowest RMSE/MAE due to effective latent factor modeling
            > - **KNN** is interpretable but can be slower on large datasets
            > - **NMF** enforces non-negativity which can improve interpretability
            """
        )
    else:
        st.info("Click the button above to run cross-validation. It takes about 30-60 seconds.")


# ─────────────────────────────────────────────
# Page: Dataset Explorer
# ─────────────────────────────────────────────
elif page == "📈 Dataset Explorer":
    st.markdown(
        """
        <div class="main-header">
            <h1>📈 Dataset Explorer</h1>
            <p>Exploratory data analysis of the MovieLens dataset</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset stats
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Ratings", f"{stats['n_ratings']:,}")
    with cols[1]:
        st.metric("Unique Users", f"{stats['n_users']:,}")
    with cols[2]:
        st.metric("Unique Movies", f"{stats['n_movies']:,}")
    with cols[3]:
        st.metric("Avg Rating", f"{stats['avg_rating']:.2f}")

    col1, col2 = st.columns(2)

    # Rating distribution
    with col1:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        rating_counts = ratings["rating"].value_counts().sort_index()
        bars = ax.bar(
            rating_counts.index.astype(str),
            rating_counts.values,
            color="#667eea",
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 200,
                f"{int(height):,}",
                ha="center",
                color="white",
                fontsize=9,
            )

        ax.set_xlabel("Rating", color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_title("Distribution of Ratings", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # Genre distribution
    with col2:
        st.markdown('<div class="section-header">Top Genres</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        all_genres = [g for gl in movies["genre_list"] for g in gl]
        genre_counts = pd.Series(all_genres).value_counts().head(15)

        bars = ax.barh(
            genre_counts.index[::-1],
            genre_counts.values[::-1],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, 15)),
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xlabel("Number of Movies", color="white")
        ax.set_title("Top 15 Genres", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # Ratings over time
    st.markdown('<div class="section-header">Rating Activity Over Time</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ratings_monthly = ratings.set_index("datetime").resample("M").size()
    ax.fill_between(
        ratings_monthly.index,
        ratings_monthly.values,
        alpha=0.4,
        color="#667eea",
    )
    ax.plot(ratings_monthly.index, ratings_monthly.values, color="#764ba2", linewidth=1.5)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Number of Ratings", color="white")
    ax.set_title("Monthly Rating Activity", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # User activity distribution
    st.markdown('<div class="section-header">User Activity</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        user_activity = ratings.groupby("userId").size()
        ax.hist(
            user_activity, bins=50, color="#667eea",
            edgecolor="white", linewidth=0.5, alpha=0.8,
        )
        ax.set_xlabel("Number of Ratings", color="white")
        ax.set_ylabel("Number of Users", color="white")
        ax.set_title("Ratings per User Distribution", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        movie_popularity = ratings.groupby("movieId").size()
        ax.hist(
            movie_popularity, bins=50, color="#764ba2",
            edgecolor="white", linewidth=0.5, alpha=0.8,
        )
        ax.set_xlabel("Number of Ratings", color="white")
        ax.set_ylabel("Number of Movies", color="white")
        ax.set_title("Ratings per Movie Distribution", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────
# Page: Algorithm Guide
# ─────────────────────────────────────────────
elif page == "📖 Algorithm Guide":
    st.markdown(
        """
        <div class="main-header">
            <h1>📖 Algorithm Guide</h1>
            <p>Plain-English explanations of every algorithm and metric used in this app</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Algorithms ──
    st.markdown('<div class="section-header">🤖 Recommendation Algorithms</div>', unsafe_allow_html=True)
    st.markdown(
        "This app uses **five approaches** to recommend movies. "
        "Each has different strengths — the Hybrid combines the best of all."
    )

    algorithms = [
        {
            "name": "Content-Based Filtering",
            "badge": "TF-IDF + Cosine Similarity",
            "color": "#667eea",
            "icon": "🎭",
            "what": "Recommends movies that are <b>similar in content</b> to movies you have already liked — based on genres and user-written tags.",
            "how": "Each movie is converted into a <b>TF-IDF vector</b> (a list of numbers representing its genres/tags). Movies whose vectors point in the same direction score a high <b>cosine similarity</b> (closer to 1.0 = more similar, 0 = totally different).",
            "strength": "Works even for brand-new users with no rating history — no cold-start problem.",
            "weakness": "Can only recommend movies similar to what you already know — limited serendipity.",
        },
        {
            "name": "KNN Basic (K-Nearest Neighbors)",
            "badge": "Memory-Based Collaborative",
            "color": "#f093fb",
            "icon": "👥",
            "what": "Finds the <b>K most similar users</b> to you based on shared ratings, then recommends movies those users loved that you have not seen yet.",
            "how": "Similarity between users is computed using <b>cosine similarity</b> on their rating vectors. The predicted rating for a movie is a weighted average of the K neighbours' ratings for that movie.",
            "strength": "Simple, transparent, and highly interpretable — you can see exactly which similar users drove the recommendation.",
            "weakness": "Slow on large datasets (must compare against all users). Struggles when the rating matrix is very sparse.",
        },
        {
            "name": "SVD (Singular Value Decomposition)",
            "badge": "Model-Based Collaborative — Best Accuracy",
            "color": "#43e97b",
            "icon": "🧮",
            "what": "Learns <b>hidden taste profiles</b> for every user and hidden <b>feature profiles</b> for every movie, then predicts ratings by matching them.",
            "how": "The user-movie rating matrix is factorised into latent factors — abstract concepts like 'prefers slow-burn dramas' or 'loves 80s comedies' — without being explicitly defined. Predicted rating = user-factor vector · movie-factor vector.",
            "strength": "Best rating-prediction accuracy. Fast at inference time. Handles sparsity well through dimensionality reduction.",
            "weakness": "Less interpretable — the latent factors have no human-readable label.",
        },
        {
            "name": "NMF (Non-Negative Matrix Factorization)",
            "badge": "Model-Based Collaborative",
            "color": "#f9844a",
            "icon": "📐",
            "what": "Like SVD, but with one key constraint: <b>all learned values must be >= 0</b>. This forces additive, part-based representations.",
            "how": "Because factors can only add (never cancel each other), they tend to align with interpretable concepts such as genre clusters. A user profile might look like '40% Action, 35% Sci-Fi, 25% Thriller'.",
            "strength": "More interpretable than SVD — factors often correspond to recognisable genres or themes.",
            "weakness": "Slightly lower accuracy than SVD due to the non-negativity constraint.",
        },
        {
            "name": "Hybrid Recommender",
            "badge": "Weighted Combination",
            "color": "#764ba2",
            "icon": "🔀",
            "what": "Blends <b>Content-Based</b> and <b>Collaborative Filtering</b> scores into a single recommendation score, getting the best of both worlds.",
            "how": "<code>hybrid_score = alpha x content_score + (1 - alpha) x collab_score</code><br>Both scores are first normalised to [0, 1]. The <b>alpha slider</b> in the sidebar controls the balance: alpha=0 is pure collaborative, alpha=1 is pure content-based.",
            "strength": "More robust: collaborative filtering is strong for active users; content-based fills in gaps for new users (cold-start). Also provides explainability.",
            "weakness": "Slightly more complex to tune — alpha needs to be adjusted per use-case.",
        },
    ]

    for algo in algorithms:
        st.markdown(
            f"""
            <div class="movie-card" style="border-left: 4px solid {algo['color']}; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.7rem;">
                    <span style="font-size: 1.8rem;">{algo['icon']}</span>
                    <div>
                        <div class="movie-title" style="font-size: 1.15rem;">{algo['name']}</div>
                        <span style="background: {algo['color']}22; color: {algo['color']}; padding: 0.15rem 0.6rem;
                               border-radius: 12px; font-size: 0.78rem; font-weight: 600;">
                            {algo['badge']}
                        </span>
                    </div>
                </div>
                <table style="width: 100%; border-collapse: collapse; color: #cbd5e1; font-size: 0.9rem;">
                    <tr>
                        <td style="padding: 0.35rem 0.6rem; vertical-align: top; white-space: nowrap;
                                   color: #94a3b8; width: 110px;"><b>What</b></td>
                        <td style="padding: 0.35rem 0.6rem;">{algo['what']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.35rem 0.6rem; vertical-align: top; color: #94a3b8;"><b>How</b></td>
                        <td style="padding: 0.35rem 0.6rem;">{algo['how']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.35rem 0.6rem; vertical-align: top; color: #4ade80;"><b>Strength</b></td>
                        <td style="padding: 0.35rem 0.6rem; color: #86efac;">{algo['strength']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.35rem 0.6rem; vertical-align: top; color: #f87171;"><b>Weakness</b></td>
                        <td style="padding: 0.35rem 0.6rem; color: #fca5a5;">{algo['weakness']}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Metrics ──
    st.markdown('<div class="section-header">📏 Evaluation Metrics Explained</div>', unsafe_allow_html=True)
    st.markdown(
        "These numbers measure how good the recommendations are. "
        "They are computed using **5-fold cross-validation** — the dataset is split 5 ways "
        "so every rating is used for testing exactly once."
    )

    metrics = [
        {
            "name": "RMSE — Root Mean Squared Error",
            "icon": "📉",
            "color": "#667eea",
            "formula": "sqrt( sum( (predicted_rating - actual_rating)^2 ) / n )",
            "meaning": "The average size of rating-prediction errors, in <b>star units</b> (scale: 0.5 to 5.0 stars). Errors are squared before averaging, so large mistakes are penalised more heavily than small ones.",
            "example": "RMSE = 0.87 means predictions are typically off by roughly <b>+/- 0.87 stars</b>.",
            "direction": "Lower is better",
            "dir_color": "#4ade80",
        },
        {
            "name": "MAE — Mean Absolute Error",
            "icon": "📉",
            "color": "#f093fb",
            "formula": "sum( |predicted_rating - actual_rating| ) / n",
            "meaning": "Like RMSE but errors are not squared, so all mistakes are penalised equally. MAE is easier to interpret directly as a typical star-error in everyday terms.",
            "example": "MAE = 0.67 means the typical prediction is off by <b>+/- 0.67 stars</b>.",
            "direction": "Lower is better",
            "dir_color": "#4ade80",
        },
        {
            "name": "Precision@K",
            "icon": "🎯",
            "color": "#43e97b",
            "formula": "(Relevant items in top-K recommendations) / K",
            "meaning": "Of the top <b>K</b> movies the system shows, what <b>fraction</b> did the user actually like? A movie counts as 'relevant' if the user's true rating is >= 3.5 stars.",
            "example": "Precision@10 = 0.40 means <b>4 out of 10</b> recommended movies were ones the user genuinely liked.",
            "direction": "Higher is better",
            "dir_color": "#4ade80",
        },
        {
            "name": "Recall@K",
            "icon": "🔍",
            "color": "#f9844a",
            "formula": "(Relevant items in top-K) / (Total relevant items for this user)",
            "meaning": "Of <b>all</b> movies the user would have enjoyed, what fraction did the system successfully surface in its top-K list? Recall captures how much of the user's taste profile was covered.",
            "example": "Recall@10 = 0.25 means the system found <b>25% of all movies</b> the user would have liked.",
            "direction": "Higher is better",
            "dir_color": "#4ade80",
        },
        {
            "name": "Coverage@K",
            "icon": "🗺️",
            "color": "#764ba2",
            "formula": "(Unique items recommended across all users) / (Total catalog size) x 100%",
            "meaning": "The percentage of the <b>full movie catalog</b> that the system ever recommends to at least one user. Low coverage means the system always recommends the same popular blockbusters.",
            "example": "Coverage = 35% means the system draws from <b>3,410 of the 9,742 movies</b>, not just top-rated ones.",
            "direction": "Higher = more diverse",
            "dir_color": "#4ade80",
        },
    ]

    for metric in metrics:
        st.markdown(
            f"""
            <div class="movie-card" style="border-left: 4px solid {metric['color']}; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <span style="font-size: 1.5rem;">{metric['icon']}</span>
                        <span class="movie-title" style="font-size: 1.05rem;">{metric['name']}</span>
                    </div>
                    <span style="background: {metric['dir_color']}22; color: {metric['dir_color']};
                           padding: 0.2rem 0.7rem; border-radius: 12px; font-size: 0.8rem; font-weight: 700;">
                        {metric['direction']}
                    </span>
                </div>
                <div style="font-family: monospace; background: rgba(255,255,255,0.06); padding: 0.4rem 0.8rem;
                            border-radius: 6px; color: {metric['color']}; font-size: 0.85rem; margin-bottom: 0.5rem;">
                    {metric['formula']}
                </div>
                <div style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.35rem;">{metric['meaning']}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    <b style="color: #e2e8f0;">Example:</b> {metric['example']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Quick Reference Table ──
    st.markdown('<div class="section-header">⚡ Quick Reference Card</div>', unsafe_allow_html=True)
    ref_data = {
        "Algorithm": ["Content-Based", "KNN Basic", "SVD", "NMF", "Hybrid"],
        "Type": ["Content", "Memory-Based CF", "Model-Based CF", "Model-Based CF", "Combined"],
        "Best For": [
            "New users, item similarity queries",
            "Interpretability, small datasets",
            "Best prediction accuracy",
            "Explainable latent factors",
            "All-round performance",
        ],
        "Handles Cold Start": ["Yes", "No", "No", "No", "Partial (via content-based)"],
        "Speed": ["Fast", "Slow (O(n) scan)", "Fast (after training)", "Medium", "Medium"],
        "Interpretable": ["Yes", "Yes", "Partial", "Yes", "Yes (via explain())"],
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)
