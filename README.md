# 🎬 Hybrid Movie Recommendation System

A portfolio-grade recommendation system that combines **Content-Based Filtering** and **Collaborative Filtering** into a hybrid approach, built on the free MovieLens dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Live Demo

View the live project on Streamlit: [**Movie Recommendation System**](https://entity-movie-recommendation-system.streamlit.app/)

---

## 🏗️ Architecture

```
Content-Based (TF-IDF + Cosine Similarity)
          ↘
            → Hybrid Recommender → Top-N Recommendations
          ↗
Collaborative (KNN / SVD / NMF)
```

| Component | Description |
|-----------|-------------|
| **Content-Based** | TF-IDF vectorization on genres + tags → cosine similarity |
| **Collaborative** | KNN Basic, SVD, NMF via `surprise` library |
| **Hybrid** | Weighted combination: `α × content + (1-α) × collaborative` |
| **Evaluation** | RMSE, MAE, Precision@K, Recall@K, Coverage |

## 📊 Dataset

**MovieLens Latest Small** — 100,836 ratings from 610 users on 9,742 movies.

- Free for research & education ([GroupLens](https://grouplens.org/datasets/movielens/))
- Downloaded automatically when launching the Streamlit app, or via `data/download_data.py`

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/hybrid-movie-recommender.git
cd hybrid-movie-recommender
pip install -r requirements.txt
```

### 2. Download Data

```bash
python data/download_data.py
```

### 3. Run the Notebook

```bash
jupyter notebook notebooks/recommendation_system.ipynb
```

### 4. Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── data/
│   ├── download_data.py             # Auto-download MovieLens dataset
│   └── ml-latest-small/             # Dataset (auto-downloaded)
├── notebooks/
│   └── recommendation_system.ipynb  # Full walkthrough notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading & preprocessing
│   ├── content_based.py             # Content-based filtering engine
│   ├── collaborative.py             # Collaborative filtering (KNN/SVD/NMF)
│   ├── hybrid.py                    # Hybrid recommender
│   └── evaluation.py                # Metrics & evaluation
├── app/
│   └── streamlit_app.py             # Streamlit web app
└── models/                          # Saved models (gitignored)
```

## 🔬 Algorithms Compared

| Algorithm | Type | RMSE ↓ | MAE ↓ |
|-----------|------|--------|-------|
| KNN Basic | Memory-based | ~0.97 | ~0.75 |
| **SVD** | Model-based | **~0.87** | **~0.67** |
| NMF | Model-based | ~0.92 | ~0.72 |

> Results from 5-fold cross-validation on MovieLens Latest Small.  
> **↓ Lower is better** — SVD typically achieves the best accuracy.

### 🤖 What Do These Algorithms Mean?

| Algorithm | Full Name | How It Works |
|-----------|-----------|-------------|
| **KNN Basic** | The Neighbor Finder | Finds other people whose tastes perfectly match yours (your "neighbors") and recommends what they liked. Very transparent, but slow on large datasets. |
| **SVD** | The AI Matchmaker | Uncovers deep, abstract connections between movies—like "slow-paced films with dark atmospheres"—to figure out exactly what you like and dislike. Highest accuracy, but works like a black box. |
| **NMF** | The Recipe Builder | Figures out your movie taste by combining basic ingredients—like "3 parts Action, 2 parts Comedy"—to find the perfect movie for you. Highly intuitive, slightly less accurate than SVD. |
| **Content-Based** | TF-IDF + Cosine Similarity | Recommends movies similar in *content* (genres, tags) to what you've liked. Works even without other users' data. |
| **Hybrid** | Weighted Combination | Blends Content-Based and Collaborative scores: `score = α × content + (1−α) × collaborative`. Balances the strengths of both. |

### 📏 What Do the Metrics Mean?

| Metric | What It Measures | How to Read It |
|--------|-----------------|----------------|
| **RMSE** | Root Mean Squared Error — average prediction error in rating units (0.5–5.0 scale) | An RMSE of 0.87 means predictions are off by ~0.87 stars on average. **Lower = better.** |
| **MAE** | Mean Absolute Error — similar to RMSE but without squaring large errors | An MAE of 0.67 means the typical prediction error is ±0.67 stars. **Lower = better.** |
| **Precision@K** | Of the top-K recommendations shown, what fraction did the user actually like? | e.g. Precision@10 = 0.40 means 4 out of 10 recommendations were genuinely relevant. **Higher = better.** |
| **Recall@K** | Of all movies the user would have liked, what fraction appear in the top-K? | e.g. Recall@10 = 0.25 means the system surfaced 25% of all relevant movies. **Higher = better.** |
| **Coverage** | Percentage of the full movie catalog that ever gets recommended | Higher coverage = more diverse, less popularity-biased recommendations. |

## 🌐 Deployment

The Streamlit app can be deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app/streamlit_app.py`
5. Deploy!

## 💰 Cost

| Component | Cost |
|-----------|------|
| Dataset | Free |
| Python + Libraries | Free |
| Streamlit Cloud | Free |
| **Total** | **$0** |

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [GroupLens](https://grouplens.org/) for the MovieLens dataset
- [Surprise](https://surpriselib.com/) library for collaborative filtering
- [Streamlit](https://streamlit.io/) for the web app framework