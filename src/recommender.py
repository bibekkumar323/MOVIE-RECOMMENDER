from __future__ import annotations
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

def _normalize_genres(genres: str) -> str:
    if pd.isna(genres) or genres == "(no genres listed)":
        return ""
    parts = [g.strip().lower() for g in genres.split("|") if g]
    return " ".join(parts + parts)

def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()

@dataclass
class ContentRecommender:
    movies: pd.DataFrame = None
    vectorizer: TfidfVectorizer = None
    tfidf_matrix: np.ndarray = None
    title_index: pd.Series = None

    def fit(self, movies: pd.DataFrame):
        df = movies.copy()
        df["title"] = df["title"].map(_clean_title)
        df["genres_norm"] = df["genres"].map(_normalize_genres)
        df["text_soup"] = df["title"].fillna("") + " " + df["genres_norm"].fillna("")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(df["text_soup"])
        self.movies, self.title_index = df, pd.Series(df.index, index=df["title"].str.lower(), name="row")
        return self

    def recommend_by_title(self, title: str, topn: int = 10) -> pd.DataFrame:
        if self.movies is None or self.tfidf_matrix is None:
            raise RuntimeError("Model not fitted.")
        choices = list(self.title_index.index)
        match, score, _ = process.extractOne(title.lower(), choices, scorer=fuzz.WRatio)
        if score < 60:
            raise ValueError(f"No close match for '{title}'.")
        idx = int(self.title_index.loc[match])
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).ravel()
        sims[idx] = -1.0
        top_idx = np.argpartition(-sims, range(topn))[:topn]
        top_sorted = top_idx[np.argsort(-sims[top_idx])]
        out = self.movies.loc[top_sorted, ["movieId", "title", "genres"]].copy()
        out.insert(3, "similarity", np.round(sims[top_sorted], 4))
        return out.reset_index(drop=True)

    def recommend_by_keywords(self, query: str, topn: int = 10) -> pd.DataFrame:
        if self.movies is None or self.tfidf_matrix is None or self.vectorizer is None:
            raise RuntimeError("Model not fitted.")
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.tfidf_matrix).ravel()
        top_idx = np.argpartition(-sims, range(topn))[:topn]
        top_sorted = top_idx[np.argsort(-sims[top_idx])]
        out = self.movies.loc[top_sorted, ["movieId", "title", "genres"]].copy()
        out.insert(3, "similarity", np.round(sims[top_sorted], 4))
        return out.reset_index(drop=True)
