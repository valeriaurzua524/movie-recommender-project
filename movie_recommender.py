from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent
#load the MovieLens small dataset
movies = pd.read_csv(DATA_DIR / "movies.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
movies = movies[["movieId", "title", "genres"]].copy()

#turns pipe-separated genres into text for TF-IDF
movies["genre_text"] = (
    movies["genres"]
    .fillna("")
    .replace("(no genres listed)", "")
    .str.replace("|", " ", regex=False)
    .str.replace("Sci-Fi", "SciFi", regex=False)
    .str.replace("Film-Noir", "FilmNoir", regex=False)
)
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
genre_features = vectorizer.fit_transform(movies["genre_text"])

#compute each movie's mean rating and number of ratings.
rating_stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
movies = movies.join(rating_stats, on="movieId")

#missing ratings receive the dataset-wide mean and zero rating count
global_mean = float(ratings["rating"].mean())
movies["mean"] = movies["mean"].fillna(global_mean)
movies["count"] = movies["count"].fillna(0)

#center rating features and scale them so they do not overpower genres
rating_values = np.column_stack([
    movies["mean"].to_numpy(),
    np.log1p(movies["count"].to_numpy()),
])
rating_features = StandardScaler().fit_transform(rating_values)
#genres have more weight than rating statistics in the similarity score
features = hstack([
    genre_features * 3.0,
    csr_matrix(rating_features * 0.35),
], format="csr")
#allow exact titles, including their release year
title_to_row = {
    title.lower(): i for i, title in enumerate(movies["title"])
}
def recommend(title, n=5):
    """Return up to n similar movies and their cosine similarity scores."""
    row = title_to_row.get(title.strip().lower())
    if row is None:
        return ["Movie not found. Please enter the full title with the year."]

    #compare only the selected movie with the catalog, not every movie pair.
    similarities = cosine_similarity(features[row], features).ravel()
    similarities[row] = -1  #dont recommend the input movie itself
    best_rows = np.argsort(similarities)[::-1][:n]

    return [
        f"{movies.iloc[i]['title']} (similarity={similarities[i]:.3f})"
        for i in best_rows
    ]

if __name__ == "__main__":
    test_movies = [
        "Toy Story (1995)",
        "Jumanji (1995)",
        "Heat (1995)",
        "GoldenEye (1995)",
    ]
    for movie in test_movies:
        print(f"\nInput movie: {movie}")
        print("Recommendations:")
        for result in recommend(movie, n=5):
            print(f"  - {result}")
