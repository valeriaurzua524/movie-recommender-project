from pathlib import Path
import argparse
import difflib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent
GENRE_WEIGHT = 3.0
RATING_WEIGHT = 0.20
PRIOR_RATINGS = 20 #Shrink low-sample movie averages toward the overall mean


def load_model(data_dir=DATA_DIR):
    """Load data and construct sparse genre + rating feature vectors."""
    try:
        movies = pd.read_csv(data_dir / 'movies.csv', usecols=['movieId', 'title', 'genres'])
        ratings = pd.read_csv(data_dir / 'ratings.csv', usecols=['movieId', 'rating'])
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f'Could not load MovieLens CSV files from {data_dir}: {exc}') from exc
    if movies.empty or ratings.empty:
        raise RuntimeError('Movie and rating CSV files must not be empty.')
    movies = movies.drop_duplicates('movieId').reset_index(drop=True)
    genre_text = (movies['genres'].fillna('').replace('(no genres listed)', '')
                  .str.replace('Sci-Fi', 'SciFi', regex=False)
                  .str.replace('Film-Noir', 'FilmNoir', regex=False)
                  .str.replace('|', ' ', regex=False))
    if not genre_text.str.strip().any():
        raise RuntimeError('No usable genres found in movies.csv.')
    genres = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b').fit_transform(genre_text)
    stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
    movies = movies.join(stats, on='movieId')
    overall_mean = float(ratings['rating'].mean())
    count = movies['count'].fillna(0).to_numpy()
    mean = movies['mean'].fillna(overall_mean).to_numpy()
    #Bayesian smoothing prevents a single five-star rating from dominating
    smoothed_mean = (count * mean + PRIOR_RATINGS * overall_mean) / (count + PRIOR_RATINGS)
    rating_data = np.column_stack([smoothed_mean, np.log1p(count)])
    rating_features = StandardScaler().fit_transform(rating_data)
    features = hstack([genres * GENRE_WEIGHT, csr_matrix(rating_features * RATING_WEIGHT)], format='csr')
    titles = movies['title'].astype(str).tolist()
    lookup = {title.casefold(): idx for idx, title in enumerate(titles)}
    return titles, lookup, features


def recommend(title, titles, lookup, features, n=5):
    """Return movie title and similarity pairs; suggest close titles on misses."""
    if n < 1:
        raise ValueError('n must be at least 1')
    query = title.strip().casefold()
    row = lookup.get(query)
    if row is None:
        matches = difflib.get_close_matches(query, list(lookup), n=5, cutoff=0.45)
        suggestions = [titles[lookup[match]] for match in matches]
        return [], suggestions
    scores = cosine_similarity(features[row], features).ravel()
    scores[row] = -np.inf
    best = np.argsort(scores)[::-1][:min(n, len(titles) - 1)]
    return [(titles[i], float(scores[i])) for i in best], []


def main():
    parser = argparse.ArgumentParser(description='Recommend similar MovieLens movies.')
    parser.add_argument('--movie', help='Movie title including year, e.g. "Toy Story (1995)"')
    parser.add_argument('--top', type=int, default=5, help='Number of recommendations (default: 5)')
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be at least 1')
    try:
        titles, lookup, features = load_model()
    except RuntimeError as exc:
        parser.exit(1, f'Error: {exc}\n')
    examples = ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 'GoldenEye (1995)']
    for movie in ([args.movie] if args.movie else examples):
        results, suggestions = recommend(movie, titles, lookup, features, args.top)
        print(f'\nInput movie: {movie}\nRecommendations:')
        if suggestions:
            print('  Movie not found. Did you mean:')
            for suggestion in suggestions:
                print(f'  - {suggestion}')
        elif not results:
            print('  No recommendations available.')
        else:
            for title, score in results:
                print(f'  - {title} (similarity={score:.3f})')


if __name__ == '__main__':
    main()
