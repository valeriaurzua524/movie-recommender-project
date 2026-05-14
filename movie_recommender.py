import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#load data
movies=pd.read_csv("movies.csv")
ratings=pd.read_csv("ratings.csv")
#clean movie data
movies=movies[["movieId","title","genres"]].copy()
movies["genres"]=movies["genres"].fillna("").str.replace("|"," ",regex=False)
#make rating pattern table
rating_table=ratings.pivot_table(index="movieId",columns="userId",values="rating").fillna(0)
#normalize ratings by subtracting each movie average
movie_avg=rating_table.replace(0,np.nan).mean(axis=1)
rating_table_norm=rating_table.sub(movie_avg,axis=0).fillna(0)
#genre feature table
genre_table=movies.set_index("movieId")["genres"].str.get_dummies(sep=" ")
#make sure both tables use same movies
common_ids=rating_table_norm.index.intersection(genre_table.index)
rating_features=rating_table_norm.loc[common_ids]
genre_features=genre_table.loc[common_ids]
#combine genre features and rating patterns
all_features=pd.concat([genre_features,rating_features],axis=1)
#compare movies using cosine similarity
sim_matrix=cosine_similarity(all_features)
#map ids and titles
movie_ids=list(common_ids)
id_to_row={movie_id:i for i,movie_id in enumerate(movie_ids)}
title_to_id=pd.Series(movies.movieId.values,index=movies.title.str.lower()).drop_duplicates()
#recommendation function
def recommend(title,n=5):
    title=title.lower()
    if title not in title_to_id:
        return ["movie not found. please type the full title with the year."]
    movie_id=title_to_id[title]
    if movie_id not in id_to_row:
        return ["movie does not have enough data for recommendations."]
    row=id_to_row[movie_id]
    scores=list(enumerate(sim_matrix[row]))
    scores=sorted(scores,key=lambda x:x[1],reverse=True)[1:n+1]
    recs=[]
    for rec_row,score in scores:
        rec_id=movie_ids[rec_row]
        rec_title=movies.loc[movies["movieId"]==rec_id,"title"].values[0]
        recs.append(f"{rec_title} (similarity={score:.3f})")
    return recs
#demo examples
test_movies=[
    "Toy Story (1995)",
    "Jumanji (1995)",
    "Heat (1995)",
    "GoldenEye (1995)"
]
for movie in test_movies:
    print("\ninput movie:",movie)
    print("recommendations:")
    for rec in recommend(movie,5):
        print("-",rec)