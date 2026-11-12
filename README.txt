# Movie Recommendation System

Valeria Urzua

## Project Description

For this project, I built a movie recommendation system using Python and the MovieLens dataset. The goal was to create a program that recommends movies based on a movie someone already likes, similar to how platforms like Netflix suggest movies and shows.

I used machine learning concepts like TF-IDF vectorization and cosine similarity to compare movies based on their genres and rating information. I also included average movie ratings to give the system more information when finding similar movies.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

## Dataset Used

MovieLens Small Dataset

https://grouplens.org/datasets/movielens/

The dataset includes:
- 9,742 movies
- 100,836 ratings
- 610 users
- Movie titles and genres

## Files Included

- movie_recommender.py - Main Python code for the recommendation system
- movies.csv - Movie titles and genres
- ratings.csv - User ratings for different movies
- tags.csv - Movie tags
- links.csv - Movie IDs and external links

## How the Program Works

The program:

1. Loads the movie and ratings datasets.
2. Cleans and organizes the movie information.
3. Uses TF-IDF vectorization to turn movie genres into numerical features.
4. Calculates average movie ratings and other rating information.
5. Combines genre and rating features to compare movies.
6. Uses cosine similarity to find movies with similar characteristics.
7. Returns the top 5 recommended movies with their similarity scores.

## How to Run the Project

First, install the required Python libraries:

```bash
python3 -m pip install pandas numpy scikit-learn
```

Make sure the Python file and CSV files are saved in the same folder.

Then run:

```bash
python3 movie_recommender.py
```

The program will display movie recommendations in the terminal.

## Example Movies Tested

I tested the recommendation system using:

- Toy Story (1995)
- Jumanji (1995)
- Heat (1995)
- GoldenEye (1995)

## Example Output

Input Movie: Toy Story (1995)

```text
Recommendations:
- Monsters, Inc. (2001) (similarity=0.999)
- Toy Story 2 (1999) (similarity=0.997)
- Emperor's New Groove, The (2000) (similarity=0.985)
- Antz (1998) (similarity=0.985)
- Shrek the Third (2007) (similarity=0.968)
```

Input Movie: GoldenEye (1995)

```text
Recommendations:
- Rock, The (1996) (similarity=1.000)
- Hunt for Red October, The (1990) (similarity=0.998)
- Cliffhanger (1993) (similarity=0.998)
- Die Hard 2 (1990) (similarity=0.998)
- Broken Arrow (1996) (similarity=0.997)
```

The similarity scores show how closely two movies match based on the features used by the program. A score closer to 1 means the movies have more similar characteristics.

## Machine Learning Concepts Used

- Content-based filtering
- TF-IDF vectorization
- Feature engineering
- Cosine similarity
- Recommendation systems
- Data preprocessing

## What I Learned

This project helped me better understand how recommendation systems work and how machine learning can be used in applications we use every day.

I learned how to work with larger datasets, turn text information into numerical features, and use cosine similarity to compare different movies.

I also got more experience working with Python libraries like Pandas, NumPy, and Scikit-learn.

One thing I found interesting was seeing how movies with similar genres could receive very high similarity scores. In the future, I would like to experiment with adding movie descriptions and adjusting feature weights to make the recommendations more diverse.

## Future Improvements

Some things I would like to add in the future include:

- A simple interface where users can search for movies
- Movie descriptions and tags for more detailed recommendations
- More personalized recommendations based on individual user preferences
- Additional testing to compare different recommendation methods

## Author

Valeria Urzua

GitHub: https://github.com/valeriaurzua524
