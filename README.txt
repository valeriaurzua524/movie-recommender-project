Movie Recommendation System using Content-Based Filtering
Valeria Urzua – CS4210 Final Project

Project Description
For my final project, I built a movie recommendation system in Python using content-based filtering. The goal of the system is to recommend movies that are similar to a movie the user already likes, similar to how platforms like Netflix suggest movies and shows.

This project uses the MovieLens dataset and machine learning concepts like feature vectorization and cosine similarity to compare movies based on their genres and ratings data.

Dataset Used
MovieLens Small Dataset
https://grouplens.org/datasets/movielens/

Files Included
- movie_recommender.py → main Python code for the recommendation system
- movies.csv → movie titles and genres
- ratings.csv → user ratings data
- tags.csv → movie tag information
- links.csv → movie IDs and external links

Libraries / Dependencies
This project uses:
- pandas
- numpy
- scikit-learn

To install the required libraries:

pip install pandas numpy scikit-learn

How to Run the Project
Open the project folder in VS Code or terminal and run:

python3 movie_recommender.py

What the Program Does
The system:
- loads and cleans the movie dataset
- processes movie genres into numerical features
- computes similarity between movies using cosine similarity
- generates movie recommendations
- outputs the top recommended movies with similarity scores

Example Movies Tested
- Toy Story (1995)
- Jumanji (1995)
- Heat (1995)
- GoldenEye (1995)

Example Output
The program returns the top 5 most similar movies along with similarity scores for each recommendation.

Machine Learning Concepts Used
- content-based filtering
- feature vectorization
- TF-IDF vectorization
- cosine similarity
- recommendation systems

Overall, this project helped me better understand how recommendation systems work and how machine learning can be applied to real-world applications that people use every day.

Author
Valeria Urzua
Cal Poly Pomona
Spring 2026
