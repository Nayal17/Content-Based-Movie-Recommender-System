import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

def vectors(df):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(df['tags']).toarray()
    return vectors

def similarity(df):
    vectors_ = vectors(df)
    similarity_ = cosine_similarity(vectors_)
    return similarity_

def fetch_poster(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=76df17040ad5eedc415f09ac17c35ed5".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"], data['vote_average']

def recommend(df,movie=None):
    movie_index = df[df['title']==movie].index[0]
    similarity_all = similarity(df)
    similarity_ = similarity_all[movie_index]
    top_8_recommended_movies = sorted(enumerate(similarity_),reverse=True,key=lambda x:x[1])[1:9]
    rec_movies = []
    movie_posters = []
    user_scores = []
    for movie in top_8_recommended_movies:
        idx = movie[0]
        rec_movies.append(df.iloc[idx].title)
        movie_posters.append(fetch_poster(df.iloc[idx].id)[0])
        user_scores.append(fetch_poster(df.iloc[idx].id)[1])

    return rec_movies, movie_posters, user_scores


    

