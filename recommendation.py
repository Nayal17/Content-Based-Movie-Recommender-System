import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vectors(df):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(df['tags']).toarray()
    return vectors

def similarity(df):
    vectors_ = vectors(df)
    similarity_ = cosine_similarity(vectors_)
    return similarity_

def recommend(df,movie=None):
    movie_index = df[df['title']==movie].index[0]
    similarity_all = similarity(df)
    similarity_ = similarity_all[movie_index]
    top_10_recommended_movies = sorted(enumerate(similarity_),reverse=True,key=lambda x:x[1])[1:11]
    rec_movies = []
    for movie in top_10_recommended_movies:
        movie = movie[0]
        rec_movies.append(df.iloc[movie].title)

    return rec_movies

