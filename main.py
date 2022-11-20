import pandas as pd
from preprocess import preprocess

if __name__=='__main__':
    df = pd.read_csv(r'Dataset\tmdb_movies.csv')
    df = preprocess(df)
