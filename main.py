import pandas as pd
from preprocess import preprocess

def read_file(path=None):
    df = pd.read_csv(path)
    return df

if __name__=='__main__':
    path = r'Dataset\tmdb_movies.csv'
    df = read_file(path)
    df = preprocess(df)
