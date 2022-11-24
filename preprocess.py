import pandas as pd
import json
import numpy as np
import spacy

def fetch_name_from_dict(obj):
    L = []
    obj = json.loads(obj)
    for info in obj:
        L.append(info['name'])
    return L

def fetch_director(obj):
    L = []
    obj = json.loads(obj)
    for info in obj:
        if info['job']=='Director':
            L.append(info['name'])
    return L

def top_4_cast(obj):
    L = []
    counter = 0
    obj = json.loads(obj)
    for info in obj:
        if counter < 4:
            L.append(info['name'])
            counter += 1
        else:
            break
    return L

def lemmatize(sents):
    nlp = spacy.load("en_core_web_sm")
    docs = sents.tolist()
    lem_sents = []
    for doc in nlp.pipe(docs, batch_size=512, n_process=3, disable=["parser", "ner"]):
        lem_tokens = [token.lemma_ for token in doc]
        lem_sent = " ".join(lem_tokens)
        lem_sents.append(lem_sent)
    return lem_sents

def preprocess(df):
    df = df.rename(columns={'title_x':'title'})
    # # # feature to consider
    # # genres
    # # id
    # # keywords
    # # original_language
    # # overview
    # # title
    # # cast
    # # crew
    df = df[['genres','id','keywords','original_language','overview','title','cast','crew']]
    df = df.dropna().reset_index(drop=True)

    df['genres'] = df['genres'].apply(fetch_name_from_dict)
    df['keywords'] = df['keywords'].apply(fetch_name_from_dict)
    df['crew'] = df['crew'].apply(fetch_director)
    df = df.rename(columns={'crew':'director'})
    df['cast'] = df['cast'].apply(top_4_cast)
    
    ### removing spaces ###
    df['genres'] = df['genres'].apply(lambda x:[a.replace(" ","") for a in x])
    df['cast'] = df['cast'].apply(lambda x:[a.replace(" ","") for a in x])
    df['director'] = df['director'].apply(lambda x:[a.replace(" ","") for a in x])
    df['keywords'] = df['keywords'].apply(lambda x:[a.replace(" ","") for a in x])
    
    ### Lemmatizing "overview" ###
    df['overview'] = lemmatize(df.overview)

    ### creating tag out of features ###
    df['overview'] = df['overview'].apply(lambda x: [x])
    df['original_language'] = df['original_language'].apply(lambda x:[x])
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['original_language'] + df['cast'] + df['director']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x))
    df['tags'] = df['tags'].apply(lambda x: x.lower())
    df = df[['id','title','tags']]
 
    return df



