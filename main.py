import pandas as pd
from preprocess import preprocess
from recommendation import recommend
import streamlit as st
import time

def read_file(path=None):
    df = pd.read_csv(path)
    return df

if __name__=='__main__':
    st.set_page_config(layout="wide")
    path = r'Dataset\tmdb_movies.csv'
    df = read_file(path)
    
    st.title("Movie Recommender System")
    df = preprocess(df)
    selected_movie = st.selectbox("Select movie of your interest: ",df.title.values)
    if st.button("Recommend"):
        movies, posters, user_scores = recommend(df,selected_movie)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.header(movies[0])
            st.image(posters[0])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[0]*10)}%</p>', unsafe_allow_html=True)

            st.header(movies[4])
            st.image(posters[4])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[4]*10)}%</p>', unsafe_allow_html=True)

        with col2:
            st.header(movies[1])
            st.image(posters[1])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[1]*10)}%</p>', unsafe_allow_html=True)

            st.header(movies[5])
            st.image(posters[5])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[5]*10)}%</p>', unsafe_allow_html=True)

        
        with col3:
            st.header(movies[2])
            st.image(posters[2])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[2]*10)}%</p>', unsafe_allow_html=True)

            st.header(movies[6])
            st.image(posters[6])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[6]*10)}%</p>', unsafe_allow_html=True)

        with col4:
            st.header(movies[3])
            st.image(posters[3])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[3]*10)}%</p>', unsafe_allow_html=True)

            st.header(movies[7])
            st.image(posters[7])
            st.markdown(
                f'<p style="font-family:Courier; color:white; font-size: 16px;">User Score: {round(user_scores[7]*10)}%</p>', unsafe_allow_html=True)





