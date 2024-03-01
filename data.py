from scraper import scrape_user
import os
import time
import pandas as pd


def get_file_age_in_days(filename):
    """Returns the age of the file in days."""
    return (time.time() - os.path.getmtime(filename)) / (60 * 60 * 24)


def get_user_data(username):
    """Returns a DataFrame of the user's films."""
    filename = f"data/{username}.parquet"
    
    if os.path.exists(filename) and get_file_age_in_days(filename) < 1:
        return pd.read_parquet(f"data/{username}.parquet")

    df_film = scrape_user(username)
    df_film.to_parquet(filename)
    return df_film


def get_movie_dict_from_df(df_film):
    movie_dict = dict(zip(df_film['id'], df_film['rating']))
    return movie_dict