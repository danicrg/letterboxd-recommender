from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from data import get_user_data, get_movie_dict_from_df

def cross_user_score(username1, username2):
    """Returns the cross-user score between username1 and username2."""
    user1_movies = get_user_data(username1)
    user2_movies = get_user_data(username2)

    user1_movies = get_movie_dict_from_df(user1_movies)
    user2_movies = get_movie_dict_from_df(user2_movies)

    return cross_user_movies_dict_score(user1_movies, user2_movies)


def cross_user_movies_dict_score(user1_movies: Dict[str, float], user2_movies: Dict[str, float]):
    """Returns the cross-user score between user1 and user2."""
    return jaccard_user_score(user1_movies, user2_movies) * cosine_user_score(user1_movies, user2_movies)


def n_movies_in_common(user1_movies: Dict[str, float], user2_movies: Dict[str, float]):
    """Returns the number of movies in common between user1 and user2."""
    return len(set(user1_movies.keys()).intersection(set(user2_movies.keys())))


def jaccard_similarity(list1: List[str], list2: List[str]):
    """Returns the Jaccard similarity between list1 and list2."""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def jaccard_user_score(user1_movies: Dict[str, float], user2_movies: Dict[str, float]):
    """Returns the Jaccard score between user1 and user2."""
    return jaccard_similarity(user1_movies.keys(), user2_movies.keys())


def cosine_user_score(user1_movies: Dict[str, float], user2_movies: Dict[str, float]):
    """Returns the cosine similarity between user1 and user2."""
    user1_ratings = []
    user2_ratings = []

    user_1_movies_scaled = scale_ratings(user1_movies)
    user_2_movies_scaled = scale_ratings(user2_movies)
    
    for movie in user1_movies.keys():
        if movie in user2_movies.keys() and user1_movies[movie] != -1 and user2_movies[movie] != -1:
            user1_ratings.append(user_1_movies_scaled[movie])
            user2_ratings.append(user_2_movies_scaled[movie])
    return cosine_similarity([user1_ratings], [user2_ratings])[0][0]

def scale_ratings(user_movies: Dict[str, float]):
    """Returns the normalized ratings of the user."""
    user_ratings = []
    for movie in user_movies.keys():
        if user_movies[movie] != -1:
            user_ratings.append(user_movies[movie])
    avg_rating = sum(user_ratings) / len(user_ratings)
    for movie in user_movies.keys():
        if user_movies[movie] != -1:
            user_movies[movie] = (user_movies[movie] - avg_rating)
    return user_movies