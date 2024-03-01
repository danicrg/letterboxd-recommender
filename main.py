from data import get_movie_dict_from_df, get_user_data
from score import jaccard_user_score, cosine_user_score, cross_user_movies_dict_score, n_movies_in_common, scale_ratings
from graph import build_bipartite_graph, biased_random_walk

USERNAME = "danicrg"

if __name__ == "__main__":
    my_data = get_user_data(USERNAME)
    users = ["danicrg", "emmaelkmw", "carlotabravo", "jazze", "thomasflight", "kurstboy", "blazques"]
    other_data = get_user_data("blazques")

    my_movie_dict = get_movie_dict_from_df(my_data)
    other_movie_dict = get_movie_dict_from_df(other_data)

    print("n_movies_in_common", n_movies_in_common(my_movie_dict, other_movie_dict))
    print("Cross_user_score", cross_user_movies_dict_score(my_movie_dict, other_movie_dict))
    print("Jaccard_user_score", jaccard_user_score(my_movie_dict, other_movie_dict))
    print("Cosine_user_score", cosine_user_score(my_movie_dict, other_movie_dict))

    user_ratings = {
        user: get_movie_dict_from_df(get_user_data(user))
        for user in users
    }

    graph = build_bipartite_graph(user_ratings)
    start_user = USERNAME
    affinity_scores = biased_random_walk(graph, start_user)

    print(f"Affinity Scores for Movies (Biased Random Walk from {start_user}):")
    for movie, score in sorted(affinity_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{movie}: {score}")