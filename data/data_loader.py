import torch
import pandas as pd
from torch_geometric.data import Data

def preprocess_dfs(ratings_df, movies_df, min_movie_presence=3):
    # Remove unusable data
    ratings_df = ratings_df[ratings_df["rating"] != -1]
    
    # Drop duplicate records
    movies_df.drop_duplicates(subset=["slug"], inplace=True)
    ratings_df.drop_duplicates(subset=["slug", "username", "rating"], inplace=True)

    # Remove single appearance movies
    movie_counts = ratings_df.groupby("slug")["username"].count()
    valid_movies = movie_counts[movie_counts > min_movie_presence].index
    ratings_df = ratings_df[ratings_df["slug"].isin(valid_movies)]
    movies_df = movies_df[movies_df["slug"].isin(valid_movies)]
    
    num_movies = len(movies_df)
    
    # Normalize ratings
    ratings_df["normalized_ratings"] = 2 * (ratings_df["rating"] - 0.5) / 4.5 - 1
    
    # Assign id's
    movies_reverse_index = {item: index for index, item in enumerate(ratings_df['slug'].drop_duplicates())}
    usernames_reverse_index = {item: index for index, item in enumerate(ratings_df['username'].drop_duplicates())}
    
    ratings_df["movie_index"] = ratings_df["slug"].map(movies_reverse_index)
    movies_df["movie_index"] = movies_df["slug"].map(movies_reverse_index)
    
    ratings_df["user_index"] = ratings_df["username"].map(usernames_reverse_index) + num_movies

    return ratings_df, movies_df


def prepare_graph_data(ratings_df, movies_df, embedding_dim):    
    num_movies = len(movies_df)
    num_users = len(ratings_df['username'].drop_duplicates())

    # Features and edges
    movie_features = torch.rand((num_movies, embedding_dim), dtype=torch.float)
    user_features = torch.rand((num_users, embedding_dim), dtype=torch.float)
    all_features = torch.cat([movie_features, user_features], dim=0)

    edges = torch.tensor([ratings_df['user_index'], ratings_df['movie_index']], dtype=torch.long)
    edge_attr = torch.tensor(ratings_df['normalized_ratings'].values, dtype=torch.float)

    index_mappings = {
        'movies': movies_df.set_index("slug")["movie_index"].to_dict(),
        'users': ratings_df[["username", "user_index"]].drop_duplicates().set_index('username')['user_index'].to_dict(),
    }

    return Data(x=all_features, edge_index=edges, edge_attr=edge_attr), index_mappings