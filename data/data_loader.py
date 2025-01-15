import torch
import pandas as pd
from torch_geometric.data import Data

def prepare_graph_data(ratings_df, movies_df, embedding_dim):
    num_movies = len(movies_df)
    num_users = len(ratings_df['username'].drop_duplicates())
    
    # Index mappings
    movies_reverse_index = {item: index for index, item in movies_df['slug'].to_dict().items()}
    usernames_reverse_index = {item: index for index, item in enumerate(ratings_df['username'].drop_duplicates())}
    
    ratings_df["movie_index"] = ratings_df["slug"].map(movies_reverse_index)
    ratings_df["user_index"] = ratings_df["username"].map(usernames_reverse_index) + num_movies

    # Features and edges
    movie_features = torch.rand((num_movies, embedding_dim), dtype=torch.float)
    user_features = torch.rand((num_users, embedding_dim), dtype=torch.float)
    all_features = torch.cat([movie_features, user_features], dim=0)

    edges = torch.tensor([ratings_df['user_index'], ratings_df['movie_index']], dtype=torch.long)
    edge_attr = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

    return Data(x=all_features, edge_index=edges, edge_attr=edge_attr), num_movies, num_users