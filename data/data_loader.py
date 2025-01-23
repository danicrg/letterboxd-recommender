import torch
import pandas as pd
from torch_geometric.data import Data

def train_test_split(ratings_df, movies_df, test_ratio=0.2):
    grouped = ratings_df.groupby("username")
    train_records = []
    test_records = []

    for user, group in grouped:
        user_data = group.sample(frac=1.0)

        num_test = max(1, int(len(user_data) * test_ratio))
        
        user_test = user_data.iloc[:num_test]
        user_train = user_data.iloc[num_test:]

        if len(user_train) == 0:
            user_train = user_test.iloc[:1]
            user_test = user_test.iloc[1:]

        train_records.append(user_train)
        test_records.append(user_test)

    train_ratings_df = pd.concat(train_records, ignore_index=True)
    test_ratings_df = pd.concat(test_records, ignore_index=True)

    train_movies = train_ratings_df["slug"].drop_duplicates().tolist()
    test_movies = test_ratings_df["slug"].drop_duplicates().tolist()

    train_movies_df = movies_df[movies_df["slug"].isin(train_movies)]
    test_movies_df = movies_df[movies_df["slug"].isin(test_movies)]

    return train_ratings_df, test_ratings_df, train_movies_df, test_movies_df


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