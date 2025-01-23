import torch

class EmbeddingManager:
    def __init__(self, embeddings, index_mappings, movies_df, ratings_df):
        self.embeddings = embeddings
        self.index_mappings = index_mappings
        self.movies_df = movies_df
        self.ratings_df = ratings_df

    def get_user_embedding(self, username):
        user_index = self.index_mappings['users'][username]
        return self.embeddings[user_index]

    def get_movie_embedding(self, movie_slug):
        movie_index = self.index_mappings['movies'][movie_slug]
        return self.embeddings[movie_index]

    def get_top_movies_for_user(self, username, top_n=10):
        # Get the user embedding
        user_embedding = self.get_user_embedding(username)
        if user_embedding is None:
            raise ValueError(f"User '{username}' not found in index mappings.")

        # Get movie embeddings
        movie_indices = list(self.index_mappings['movies'].values())
        movie_embeddings = self.embeddings[movie_indices]

        # Compute similarity scores
        similarities = torch.matmul(movie_embeddings, user_embedding.unsqueeze(1)).squeeze()
        top_movie_indices = torch.topk(similarities, top_n).indices.tolist()  # Convert to list of Python integers

        # Retrieve the top movies
        return [self.movies_df.iloc[idx]['slug'] for idx in top_movie_indices]