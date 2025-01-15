import random

movie_embeddings = final_embeddings[:num_movies]
user_embeddings = final_embeddings[num_movies:]

print("Final Movie Embeddings Shape:", movie_embeddings.shape)
print("Final User Embeddings Shape:", user_embeddings.shape)

def get_user_movie_embeddings(username, movie_embeddings, ratings_df, movies_df):
    user_ratings = ratings_df[ratings_df["username"] == username]
    movie_indexes = user_ratings["movie_index"].tolist()
    return torch.index_select(movie_embeddings, dim=0, index=torch.tensor(movie_indexes)), movies_df.loc[movie_indexes]

class SimpleRecommender:
    def __init__(self, movie_embeddings, learning_rate=0.1):
        self.movie_embeddings = movie_embeddings
        self.user_embedding = torch.zeros(movie_embeddings.shape[1])  # Init user embedding
        self.learning_rate = learning_rate

    def update_user_embedding(self, movie_a_idx, movie_b_idx, preference):
        """
        Update user embedding based on pairwise comparison.
        preference: 1 if user prefers movie_a, -1 if prefers movie_b
        """
        movie_a = self.movie_embeddings[movie_a_idx]
        movie_b = self.movie_embeddings[movie_b_idx]
        update = preference * (movie_a - movie_b)
        self.user_embedding += self.learning_rate * update

    def recommend_movies(self, top_n=10):
        """
        Recommend top N movies based on similarity to the user embedding.
        """
        # Rank all movies by similarity to the user embedding
        scores = self.movie_embeddings @ self.user_embedding
        top_movies = torch.argsort(scores, descending=True)[:top_n]
        return top_movies

class LearnToRankRecommender:
    def __init__(self, movie_embeddings, embedding_dim, learning_rate=0.1):
        """
        Initialize the Recommender system with a trainable user embedding.

        Args:
        - movie_embeddings: Torch tensor of shape (num_movies, embedding_dim).
        - embedding_dim: Dimensionality of the embedding space.
        - learning_rate: Learning rate for the optimizer.
        """
        self.movie_embeddings = movie_embeddings
        self.user_embedding = nn.Parameter(torch.zeros(embedding_dim,dtype=torch.float32))  # Trainable user embedding
        self.optimizer = optim.Adam([self.user_embedding], lr=learning_rate)
        self.loss_fn = nn.MarginRankingLoss(margin=1.0)  # Hinge loss

    def update_user_embedding(self, movie_a_idx, movie_b_idx, preference):
        """
        Update the user embedding based on pairwise comparison.
        
        Args:
        - movie_a_idx: Index of movie A in the movie embeddings.
        - movie_b_idx: Index of movie B in the movie embeddings.
        - preference: 1 if user prefers movie_a, -1 if prefers movie_b.
        """
        # Retrieve the embeddings of the two movies
        movie_a = self.movie_embeddings[movie_a_idx]
        movie_b = self.movie_embeddings[movie_b_idx]
        
        # Compute the scores for the two movies
        score_a = torch.dot(self.user_embedding, movie_a)
        score_b = torch.dot(self.user_embedding, movie_b)
        
        # Define the target for the loss function
        target = torch.tensor([preference], dtype=torch.float32)

        # Compute the loss (hinge loss for pairwise ranking)
        loss = self.loss_fn(score_a.unsqueeze(0), score_b.unsqueeze(0), target)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def recommend_movies(self, top_n=10):
        """
        Recommend top N movies based on similarity to the user embedding.
        
        Args:
        - top_n: Number of top recommendations to return.
        
        Returns:
        - List of indices of the top N recommended movies.
        """
        # Compute similarity scores for all movies
        scores = self.movie_embeddings @ self.user_embedding
        # Get indices of top N movies
        top_movies = torch.argsort(scores, descending=True)[:top_n]
        return top_movies


class MovieComparisonApp:
    def __init__(self, recommender, movies_df):
        self.recommender = recommender
        self.movies_df = movies_df

    def get_movie_details(self, movie_idx):
        movie_slug = self.movies_df.iloc[movie_idx]['slug']
        movie_title = self.movies_df.iloc[movie_idx]['title']
        movie_img = self.movies_df.iloc[movie_idx]['img']
        return movie_slug, movie_title, movie_img

    def compare_movies(self, movie_a_idx, movie_b_idx):
        movie_a_slug, movie_a_title, movie_a_img = self.get_movie_details(movie_a_idx)
        movie_b_slug, movie_b_title, movie_b_img = self.get_movie_details(movie_b_idx)

        print("Which movie do you prefer?")
        print(f"1: {movie_a_title}\n2: {movie_b_title}\n3: Can't decide/Haven't watched either")

        while True:
            try:
                preference = int(input("Your choice: "))
                if preference in [1, 2, 3]:
                    break
                else:
                    print("Invalid input. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        if preference == 3:
            print("Skipping this pair as you haven't watched either.")
        elif preference != 3:
            if preference == 2: preference = -1
            self.recommender.update_user_embedding(movie_a_idx, movie_b_idx, preference)
            return 1
        return 0

    def recommend_top_movies(self, top_n=5):
        top_movies = self.recommender.recommend_movies(top_n=top_n)
        print("\nRecommended Movies:")
        for idx in top_movies:
            slug, title, img = self.get_movie_details(int(idx))
            print(f"- {title} (Slug: {slug})")

    def run(self):
        movie_indices = list(range(len(self.movies_df)))
        n_comparisons = 0
        while n_comparisons < 10:  # 10 comparisons
            movie_a_idx, movie_b_idx = random.sample(movie_indices, 2)
            n_comparisons += self.compare_movies(movie_a_idx, movie_b_idx)

        self.recommend_top_movies()


danicrg_embeddings, danicrg_movies_df = get_user_movie_embeddings("danicrg", movie_embeddings, ratings_df, movies_df)

recommender = LearnToRankRecommender(danicrg_embeddings, 32)
app = MovieComparisonApp(recommender, danicrg_movies_df)
app.run()

