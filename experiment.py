danicrg_ratings_df = ratings_df[ratings_df["username"] == "danicrg"]
danicrg_embeddings, danicrg_movies_df = get_user_movie_embeddings("danicrg", movie_embeddings, ratings_df, movies_df)
danicrg_ratings = list(danicrg_ratings_df["rating"])

import numpy as np
from torch.nn.functional import cosine_similarity


def generate_true_embedding(movie_embeddings, ratings):
    """
    Generate a user's true embedding from movie embeddings and ratings.

    Args:
    - movie_embeddings: A 2D numpy array (num_movies x embedding_dim).
    - ratings: A 1D numpy array of movie ratings (num_movies,).

    Returns:
    - u_true: The user's true embedding as a 1D numpy array.
    """
    # Normalize ratings
    min_rating, max_rating = np.min(ratings), np.max(ratings)
    normalized_ratings = (ratings - min_rating) / (max_rating - min_rating)
    
    # Compute weighted embedding
    weighted_embeddings = movie_embeddings.detach().numpy().T * normalized_ratings
    u_true = np.sum(weighted_embeddings, axis=1) / np.sum(normalized_ratings)
    
    return u_true


danicrg_embedding = torch.tensor(generate_true_embedding(danicrg_embeddings, danicrg_ratings))
real_recommender = LearnToRankRecommender(movie_embeddings, 32)

for i in range(10000):
    sampled = danicrg_ratings_df.sample(2)
    ratings = sampled['rating'].tolist()
    indices = sampled['movie_index'].tolist()

    if ratings[0] == ratings[1]:
        continue

    movie_a = movie_embeddings[indices[0]]
    movie_b = movie_embeddings[indices[1]]
    preference = 1 if ratings[0] > ratings[1] else -1
    real_recommender.update_user_embedding(indices[0], indices[1], preference)

danicrg_embedding = real_recommender.user_embedding.detach()

recommender = LearnToRankRecommender(movie_embeddings, 32)

learning_rate = 0.01

products = []
similarities = []
eclids = []
for i in range(100):
    sampled = danicrg_ratings_df.sample(2)
    ratings = sampled['rating'].tolist()
    indices = sampled['movie_index'].tolist()

    if ratings[0] == ratings[1]:
        continue

    movie_a = movie_embeddings[indices[0]]
    movie_b = movie_embeddings[indices[1]]
    preference = 1 if ratings[0] > ratings[1] else -1
    recommender.update_user_embedding(indices[0], indices[1], preference)


    products.append(torch.dot(recommender.user_embedding.detach().float(),danicrg_embedding.float()))
    eclids.append(torch.sqrt(torch.sum((recommender.user_embedding.detach().float() - danicrg_embedding.float()) ** 2)))
    similarity = cosine_similarity(recommender.user_embedding.detach().unsqueeze(0).float(), danicrg_embedding.unsqueeze(0).float())
    similarities.append(similarity.item())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(products, marker='o', linestyle='-', label='Values')
plt.title('Tensor Data Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.savefig('tensor_data_plot.png')
plt.show()