import os

import torch
import torch.optim as optim
import torch.nn as nn
from data.data_loader import prepare_graph_data, preprocess_dfs
from data.data_generation import load_existing_data
from models.movie_user_model import MovieUserEmbeddingModel

from utils.embedding_manager import EmbeddingManager
from utils.batch_sampler import batch_sampler
import copy
from constants import DATA_PATH

# Hyperparameters
embedding_dim = 32
hidden_dim = 64
batch_size = 2048
num_epochs = 100000
early_stopping_steps = 500

# Load data
ratings_df, movies_df = load_existing_data()
print(f"Before processing: len(ratings_df): {len(ratings_df)} - len(movies_df): {len(movies_df)}")
ratings_df, movies_df = preprocess_dfs(ratings_df, movies_df, min_movie_presence=3)
print(f"After processing: len(ratings_df): {len(ratings_df)} - len(movies_df): {len(movies_df)}")
graph_data, index_mappings = prepare_graph_data(ratings_df, movies_df, embedding_dim)

num_movies = len(index_mappings["movies"])
num_users = len(index_mappings["users"])

# Initialize model and optimizer
model = MovieUserEmbeddingModel(num_movies, num_users, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
best_model_state_dict = None
min_loss = float('inf')
no_improvement_steps = 0

for epoch in range(num_epochs):
    model.train()
    total_sum = 0.0
    total_count = 0

    for node_features, edge_batch, edge_attr_batch in batch_sampler(graph_data, batch_size):
        optimizer.zero_grad()
        predicted_ratings, _ = model(edge_batch, edge_attr_batch)
        loss = nn.MSELoss()(predicted_ratings, edge_attr_batch)
        loss.backward()
        optimizer.step()
        total_sum += loss.item() * edge_attr_batch.shape[0]
        total_count += edge_attr_batch.shape[0]

    total_loss = total_sum / total_count
    # Check for improvement
    if total_loss < min_loss:
        min_loss = total_loss
        best_model_state_dict = copy.deepcopy(model.state_dict())
        no_improvement_steps = 0  # Reset the counter when improvement occurs
    else:
        no_improvement_steps += 1

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Best Loss: {min_loss:.4f}")

    # Check if early stopping criterion is met
    if no_improvement_steps >= early_stopping_steps:
        model.load_state_dict(best_model_state_dict)
        no_improvement_steps = 0

        if batch_size >= graph_data.edge_index.size(1):
            print(f"Early stopping at epoch {epoch + 1}, no improvement for {early_stopping_steps} steps.")
            break
        
        batch_size *=2
        early_stopping_steps = min(500, early_stopping_steps + 150)
        print(f"Increasing batch size to {batch_size}")
            

# Save final embeddings from the best model
model.eval()
with torch.no_grad():
    _, final_embeddings = model(graph_data.edge_index, graph_data.edge_attr)


# Initialize embedding manager
embedding_manager = EmbeddingManager(final_embeddings, index_mappings, movies_df, ratings_df)
# Save embeddings for later use
torch.save(embedding_manager, os.path.join(DATA_PATH, 'embedding_manager.pt'))