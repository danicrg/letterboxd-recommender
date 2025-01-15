import torch
import torch.optim as optim
import torch.nn as nn
from data.data_loader import prepare_graph_data
from data.data_generation import load_existing_data
from models.movie_user_model import MovieUserEmbeddingModel
from utils.batch_sampler import batch_sampler
import copy

# Hyperparameters
embedding_dim = 32
hidden_dim = 64
batch_size = 2048
num_epochs = 100

# Load data
ratings_df, movies_df = load_existing_data()
graph_data, num_movies, num_users = prepare_graph_data(ratings_df, movies_df, embedding_dim)

# Initialize model and optimizer
model = MovieUserEmbeddingModel(num_movies, num_users, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
best_model = None
min_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for node_features, edge_batch, edge_attr_batch in batch_sampler(graph_data, batch_size):
        optimizer.zero_grad()
        predicted_ratings, _ = model(edge_batch, edge_attr_batch)
        loss = nn.MSELoss()(predicted_ratings, edge_attr_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if total_loss < min_loss:
        min_loss = total_loss
        best_model = copy.deepcopy(model)
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Save final embeddings
best_model.eval()
with torch.no_grad():
    _, final_embeddings = best_model(graph_data.edge_index, graph_data.edge_attr)
torch.save(final_embeddings, 'final_embeddings.pt')