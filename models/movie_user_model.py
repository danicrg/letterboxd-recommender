import torch
import torch.nn as nn
from .graph_sage import GraphSAGE

class MovieUserEmbeddingModel(nn.Module):
    def __init__(self, num_movies, num_users, embedding_dim, hidden_dim):
        super(MovieUserEmbeddingModel, self).__init__()
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.graphsage = GraphSAGE(embedding_dim, hidden_dim, embedding_dim)

    def forward(self, edge_index, edge_attr):
        all_embeddings = torch.cat(
            [self.movie_embeddings.weight, self.user_embeddings.weight], dim=0
        )
        refined_embeddings = self.graphsage(all_embeddings, edge_index)
        src_embeds = refined_embeddings[edge_index[0]]
        dst_embeds = refined_embeddings[edge_index[1]]
        predicted_ratings = (src_embeds * dst_embeds).sum(dim=1)
        return predicted_ratings, refined_embeddings