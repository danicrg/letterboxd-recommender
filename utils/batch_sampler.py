import torch

def batch_sampler(data, batch_size):
    total_edges = data.edge_index.size(1)
    for i in range(0, total_edges, batch_size):
        edge_batch = data.edge_index[:, i:i + batch_size]
        edge_attr_batch = data.edge_attr[i:i + batch_size].float()

        nodes_in_batch = torch.unique(edge_batch)
        node_features = data.x[nodes_in_batch].float()
        
        node_map = {node.item(): idx for idx, node in enumerate(nodes_in_batch)}
        remapped_edges = torch.tensor(
            [[node_map[src.item()], node_map[dst.item()]] for src, dst in edge_batch.t()],
            dtype=torch.long
        ).t()
        
        yield node_features, remapped_edges, edge_attr_batch