import networkx as nx
import random

def build_bipartite_graph(user_ratings):
    G = nx.Graph()

    for user, ratings in user_ratings.items():
        for movie, rating in ratings.items():
            G.add_node(user, bipartite=0)  # User node
            G.add_node(movie, bipartite=1)  # Movie node
            G.add_edge(user, movie, rating=rating)  # Edge with rating

    return G

def biased_random_walk(graph, start_user, num_walks=100000, alpha=0.1):
    affinity_scores = {node: 0 for node in graph.nodes if graph.nodes[node]['bipartite'] == 1}

    current_node = start_user  # Start with a user node

    for _ in range(num_walks):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break

        # Use ratings as probabilities for choosing the next movie
        probabilities = [graph[current_node][neighbor]['rating'] for neighbor in neighbors]
        probabilities_sum = sum(probabilities)
        probabilities = [prob / probabilities_sum for prob in probabilities]

        chosen_node = random.choices(neighbors, weights=probabilities)[0]

        # Check if the chosen node is a movie (bipartite=1)
        if graph.nodes[chosen_node]['bipartite'] == 1:
            affinity_scores[chosen_node] += 1

        if random.random() > alpha:
            current_node = chosen_node
        else:
            current_node = start_user  # Return to the starting user node

    return affinity_scores

# Example usage:
user_ratings = {
    'user1': {'movie1': 1, 'movie2': 1, 'movie3': 5, 'movie6': 1},
    'user2': {'movie1': 1, 'movie2': 1, 'movie3': 5, 'movie4': 5, 'movie5': 5},
    # Add more users and ratings as needed
}

graph = build_bipartite_graph(user_ratings)
start_user = 'user1'
affinity_scores = biased_random_walk(graph, start_user)

print(f"Affinity Scores for Movies (Biased Random Walk from {start_user}):")
for movie, score in sorted(affinity_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{movie}: {score}")