import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def build_semantic_protein_graph(embeddings, k=5):
    """
    Nodes = proteins
    Edges = kNN in embedding space
    """
    protein_ids = list(embeddings.keys())
    X = np.stack([embeddings[p] for p in protein_ids])

    sim = cosine_similarity(X)
    G = nx.Graph()

    for i, pid in enumerate(protein_ids):
        G.add_node(pid)

        neighbors = np.argsort(-sim[i])[1:k+1]
        for j in neighbors:
            pj = protein_ids[j]
            w = sim[i, j]
            G.add_edge(pid, pj, weight=w)

    return G
