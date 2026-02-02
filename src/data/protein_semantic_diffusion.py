import numpy as np
import networkx as nx
from scipy.linalg import expm


def semantic_diffusion_features(G, t=1.0):
    """
    Apply heat diffusion on protein semantic graph.
    Returns dict {protein_id: scalar feature}
    """
    nodes = list(G.nodes())
    L = nx.normalized_laplacian_matrix(G, weight="weight").toarray()
    H = expm(-t * L)

    features = {}
    for i, pid in enumerate(nodes):
        features[pid] = np.array([H[i, i]])

    return features
