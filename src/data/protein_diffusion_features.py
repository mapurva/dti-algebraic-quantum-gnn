import numpy as np
import networkx as nx
from scipy.linalg import expm


def protein_sequence_graph(sequence):
    G = nx.Graph()
    n = len(sequence)
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def heat_kernel_trace(sequence, t_values):
    """
    Compute heat kernel trace features:
        trace(exp(-t L)) for multiple t
    """
    G = protein_sequence_graph(sequence)
    L = nx.normalized_laplacian_matrix(G).toarray()

    features = []
    for t in t_values:
        Ht = expm(-t * L)
        features.append(np.trace(Ht))

    return np.array(features)


def build_protein_diffusion_features(proteins, t_values=None):
    """
    Returns:
        dict: protein_id -> diffusion feature vector
    """
    if t_values is None:
        # Multi-scale diffusion times
        t_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    features = {}
    for p_id, seq in proteins.items():
        features[p_id] = heat_kernel_trace(seq, t_values)

    return features
