import numpy as np
import networkx as nx


def protein_sequence_graph(sequence):
    """
    Build a simple sequence adjacency graph.
    """
    G = nx.Graph()
    n = len(sequence)
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def laplacian_spectrum(sequence, k=10):
    """
    Compute top-k Laplacian eigenvalues of protein sequence graph.
    """
    G = protein_sequence_graph(sequence)
    L = nx.normalized_laplacian_matrix(G).toarray()

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)

    # Pad if sequence too short
    if len(eigvals) < k:
        eigvals = np.pad(eigvals, (0, k - len(eigvals)))

    return eigvals[:k]


def build_protein_spectral_features(proteins, k=10):
    """
    Returns:
        dict: protein_id -> spectral feature vector
    """
    features = {}
    for p_id, seq in proteins.items():
        features[p_id] = laplacian_spectrum(seq, k=k)
    return features
