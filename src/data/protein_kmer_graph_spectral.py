import numpy as np
import networkx as nx
from scipy.linalg import eigvals
from data.protein_kmer_graph import build_kmer_graph


def kmer_graph_spectrum(sequence, k=3, num_eigs=10):
    G = build_kmer_graph(sequence, k=k)

    if G.number_of_nodes() < 2:
        return np.zeros(num_eigs)

    A = nx.to_numpy_array(G, weight='weight')
    L = np.diag(A.sum(axis=1)) - A  # unnormalized Laplacian

    eigs = np.sort(np.real(eigvals(L)))

    if len(eigs) < num_eigs:
        eigs = np.pad(eigs, (0, num_eigs - len(eigs)))

    return eigs[:num_eigs]


def build_kmer_spectral_features(proteins, k=3, num_eigs=10):
    features = {}
    for p_id, seq in proteins.items():
        features[p_id] = kmer_graph_spectrum(seq, k, num_eigs)
    return features
