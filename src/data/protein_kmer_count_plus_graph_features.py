import numpy as np
from data.protein_features import build_protein_kmer_features
from data.protein_kmer_graph_spectral import build_kmer_spectral_features


def build_protein_kmer_plus_graph_features(
    proteins,
    kmer_k=3,
    num_eigs=10
):
    kmer_counts = build_protein_kmer_features(proteins, k=kmer_k)
    kmer_graph = build_kmer_spectral_features(
        proteins, k=kmer_k, num_eigs=num_eigs
    )

    combined = {}
    for p_id in proteins.keys():
        combined[p_id] = np.concatenate(
            [kmer_counts[p_id], kmer_graph[p_id]],
            axis=0
        )

    return combined
