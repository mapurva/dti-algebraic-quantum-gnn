import numpy as np
from data.protein_features import build_protein_kmer_features
from data.protein_spectral_features import build_protein_spectral_features


def build_protein_kmer_spectral_features(proteins, kmer_k=3, spectral_k=10):
    """
    Combine k-mer + spectral features.
    """
    kmer_feats = build_protein_kmer_features(proteins, k=kmer_k)
    spectral_feats = build_protein_spectral_features(proteins, k=spectral_k)

    combined = {}
    for p_id in proteins.keys():
        combined[p_id] = np.concatenate(
            [kmer_feats[p_id], spectral_feats[p_id]],
            axis=0
        )

    return combined
