import numpy as np
from data.protein_features import build_protein_kmer_features
from data.protein_weighted_diffusion_features import (
    build_weighted_diffusion_features
)


def build_protein_kmer_weighted_diffusion_features(
    proteins,
    kmer_k=3,
    t_values=None
):
    kmer = build_protein_kmer_features(proteins, k=kmer_k)
    diff = build_weighted_diffusion_features(proteins, t_values)

    out = {}
    for p_id in proteins.keys():
        out[p_id] = np.concatenate([kmer[p_id], diff[p_id]], axis=0)

    return out
