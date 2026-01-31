import numpy as np
from data.protein_features import build_protein_kmer_features
from data.protein_diffusion_features import build_protein_diffusion_features


def build_protein_kmer_diffusion_features(
    proteins,
    kmer_k=3,
    t_values=None
):
    kmer_feats = build_protein_kmer_features(proteins, k=kmer_k)
    diffusion_feats = build_protein_diffusion_features(
        proteins,
        t_values=t_values
    )

    combined = {}
    for p_id in proteins.keys():
        combined[p_id] = np.concatenate(
            [kmer_feats[p_id], diffusion_feats[p_id]],
            axis=0
        )

    return combined
