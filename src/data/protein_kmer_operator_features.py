import numpy as np
from data.protein_features import build_protein_kmer_features
from data.protein_learned_operator_features import (
    build_protein_operator_features
)


def build_protein_kmer_operator_features(
    proteins,
    operator,
    kmer_k=3
):
    kmer_feats = build_protein_kmer_features(proteins, k=kmer_k)
    op_feats = build_protein_operator_features(proteins, operator)

    combined = {}
    for p_id in proteins.keys():
        combined[p_id] = np.concatenate(
            [kmer_feats[p_id], op_feats[p_id]],
            axis=0
        )

    return combined
