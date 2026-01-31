import numpy as np
from collections import Counter


def build_protein_kmer_features(proteins, k=3):
    """
    Build k-mer frequency features for each protein.

    Returns:
        dict: protein_id -> feature vector
    """
    # Build vocabulary
    vocab = set()
    for seq in proteins.values():
        for i in range(len(seq) - k + 1):
            vocab.add(seq[i:i+k])
    vocab = sorted(vocab)

    protein_features = {}

    for p_id, seq in proteins.items():
        counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
        protein_features[p_id] = np.array(
            [counts.get(kmer, 0) for kmer in vocab],
            dtype=np.float32
        )

    return protein_features
