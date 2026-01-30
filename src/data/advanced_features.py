import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter


def morgan_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def kmer_features(sequence, k=3, vocab=None):
    counts = Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))

    if vocab is None:
        return counts

    return np.array([counts.get(kmer, 0) for kmer in vocab])


def build_advanced_features(interactions, drugs, proteins, k=3):
    """
    Build strong classical features:
    - Morgan fingerprints for drugs
    - k-mer frequency vectors for proteins
    """
    # Build protein k-mer vocabulary (once)
    all_sequences = proteins.values()
    vocab = set()
    for seq in all_sequences:
        for i in range(len(seq) - k + 1):
            vocab.add(seq[i:i+k])
    vocab = sorted(vocab)

    X_drug = []
    X_protein = []
    y = []

    for _, row in interactions.iterrows():
        d_id = row["drug_id"]
        p_id = row["protein_id"]

        smiles = drugs[d_id]
        sequence = proteins[p_id]

        X_drug.append(morgan_fingerprint(smiles))
        X_protein.append(kmer_features(sequence, k=k, vocab=vocab))
        y.append(row["pkd"])

    X = np.hstack([np.array(X_drug), np.array(X_protein)])
    y = np.array(y)

    return X, y
