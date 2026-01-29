import pandas as pd


def build_simple_features(interactions, drugs, proteins):
    """
    Build simple numeric features for MLP baseline.

    Drug feature:
        - length of SMILES string

    Protein feature:
        - length of amino acid sequence

    Returns:
        X (pd.DataFrame), y (np.ndarray)
    """
    drug_smiles = drugs
    protein_seqs = proteins

    features = []
    targets = []

    for _, row in interactions.iterrows():
        d_id = row["drug_id"]
        p_id = row["protein_id"]

        smiles = drug_smiles[d_id]
        sequence = protein_seqs[p_id]

        features.append({
            "smiles_len": len(smiles),
            "seq_len": len(sequence)
        })

        targets.append(row["pkd"])

    X = pd.DataFrame(features)
    y = pd.Series(targets)

    return X, y
