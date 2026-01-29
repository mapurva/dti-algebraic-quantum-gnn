import numpy as np
import pandas as pd


def build_interaction_table(drugs, proteins, affinities):
    """
    Convert Davis affinity matrix into a flat interaction table.

    Args:
        drugs (dict): drug_id -> SMILES
        proteins (dict): protein_id -> sequence
        affinities (np.ndarray): shape (n_drugs, n_proteins), raw Kd values

    Returns:
        pd.DataFrame with columns:
        [drug_id, protein_id, kd, pkd]
    """
    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())

    assert affinities.shape == (len(drug_ids), len(protein_ids)), \
        "Affinity matrix shape does not match drug/protein counts"

    records = []

    for i, d_id in enumerate(drug_ids):
        for j, p_id in enumerate(protein_ids):
            kd = affinities[i, j]

            # Skip invalid or missing values if any
            if kd <= 0:
                continue

            pkd = -np.log10(kd)

            records.append({
                "drug_id": d_id,
                "protein_id": p_id,
                "kd": kd,
                "pkd": pkd
            })

    interactions = pd.DataFrame(records)
    return interactions
