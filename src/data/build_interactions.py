import pandas as pd
import numpy as np


def build_interaction_table(drugs, proteins, affinities):
    """
    Build interaction DataFrame with pkd.

    Returns:
        DataFrame with columns:
        drug_id, protein_id, kd, pkd
    """

    rows = []

    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())

    for i, d in enumerate(drug_ids):
        for j, p in enumerate(protein_ids):

            kd = affinities[i, j]

            # Skip invalid
            if kd <= 0 or np.isnan(kd):
                continue

            # 🔥 CRITICAL FIX
            pkd = -np.log10(kd)

            rows.append({
                "drug_id": d,
                "protein_id": p,
                "kd": kd,
                "pkd": pkd
            })

    return pd.DataFrame(rows)