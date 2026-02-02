import numpy as np
import pandas as pd
from collections import defaultdict


def build_simple_features(interactions, drugs, proteins):
    """
    Build simple baseline features.

    Returns:
        drug_features: dict[drug_id] -> np.array
        protein_features: dict[protein_id] -> np.array

    This function is dataset-agnostic:
    - Uses `pkd` if present (Davis)
    - Uses `affinity` otherwise (KIBA)
    """

    # --- Drug features: simple statistics over interactions ---
    drug_values = defaultdict(list)
    protein_values = defaultdict(list)

    for _, row in interactions.iterrows():
        d = row["drug_id"]
        p = row["protein_id"]

        if "pkd" in row:
            val = row["pkd"]
        else:
            val = row["affinity"]

        drug_values[d].append(val)
        protein_values[p].append(val)

    # Aggregate statistics
    drug_features = {}
    for d, vals in drug_values.items():
        vals = np.array(vals)
        drug_features[d] = np.array([
            vals.mean(),
            vals.std(),
            vals.min(),
            vals.max()
        ], dtype=np.float32)

    protein_features = {}
    for p, vals in protein_values.items():
        vals = np.array(vals)
        protein_features[p] = np.array([
            vals.mean(),
            vals.std(),
            vals.min(),
            vals.max()
        ], dtype=np.float32)

    return drug_features, protein_features
