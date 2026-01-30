import numpy as np


def cold_drug_split(interactions, seed=42, ratios=(0.7, 0.1, 0.2)):
    """
    Cold-drug split:
    - Drugs in test set do NOT appear in train or val.
    """
    rng = np.random.default_rng(seed)

    drug_ids = interactions["drug_id"].unique()
    rng.shuffle(drug_ids)

    n = len(drug_ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    train_drugs = set(drug_ids[:n_train])
    val_drugs = set(drug_ids[n_train:n_train + n_val])
    test_drugs = set(drug_ids[n_train + n_val:])

    train_df = interactions[interactions["drug_id"].isin(train_drugs)]
    val_df = interactions[interactions["drug_id"].isin(val_drugs)]
    test_df = interactions[interactions["drug_id"].isin(test_drugs)]

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def cold_target_split(interactions, seed=42, ratios=(0.7, 0.1, 0.2)):
    """
    Cold-target split:
    - Proteins in test set do NOT appear in train or val.
    """
    rng = np.random.default_rng(seed)

    protein_ids = interactions["protein_id"].unique()
    rng.shuffle(protein_ids)

    n = len(protein_ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    train_proteins = set(protein_ids[:n_train])
    val_proteins = set(protein_ids[n_train:n_train + n_val])
    test_proteins = set(protein_ids[n_train + n_val:])

    train_df = interactions[interactions["protein_id"].isin(train_proteins)]
    val_df = interactions[interactions["protein_id"].isin(val_proteins)]
    test_df = interactions[interactions["protein_id"].isin(test_proteins)]

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

