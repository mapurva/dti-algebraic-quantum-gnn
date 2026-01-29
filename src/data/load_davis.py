import numpy as np
import pickle
from pathlib import Path


def load_davis(data_dir):
    data_dir = Path(data_dir)

    # --- Load drugs (dictionary) ---
    with open(data_dir / "drug_smiles.csv", "r") as f:
        drugs = eval(f.read())   # dict: drug_id -> SMILES

    # --- Load proteins (dictionary) ---
    with open(data_dir / "protein_sequences.csv", "r") as f:
        proteins = eval(f.read())  # dict: protein_id -> sequence

    # --- Load affinities (Python2 pickle) ---
    with open(data_dir / "affinities.csv", "rb") as f:
        affinities = pickle.load(f, encoding="latin1")

    affinities = np.array(affinities)

    return drugs, proteins, affinities


def inspect_davis(drugs, proteins, affinities):
    print("\n=== DAVIS DATASET INSPECTION ===\n")

    print(f"Number of drugs: {len(drugs)}")
    print(f"Number of proteins: {len(proteins)}")
    print(f"Affinity matrix shape: {affinities.shape}")

    print("\nAffinity value range:")
    print(f"Min: {np.nanmin(affinities)}")
    print(f"Max: {np.nanmax(affinities)}")
