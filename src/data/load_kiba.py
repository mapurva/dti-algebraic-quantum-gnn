import pandas as pd
from pathlib import Path


def load_kiba(data_dir="data/raw/kiba"):
    data_dir = Path(data_dir)

    drugs_df = pd.read_csv(data_dir / "drugs.csv")
    proteins_df = pd.read_csv(data_dir / "proteins.csv")
    affinities = pd.read_csv(data_dir / "affinities.csv").values

    drugs = dict(zip(drugs_df.drug_id, drugs_df.smiles))
    proteins = dict(zip(proteins_df.protein_id, proteins_df.sequence))

    return drugs, proteins, affinities
