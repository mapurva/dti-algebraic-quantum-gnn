import json
import numpy as np
import pandas as pd
from pathlib import Path


def preprocess_kiba(raw_dir="data/raw/kiba"):
    raw_dir = Path(raw_dir)

    # ---------- Drugs (JSON) ----------
    with open(raw_dir / "ligands_can.txt", encoding="utf-8") as f:
        drugs_dict = json.load(f)

    drugs_df = pd.DataFrame(
        [(k, v) for k, v in drugs_dict.items()],
        columns=["drug_id", "smiles"]
    )
    drugs_df.to_csv(raw_dir / "drugs.csv", index=False)

    # ---------- Proteins (JSON) ----------
    with open(raw_dir / "proteins.txt", encoding="utf-8") as f:
        proteins_dict = json.load(f)

    proteins_df = pd.DataFrame(
        [(k, v) for k, v in proteins_dict.items()],
        columns=["protein_id", "sequence"]
    )
    proteins_df.to_csv(raw_dir / "proteins.csv", index=False)

    # ---------- Affinities ----------
    affinity = np.loadtxt(raw_dir / "kiba_binding_affinity_v2.txt")
    pd.DataFrame(affinity).to_csv(raw_dir / "affinities.csv", index=False)

    print("KIBA preprocessing complete")
    print("Drugs:", drugs_df.shape)
    print("Proteins:", proteins_df.shape)
    print("Affinity matrix:", affinity.shape)


if __name__ == "__main__":
    preprocess_kiba()
