import torch
from torch_geometric.data import Data
from rdkit import Chem

from data.mol_graph import mol_to_graph


def build_gnn_dataset(interactions, drugs, protein_features):
    """
    Build PyG dataset for Drug GNN + protein features.

    Args:
        interactions (pd.DataFrame):
            Columns: drug_id, protein_id, pkd OR affinity
        drugs (dict):
            drug_id -> SMILES
        protein_features (dict):
            protein_id -> np.array

    Returns:
        List[torch_geometric.data.Data]
    """

    data_list = []

    for _, row in interactions.iterrows():
        drug_id = row["drug_id"]
        protein_id = row["protein_id"]

        # --- Target value (Davis vs KIBA) ---
        if "pkd" in row:
            y_val = row["pkd"]
        else:
            y_val = row["affinity"]

        # --- Drug graph ---
        smiles = drugs[drug_id]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        graph = mol_to_graph(mol)

        # --- Protein features ---
        graph.protein_feat = torch.tensor(
            protein_features[protein_id],
            dtype=torch.float32
        )

        # --- Label ---
        graph.y = torch.tensor(y_val, dtype=torch.float32)

        data_list.append(graph)

    return data_list
