import torch
from data.mol_graph import mol_to_graph


def build_gnn_dataset(interactions, drugs, protein_features):
    """
    Build PyG dataset for DTI.

    protein_features: dict {protein_id: feature vector}
    """
    data_list = []

    for _, row in interactions.iterrows():
        d_id = row["drug_id"]
        p_id = row["protein_id"]

        graph = mol_to_graph(drugs[d_id])
        if graph is None:
            continue

        graph.y = torch.tensor(row["pkd"], dtype=torch.float)
        graph.protein_feat = torch.tensor(
            protein_features[p_id], dtype=torch.float
        )

        data_list.append(graph)

    return data_list
