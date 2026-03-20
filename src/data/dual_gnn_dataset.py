from data.mol_graph import mol_to_graph
from data.protein_graph import protein_to_graph


def build_dual_gnn_dataset(interactions, drugs, proteins):
    """
    Returns list of (drug_graph, protein_graph, y)
    """
    samples = []

    for _, row in interactions.iterrows():
        d_id = row["drug_id"]
        p_id = row["protein_id"]

        drug_graph = mol_to_graph(drugs[d_id])
        protein_graph = protein_to_graph(proteins[p_id])

        if drug_graph is None:
            continue

        y = row["pkd"]
        samples.append((drug_graph, protein_graph, y))

    return samples
