import torch
from torch_geometric.data import Data

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def aa_one_hot(aa):
    vec = torch.zeros(len(AMINO_ACIDS))
    if aa in AA_TO_IDX:
        vec[AA_TO_IDX[aa]] = 1.0
    return vec


def protein_to_graph(sequence):
    """
    Convert protein sequence to PyG Data object.
    """
    x = torch.stack([aa_one_hot(aa) for aa in sequence])

    edge_index = []
    for i in range(len(sequence) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
