import torch
from torch_geometric.data import Data
from data.blosum62 import blosum_score

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def aa_one_hot(aa):
    x = torch.zeros(len(AMINO_ACIDS))
    if aa in AA_TO_IDX:
        x[AA_TO_IDX[aa]] = 1.0
    return x


def protein_to_weighted_graph(
    sequence,
    blosum_threshold=1,
    max_hop=3
):
    """
    Biologically informed protein graph.
    """
    n = len(sequence)
    x = torch.stack([aa_one_hot(aa) for aa in sequence])

    edge_index = []
    edge_weight = []

    # 1) Sequence adjacency + k-hop proximity
    for i in range(n):
        for j in range(i + 1, min(i + max_hop + 1, n)):
            dist = j - i
            w = 1.0 / dist
            edge_index += [[i, j], [j, i]]
            edge_weight += [w, w]

    # 2) Biochemical similarity edges (BLOSUM)
    for i in range(n):
        for j in range(i + 2, n):
            score = blosum_score(sequence[i], sequence[j])
            if score >= blosum_threshold:
                w = score / 5.0
                edge_index += [[i, j], [j, i]]
                edge_weight += [w, w]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
