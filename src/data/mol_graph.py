import torch
from torch_geometric.data import Data
from rdkit import Chem


def mol_to_graph(mol):
    """
    Convert an RDKit Mol object into a PyTorch Geometric Data graph.
    """

    # Node features: atomic number
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data
