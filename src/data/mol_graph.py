import torch
from torch_geometric.data import Data
from rdkit import Chem


def mol_to_graph(smiles):
    """
    Convert SMILES string to PyG Data object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: atomic number
    x = []
    for atom in mol.GetAtoms():
        x.append([atom.GetAtomicNum()])
    x = torch.tensor(x, dtype=torch.float)

    # Edge index
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)
