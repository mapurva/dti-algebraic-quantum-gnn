import torch
import torch.nn as nn
from training.drug_gnn import DrugGNN
from training.protein_gnn import ProteinGNN


class DTIDualGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.drug_gnn = DrugGNN(hidden_dim)
        self.protein_gnn = ProteinGNN(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, drug_graph, protein_graph):
        d_emb = self.drug_gnn(drug_graph)
        p_emb = self.protein_gnn(protein_graph)
        x = torch.cat([d_emb, p_emb], dim=1)
        return self.mlp(x).squeeze(-1)
