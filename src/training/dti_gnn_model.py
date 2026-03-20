import torch
import torch.nn as nn
from training.drug_gnn import DrugGNN


class DTIGNN(nn.Module):
    def __init__(self, protein_dim, hidden_dim=64):
        super().__init__()

        self.drug_gnn = DrugGNN(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + protein_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),   # 🔥 IMPORTANT
            nn.Linear(128, 1)
        )

    def forward(self, drug_graph, protein_feat):
        """
        drug_graph: PyG batch
        protein_feat: Tensor [batch_size, protein_dim]
        """

        drug_emb = self.drug_gnn(drug_graph)

        # Ensure correct shape
        if protein_feat.dim() == 1:
            protein_feat = protein_feat.view(drug_emb.size(0), -1)

        x = torch.cat([drug_emb, protein_feat], dim=1)

        return self.mlp(x).squeeze(-1)
