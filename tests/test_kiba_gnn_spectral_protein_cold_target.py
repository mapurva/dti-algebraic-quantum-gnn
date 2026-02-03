import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from data.load_kiba import load_kiba
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_target_split
from data.gnn_dataset import build_gnn_dataset

from data.protein_diffusion_features import build_protein_diffusion_features
from training.dti_gnn_model import DTIGNN
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    # Load KIBA
    drugs, proteins, affinities = load_kiba()
    interactions = build_interaction_table(drugs, proteins, affinities)

    # Cold-target split
    train_df, _, test_df = cold_target_split(interactions, seed=42)

    # Protein diffusion (AGT) features
    protein_feats = build_protein_diffusion_features(proteins)

    # Build datasets
    train_data = build_gnn_dataset(train_df, drugs, protein_feats)
    test_data = build_gnn_dataset(test_df, drugs, protein_feats)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    protein_dim = len(next(iter(protein_feats.values())))
    model = DTIGNN(protein_dim)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Train
    for _ in range(10):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch, batch.protein_feat).view(-1)
            loss = loss_fn(preds, batch.y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch, batch.protein_feat).view(-1)
            y_true.extend(batch.y.numpy())
            y_pred.extend(preds.numpy())

    print("\n=== KIBA Drug GNN + Diffusion Protein (Cold-Target) ===")
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mae(y_true, y_pred))
    print("Pearson:", pearson(y_true, y_pred))
    print("CI:", concordance_index(y_true, y_pred))


if __name__ == "__main__":
    main()
