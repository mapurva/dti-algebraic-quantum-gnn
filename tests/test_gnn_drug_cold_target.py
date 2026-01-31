import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_target_split
from data.protein_features import build_protein_kmer_features
from data.gnn_dataset import build_gnn_dataset

from training.dti_gnn_model import DTIGNN
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    # Load data
    drugs, proteins, affinities = load_davis("data/raw/davis")
    interactions = build_interaction_table(drugs, proteins, affinities)

    # Cold-target split
    train_df, _, test_df = cold_target_split(interactions, seed=42)

    # Protein features (k-mer)
    protein_feats = build_protein_kmer_features(proteins)

    # Build datasets
    train_data = build_gnn_dataset(train_df, drugs, protein_feats)
    test_data = build_gnn_dataset(test_df, drugs, protein_feats)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Model
    protein_dim = len(next(iter(protein_feats.values())))
    model = DTIGNN(protein_dim=protein_dim)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Training
    for epoch in range(20):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch, batch.protein_feat)
            loss = loss_fn(preds, batch.y.view(-1))
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch, batch.protein_feat)
            y_true.extend(batch.y.view(-1).tolist())
            y_pred.extend(preds.tolist())

    print("\n=== GNN Drug Baseline (Cold-Target) ===")
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mae(y_true, y_pred))
    print("Pearson:", pearson(y_true, y_pred))
    print("CI:", concordance_index(y_true, y_pred))


if __name__ == "__main__":
    main()
