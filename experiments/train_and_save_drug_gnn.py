import os
import torch
import numpy as np

from torch_geometric.loader import DataLoader

from src.data.load_davis import load_davis
from src.data.build_interactions import build_interaction_table
from src.data.cold_splits import cold_target_split
from src.data.simple_features import build_simple_features
from src.data.gnn_dataset import build_gnn_dataset
from src.training.dti_gnn_model import DTIGNN
from src.utils.metrics import rmse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        preds = model(batch, batch.protein_feat)

        loss = loss_fn(preds, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            preds = model(batch, batch.protein_feat)

            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(batch.y.cpu().numpy())

    return rmse(np.array(targets_all), np.array(preds_all))


def main():

    print("Loading Davis dataset...")

    drugs, proteins, affinities = load_davis("data/raw/davis")

    interactions = build_interaction_table(drugs, proteins, affinities)

    train_df, val_df, test_df = cold_target_split(interactions)

    drug_feats, protein_feats = build_simple_features(interactions, drugs, proteins)

    train_data = build_gnn_dataset(train_df, drugs, protein_feats)
    val_data = build_gnn_dataset(val_df, drugs, protein_feats)
    test_data = build_gnn_dataset(test_df, drugs, protein_feats)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    print("Building model...")

    model = DTIGNN(len(next(iter(protein_feats.values()))))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print("Training...")

    for epoch in range(10):
        loss = train(model, train_loader, optimizer, loss_fn)
        val_rmse = evaluate(model, val_loader)

        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val RMSE: {val_rmse:.4f}")

    test_rmse = evaluate(model, test_loader)
    print("Final Test RMSE:", test_rmse)

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/drug_gnn_davis.pt")

    print("Model saved at models/drug_gnn_davis.pt")


if __name__ == "__main__":
    main()