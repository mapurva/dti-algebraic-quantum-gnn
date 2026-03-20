import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

from src.data.load_davis import load_davis
from src.data.build_interactions import build_interaction_table
from src.data.cold_splits import cold_target_split
from src.data.simple_features import build_simple_features
from src.data.gnn_dataset import build_gnn_dataset
from src.training.dti_gnn_model import DTIGNN
from src.utils.metrics import rmse


def mc_dropout_predictions(model, loader, T=20):
    """
    Perform MC Dropout inference
    """

    model.train()  # 🔥 enable dropout

    all_means = []
    all_vars = []
    all_targets = []

    for batch in loader:

        preds_T = []

        for _ in range(T):
            with torch.no_grad():
                p = model(batch, batch.protein_feat)
                preds_T.append(p.cpu().numpy())

        preds_T = np.stack(preds_T, axis=0)

        mean_pred = preds_T.mean(axis=0)
        var_pred = preds_T.var(axis=0)

        all_means.extend(mean_pred.flatten())
        all_vars.extend(var_pred.flatten())
        all_targets.extend(batch.y.cpu().numpy().flatten())

    return np.array(all_means), np.array(all_vars), np.array(all_targets)


def main():

    print("Loading Davis dataset...")

    drugs, proteins, affinities = load_davis("data/raw/davis")

    interactions = build_interaction_table(drugs, proteins, affinities)

    # ✅ IMPORTANT: same split as training
    train_df, val_df, test_df = cold_target_split(interactions)

    print("Building features...")

    # IMPORTANT: use FULL interactions (not subset)
    _, protein_feats = build_simple_features(interactions, drugs, proteins)

    test_data = build_gnn_dataset(test_df, drugs, protein_feats)
    test_loader = DataLoader(test_data, batch_size=32)

    print("Loading trained Drug GNN model...")

    model = DTIGNN(len(next(iter(protein_feats.values()))))

    # ✅ CRITICAL FIX: load trained weights
    model.load_state_dict(torch.load("models/drug_gnn_davis.pt"))

    model.eval()  # initialize properly

    print("Running MC Dropout...")

    mean_pred, var_pred, targets = mc_dropout_predictions(model, test_loader)

    print("Sample targets:", targets[:5])
    print("Sample preds:", mean_pred[:5])

    error = np.abs(mean_pred - targets)

    print("RMSE:", rmse(targets, mean_pred))

    # 📊 Scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(var_pred, error, alpha=0.5)

    plt.xlabel("Predictive Variance (Epistemic Uncertainty)")
    plt.ylabel("Absolute Prediction Error")
    plt.title("Uncertainty vs Prediction Error (Davis)")

    plt.tight_layout()
    plt.savefig("uncertainty_scatter.png", dpi=300)

    print("Figure saved as uncertainty_scatter.png")


if __name__ == "__main__":
    main()