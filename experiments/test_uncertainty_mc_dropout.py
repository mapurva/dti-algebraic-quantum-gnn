import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

from src.utils.metrics import (
    rmse,
    expected_calibration_error,
    plot_reliability_diagram
)

from src.data.load_davis import load_davis
from src.data.build_interactions import build_interaction_table
from src.data.cold_splits import cold_target_split
from src.data.simple_features import build_simple_features
from src.data.gnn_dataset import build_gnn_dataset
from src.training.dti_gnn_model import DTIGNN


def mc_dropout_predictions(model, loader, T=20):
    """
    MC Dropout inference
    """

    model.train()  # 🔥 keep dropout ON

    all_means, all_vars, all_targets = [], [], []

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

    return (
        np.array(all_means),
        np.array(all_vars),
        np.array(all_targets),
    )


def main():

    print("Loading Davis dataset...")
    drugs, proteins, affinities = load_davis("data/raw/davis")

    interactions = build_interaction_table(drugs, proteins, affinities)

    # Same split as training
    train_df, val_df, test_df = cold_target_split(interactions)

    print("Building features...")
    _, protein_feats = build_simple_features(interactions, drugs, proteins)

    test_data = build_gnn_dataset(test_df, drugs, protein_feats)
    test_loader = DataLoader(test_data, batch_size=32)

    print("Loading trained Drug GNN model...")
    model = DTIGNN(len(next(iter(protein_feats.values()))))

    model.load_state_dict(
        torch.load("models/drug_gnn_davis.pt", map_location="cpu")
    )

    model.eval()

    print("Running MC Dropout...")
    mean_pred, var_pred, targets = mc_dropout_predictions(model, test_loader)

    error = np.abs(mean_pred - targets)

    # ---- Metrics ----
    rmse_val = rmse(targets, mean_pred)
    ece_val = expected_calibration_error(targets, mean_pred, var_pred)

    # safer correlation
    if np.std(var_pred) > 0:
        corr = np.corrcoef(var_pred, error)[0, 1]
    else:
        corr = 0.0

    print("\n=== Uncertainty Evaluation ===")
    print("RMSE:", rmse_val)
    print("ECE:", ece_val)
    print("Uncertainty-Error Correlation:", corr)

    # ---- Scatter Plot ----
    plt.figure(figsize=(6, 4))
    plt.scatter(var_pred, error, alpha=0.5)

    plt.xlabel("Predictive Variance (Epistemic Uncertainty)")
    plt.ylabel("Absolute Prediction Error")
    plt.title("Uncertainty vs Prediction Error (Davis)")

    plt.tight_layout()
    plt.savefig("uncertainty_scatter.png", dpi=300)

    print("Saved: uncertainty_scatter.png")

    # ---- Reliability Diagram ----
    plot_reliability_diagram(
        targets,
        mean_pred,
        var_pred,
        n_bins=10,
        save_path="reliability_diagram.png"
    )

    print("Saved: reliability_diagram.png")


if __name__ == "__main__":
    main()