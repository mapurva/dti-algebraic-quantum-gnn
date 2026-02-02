import torch
import torch.optim as optim

from data.load_kiba import load_kiba
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_target_split
from data.simple_features import build_simple_features

from training.mlp_baseline import MLPBaseline
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    # Load KIBA
    drugs, proteins, affinities = load_kiba()
    interactions = build_interaction_table(drugs, proteins, affinities)

    # Cold-target split
    train_df, _, test_df = cold_target_split(interactions, seed=42)

    # Protein features (k-mer)
    _, protein_feats = build_simple_features(interactions, drugs, proteins)

    X_train = torch.tensor(
        [protein_feats[p] for p in train_df.protein_id],
        dtype=torch.float32
    )
    y_train = torch.tensor(train_df.affinity.values, dtype=torch.float32)

    X_test = torch.tensor(
        [protein_feats[p] for p in test_df.protein_id],
        dtype=torch.float32
    )
    y_test = torch.tensor(test_df.affinity.values, dtype=torch.float32)

    model = MLPBaseline(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Train
    for _ in range(20):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train).view(-1)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).view(-1)

    y_true = y_test.numpy()
    y_pred = preds.numpy()

    print("\n=== KIBA k-mer MLP (Cold-Target) ===")
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mae(y_true, y_pred))
    print("Pearson:", pearson(y_true, y_pred))
    print("CI:", concordance_index(y_true, y_pred))


if __name__ == "__main__":
    main()
