import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_drug_split, cold_target_split
from data.advanced_features import build_advanced_features

from training.mlp_baseline import MLPBaseline
from utils.metrics import rmse, mae, pearson, concordance_index


def run(split_name, split_fn):
    print(f"\n===== ADVANCED MLP under {split_name.upper()} SPLIT =====\n")

    drugs, proteins, affinities = load_davis("data/raw/davis")
    interactions = build_interaction_table(drugs, proteins, affinities)

    train_df, _, test_df = split_fn(interactions, seed=42)

    X_train, y_train = build_advanced_features(train_df, drugs, proteins)
    X_test, y_test = build_advanced_features(test_df, drugs, proteins)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )

    model = MLPBaseline(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for _ in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()

    print("RMSE:", rmse(y_test.numpy(), preds))
    print("MAE:", mae(y_test.numpy(), preds))
    print("Pearson:", pearson(y_test.numpy(), preds))
    print("CI:", concordance_index(y_test.numpy(), preds))


def main():
    run("cold-drug", cold_drug_split)
    run("cold-target", cold_target_split)


if __name__ == "__main__":
    main()
