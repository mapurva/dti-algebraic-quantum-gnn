import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.split_interactions import split_interactions
from data.simple_features import build_simple_features

from training.mlp_baseline import MLPBaseline
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    data_dir = "data/raw/davis"

    # Load data
    drugs, proteins, affinities = load_davis(data_dir)
    interactions = build_interaction_table(drugs, proteins, affinities)

    train_df, val_df, test_df = split_interactions(interactions, seed=42)

    # Build features
    X_train, y_train = build_simple_features(train_df, drugs, proteins)
    X_val, y_val = build_simple_features(val_df, drugs, proteins)
    X_test, y_test = build_simple_features(test_df, drugs, proteins)

    # Convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )

    # Model
    model = MLPBaseline(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Train MSE: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy()
        test_pred = model(X_test).numpy()

    print("\n=== MLP Baseline Results ===\n")

    print("Validation Metrics:")
    print("RMSE:", rmse(y_val.numpy(), val_pred))
    print("MAE:", mae(y_val.numpy(), val_pred))
    print("Pearson:", pearson(y_val.numpy(), val_pred))
    print("CI:", concordance_index(y_val.numpy(), val_pred))

    print("\nTest Metrics:")
    print("RMSE:", rmse(y_test.numpy(), test_pred))
    print("MAE:", mae(y_test.numpy(), test_pred))
    print("Pearson:", pearson(y_test.numpy(), test_pred))
    print("CI:", concordance_index(y_test.numpy(), test_pred))


if __name__ == "__main__":
    main()
