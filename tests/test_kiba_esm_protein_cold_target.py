import torch
import torch.optim as optim

from data.load_kiba import load_kiba
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_target_split

from data.embed_proteins_esm2 import embed_proteins_esm2
from training.mlp_baseline import MLPBaseline
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    # Load KIBA
    drugs, proteins, affinities = load_kiba()
    interactions = build_interaction_table(drugs, proteins, affinities)

    # Cold-target split
    train_df, _, test_df = cold_target_split(interactions, seed=42)

    # ESM-2 embeddings (protein-only)
    protein_embeddings = embed_proteins_esm2(proteins)

    def build_xy(df):
        X, y = [], []
        for _, row in df.iterrows():
            X.append(protein_embeddings[row["protein_id"]])
            y.append(row["affinity"])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    X_train, y_train = build_xy(train_df)
    X_test, y_test = build_xy(test_df)

    model = MLPBaseline(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Train
    for _ in range(20):
        optimizer.zero_grad()
        preds = model(X_train).view(-1)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).view(-1)

    print("\n=== KIBA ESM Protein-Only (Cold-Target) ===")
    print("RMSE:", rmse(y_test.numpy(), preds.numpy()))
    print("MAE:", mae(y_test.numpy(), preds.numpy()))
    print("Pearson:", pearson(y_test.numpy(), preds.numpy()))
    print("CI:", concordance_index(y_test.numpy(), preds.numpy()))


if __name__ == "__main__":
    main()
