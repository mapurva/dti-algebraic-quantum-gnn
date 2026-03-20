import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_target_split
from data.dual_gnn_dataset import build_dual_gnn_dataset

from training.dti_dual_gnn import DTIDualGNN
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    drugs, proteins, affinities = load_davis("data/raw/davis")
    interactions = build_interaction_table(drugs, proteins, affinities)

    train_df, _, test_df = cold_target_split(interactions, seed=42)

    train_samples = build_dual_gnn_dataset(train_df, drugs, proteins)
    test_samples = build_dual_gnn_dataset(test_df, drugs, proteins)

    # Separate loaders
    train_drug_loader = DataLoader([s[0] for s in train_samples], batch_size=8, shuffle=True)
    train_prot_loader = DataLoader([s[1] for s in train_samples], batch_size=8, shuffle=True)
    train_y = torch.tensor([s[2] for s in train_samples], dtype=torch.float)

    test_drug_loader = DataLoader([s[0] for s in test_samples], batch_size=8)
    test_prot_loader = DataLoader([s[1] for s in test_samples], batch_size=8)
    test_y = torch.tensor([s[2] for s in test_samples], dtype=torch.float)

    model = DTIDualGNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Training
    model.train()
    idx = 0
    for epoch in range(20):
        idx = 0
        for drug_batch, prot_batch in zip(train_drug_loader, train_prot_loader):
            optimizer.zero_grad()
            y_batch = train_y[idx:idx + drug_batch.num_graphs]
            idx += drug_batch.num_graphs

            preds = model(drug_batch, prot_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    idx = 0

    with torch.no_grad():
        for drug_batch, prot_batch in zip(test_drug_loader, test_prot_loader):
            preds = model(drug_batch, prot_batch)
            y_batch = test_y[idx:idx + drug_batch.num_graphs]
            idx += drug_batch.num_graphs

            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())

    print("\n=== Dual GNN (Drug + Protein) Cold-Target ===")
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mae(y_true, y_pred))
    print("Pearson:", pearson(y_true, y_pred))
    print("CI:", concordance_index(y_true, y_pred))


if __name__ == "__main__":
    main()
