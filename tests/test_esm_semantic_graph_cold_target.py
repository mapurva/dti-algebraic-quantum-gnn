import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_target_split
from data.gnn_dataset import build_gnn_dataset

from data.embed_proteins_esm2 import embed_proteins_esm2
from data.protein_semantic_graph import build_semantic_protein_graph
from data.protein_semantic_diffusion import semantic_diffusion_features
from data.protein_esm_plus_diffusion_features import combine_esm_and_diffusion

from training.dti_gnn_model import DTIGNN
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    drugs, proteins, affinities = load_davis("data/raw/davis")
    interactions = build_interaction_table(drugs, proteins, affinities)

    train_df, _, test_df = cold_target_split(interactions, seed=42)

    # STEP 1: semantic embeddings
    esm_embeds = embed_proteins_esm2(proteins)

    # STEP 2: semantic graph (TRAIN ONLY)
    train_proteins = set(train_df.protein_id.unique())
    train_embeds = {p: esm_embeds[p] for p in train_proteins}

    G = build_semantic_protein_graph(train_embeds, k=5)

    # STEP 3: diffusion
    diff_feats = semantic_diffusion_features(G, t=1.0)

    # Fill missing (cold proteins) with zero diffusion
    for p in proteins:
        if p not in diff_feats:
            diff_feats[p] = np.zeros(1)

    # STEP 4: combine features
    protein_feats = combine_esm_and_diffusion(esm_embeds, diff_feats)

    train_data = build_gnn_dataset(train_df, drugs, protein_feats)
    test_data = build_gnn_dataset(test_df, drugs, protein_feats)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    protein_dim = len(next(iter(protein_feats.values())))
    model = DTIGNN(protein_dim=protein_dim)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for _ in range(20):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch, batch.protein_feat)
            loss = loss_fn(preds, batch.y.view(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch, batch.protein_feat)
            y_true.extend(batch.y.view(-1).tolist())
            y_pred.extend(preds.tolist())

    print("\n=== ESM-2 Semantic Graph + Diffusion (Cold-Target) ===")
    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mae(y_true, y_pred))
    print("Pearson:", pearson(y_true, y_pred))
    print("CI:", concordance_index(y_true, y_pred))


if __name__ == "__main__":
    main()
