# Davis Cold-Target Results (Used in JBHI Paper)

| Model | RMSE | MAE | Pearson | CI |
|------|------|------|------|------|
| Feature MLP | 1.228 | 0.816 | -0.166 | 0.413 |
| Drug GNN | 0.798 | 0.534 | 0.434 | 0.720 |
| Drug GNN + Algebraic Protein | 0.923 | 0.636 | 0.218 | 0.615 |

Notes:
- Cold-target split (no target overlap)
- Same training settings across models
- Derived from `tests/test_gnn_drug_cold_target.py`
