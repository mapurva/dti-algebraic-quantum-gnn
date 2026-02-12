# KIBA Cold-Target Results (Used in JBHI Paper)

| Model | RMSE | MAE | Pearson | CI |
|------|------|------|------|------|
| Feature MLP | 3.315 | 3.214 | 0.255 | -- |
| Drug GNN | 0.789 | 0.560 | 0.366 | 0.668 |
| Drug GNN + Algebraic Protein | 0.853 | 0.646 | 0.160 | 0.580 |
| Protein-Only (ESM) | 10.993 | 10.961 | -0.034 | 0.493 |

Notes:
- Cold-target split
- Protein-only model evaluated only on KIBA as diagnostic experiment
- Script: `tests/test_kiba_gnn_drug_cold_target.py`
