import matplotlib.pyplot as plt
import numpy as np

# ===== Data from paper tables =====

models = [
    "Feature MLP",
    "Drug GNN",
    "Drug GNN + Algebraic",
    "Protein-Only (ESM)"
]

# Davis Pearson
davis = [-0.166, 0.434, 0.218, None]

# KIBA Pearson
kiba = [0.255, 0.366, 0.160, -0.034]

# Replace None with NaN for plotting
davis = [np.nan if v is None else v for v in davis]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(7,4))

plt.bar(x - width/2, davis, width, label="Davis", edgecolor="black")
plt.bar(x + width/2, kiba, width, label="KIBA", edgecolor="black")

plt.xticks(x, models, rotation=20)
plt.ylabel("Pearson Correlation")
plt.title("Cold-Target DTI Performance Comparison")

plt.axhline(0, linestyle="--", linewidth=1)
plt.legend()

plt.tight_layout()

plt.savefig("figure1_performance.png", dpi=300)
plt.savefig("figure1_performance.pdf")

print("Figure saved as figure1_performance.png and .pdf")
