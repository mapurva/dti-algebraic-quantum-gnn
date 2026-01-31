import numpy as np
import networkx as nx
from scipy.linalg import expm
from data.protein_weighted_graph import protein_to_weighted_graph


def weighted_heat_trace(sequence, t_values):
    data = protein_to_weighted_graph(sequence)
    G = nx.Graph()

    for i in range(data.edge_index.size(1)):
        u = int(data.edge_index[0, i])
        v = int(data.edge_index[1, i])
        w = float(data.edge_weight[i])
        G.add_edge(u, v, weight=w)

    L = nx.normalized_laplacian_matrix(G, weight='weight').toarray()

    traces = []
    for t in t_values:
        H = expm(-t * L)
        traces.append(np.trace(H))

    return np.array(traces)


def build_weighted_diffusion_features(
    proteins,
    t_values=None
):
    if t_values is None:
        t_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    feats = {}
    for p_id, seq in proteins.items():
        feats[p_id] = weighted_heat_trace(seq, t_values)

    return feats
