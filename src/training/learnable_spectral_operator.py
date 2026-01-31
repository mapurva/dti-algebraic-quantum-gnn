import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from scipy.linalg import expm


class LearnableSpectralOperator(nn.Module):
    """
    Learnable combination of heat kernels:
        sum_i alpha_i * exp(-t_i L)
    """

    def __init__(self, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

        # Learnable diffusion times (positive)
        self.log_t = nn.Parameter(torch.zeros(num_scales))

        # Learnable mixing weights
        self.alpha = nn.Parameter(torch.ones(num_scales))

    def forward(self, sequence):
        """
        sequence: protein sequence string
        returns: spectral feature vector
        """
        # Build sequence graph
        G = nx.Graph()
        n = len(sequence)
        G.add_nodes_from(range(n))
        for i in range(n - 1):
            G.add_edge(i, i + 1)

        L = nx.normalized_laplacian_matrix(G).toarray()

        features = []
        for i in range(self.num_scales):
            t = torch.exp(self.log_t[i]).item()
            H = expm(-t * L)
            features.append(np.trace(H))

        features = torch.tensor(features, dtype=torch.float)
        weights = torch.softmax(self.alpha, dim=0)

        return torch.dot(weights, features)
