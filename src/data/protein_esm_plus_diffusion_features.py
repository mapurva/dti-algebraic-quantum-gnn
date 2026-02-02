import numpy as np


def combine_esm_and_diffusion(esm_embeds, diff_feats):
    combined = {}
    for pid in esm_embeds:
        combined[pid] = np.concatenate(
            [esm_embeds[pid], diff_feats[pid]],
            axis=0
        )
    return combined
