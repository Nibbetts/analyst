import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import clusters

def compute_extremities(metric, encoder, furthest_neighbors_ix,
                        strings, show_progress=True)
    return [
        clusters.Node(strings[i],
            strings[furthest_neighbors_ix[i]],
            encoder, metric)
        for i in tqdm(range(len(strings)),
            desc="Measuring the Reaches",
            disable=(not show_progress))
        if (i == furthest_neighbors_ix[furthest_neighbors_ix[i]]
            and i < furthest_neighbors_ix[i])]

def compute_nodes()