import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import clusters

def compute_extremities(metric_fn, encoder_fn, furthest_neighbors_ix,
                        strings, show_progress=True):
    return [
        clusters.Node(
            strings[i],
            strings[furthest_neighbors_ix[i]],
            encoder_fn,
            metric_fn)
        for i in tqdm(
            range(len(strings)),
            desc="Measuring the Reaches",
            disable=(not show_progress))
        if (i == furthest_neighbors_ix[furthest_neighbors_ix[i]]
            and i < furthest_neighbors_ix[i])
    ]

def compute_nodes(metric_fn, encoder_fn, nearest_neighbors_ix,
                  strings, show_progress=True):
    return [
        clusters.Node(strings[i],
            strings[nearest_neighbors_ix[i]],
            encoder_fn,
            metric_fn)
        for i in tqdm(
            range(len(strings)),
            desc="Watching the Galaxies Coelesce",
            disable=(not show_progress))
        if (i == nearest_neighbors_ix[nearest_neighbors_ix[i]]
            and i < nearest_neighbors_ix[i])
    ]

def compute_hubs(metric_fn, encoder_fn, nearest_fn, nearest_neighbors_ix,
                 strings, show_progress=True):
    hubs = []
    temp_hubs = []
    for i in tqdm(range(len(strings)),
            desc="Finding Galactic Hubs",
            disable=(not show_progress)):
        temp_hubs.append(clusters.Cluster(
            encoder_fn, metric_fn, nearest=nearest_fn,
            objects=[strings[i]], nodes=[], auto=False,
            name=strings[i]))
            # Its name is the original object's decoded string.
        for index, neighbor in enumerate(nearest_neighbors_ix):
            if neighbor == i:
                temp_hubs[i].add_objects([strings[index]])
            # The 0th index in the hub's list of objects
            #   is also it's original object (is included in hub).
    j = 0
    for h in tqdm(temp_hubs, desc="Erecting Centers of Commerce",
            disable=(not show_progress)):
        if len(h) >= 4: # obj plus 3 or more for whom it is nearest.
            hubs.append(h)
            hubs[j].ID = j
            hubs[j].calculate()
            j += 1
    return hubs

def compute_supernodes(nodes, printer_fn, metric_str, metric_fn,
                       show_progress=True):
    centroids = [n.centroid for n in nodes]
    printer_fn("Fracturing the Empire")
    dist_matrix = sp.distance.squareform(
        sp.distance.pdist(
            centroids,
            metric_str if metric_str != None else metric_fn))
    printer_fn("Establishing a Hierocracy")
    neighbors = np.argmax(dist_matrix, axis=1)
    #neighbors_dist = dist_matrix[range(len(dist_matrix)), neighbors]

    # Compute the Supernodes:
    return [
        clusters.Node(node,
            nodes[neighbors[i]],
            clusters.Node.get_centroid, metric_fn)
        for i, node in enumerate(tqdm(nodes,
            desc="Ascertaining Universe Filaments",
            disable=(not show_progress)))
        if (i == neighbors[neighbors[i]]
            and i < neighbors[i])]

#def compute_nuclei():
#def compute_chains():