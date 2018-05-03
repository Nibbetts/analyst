import numpy as np
import scipy.spatial as sp
from tqdm import tqdm

from ..clustertypes.node import Node
from .nodes import NodeClusterizer

class SupernodeClusterizer(NodeClusterizer, object):

    def __init__(self, node_clusterizer,
            category="Supernodes", generic_stats=False):
        #   Notice we can add parameters.
        super(SupernodeClusterizer, self).__init__(category, generic_stats)
        self.node_clusterizer = node_clusterizer
        self.nodes = []

    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        printer = kwargs["printer_fn"]
        metric_str = kwargs["metric_str"]
        metric_fn = kwargs["metric_fn"]

        # Make sure Nodes are computed before Supernodes:
        self.node_clusterizer.compute_clusters(space, show_progress, **kwargs)
        self.nodes = self.node_clusterizer.clusters

        # Compute distance matrix and nearest neighbors for node centroids:
        centroids = [n.centroid for n in self.nodes]
        printer("Fracturing the Empire (Computing Nodal Distance Matrix)")
        node_dist_matrix = sp.distance.squareform(
            sp.distance.pdist(
                centroids,
                metric_str if metric_str != None else metric_fn))
        printer("Establishing a Hierocracy (Computing Nearest Neighbor Nodes)")
        neighbors = np.argmax(node_dist_matrix, axis=1)
            
        # Compute the Supernodes:
        self.clusters = [
            Node(node,
                self.nodes[neighbors[i]],
                Node.get_centroid, metric_fn)
            for i, node in enumerate(tqdm(self.nodes,
                desc="Ascertaining Universe Filaments (Finding Supernodes)",
                disable=(not show_progress)))
            if (i == neighbors[neighbors[i]]
                and i < neighbors[i])]

    # Don't need to override vectors_to_clusters, since NodeClusterizer does,
    #   and it is parent.

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        printer = kwargs["printer_fn"]
        space = kwargs["space"]

        if self.generic_stats:
            # Supernode Count
            self.data_dict["Count"] = len(self.clusters)

            if len(self.clusters) > 0:
                # Span Stats
                printer("Measuring their Magnitude \
                    (Calculating Supernode Span)")
                self._compute_list_stats([n.span for n in self.clusters],
                    "Span", self.data_dict)

                # Island Factor
                printer("Minding the Macrocosm (Calculating Island Factor)")
                self.data_dict["Island Factor"] = (
                    len(self.clusters)*4.0/float(len(space)))
                self.add_star("Island Factor")

                # Hierarchical Factor
                printer("Deliberating over Dominions \
                    (Calculating Hierarchical Factor)")
                self.data_dict["Hierarchical Factor"] = (
                    len(self.clusters)*2.0/float(len(self.nodes)))
                self.add_star("Hierarchical Factor")