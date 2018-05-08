import numpy as np
import scipy.spatial as sp
from tqdm import tqdm

from ..clustertypes.node import Node
from .nodes import NodeClusterizer

class SupernodeClusterizer(NodeClusterizer, object):

    def __init__(self, category="Supernodes", starred=None,
            node_category="Nodes"):
        super(SupernodeClusterizer, self).__init__(
            category=category, starred=starred, node_category=node_category)
        self.nodes = None

    def compute_clusters(self, space, show_progress=True, **kwargs):
        printer            = kwargs["printer_fn"]
        metric_str         = kwargs["metric_str"]
        metric_fn          = kwargs["metric_fn"]
        clusterizer_getter = kwargs["find_evaluator_fn"]
        metric_args        = kwargs["metric_args"]

        # No need to make sure Nodes are computed before Supernodes,
        #   since get_nodes ensures this for us:
        node_clusterizer = clusterizer_getter(self.node_category)
        self.nodes = node_clusterizer.get_nodes(space, show_progress, **kwargs)

        # Compute distance matrix and nearest neighbors for node centroids:
        centroids = [n.centroid for n in self.nodes]
        printer("Fracturing the Empire (Computing Nodal Distance Matrix)")
        node_dist_matrix = sp.distance.squareform(
            sp.distance.pdist(
                centroids,
                metric_str if metric_str != None else metric_fn,
                **metric_args))
        printer("Establishing a Hierocracy (Computing Nearest Neighbor Nodes)")
        neighbors = np.argmax(node_dist_matrix, axis=1)
            
        # Compute the Supernodes:
        self.clusters = [
            Node(node,
                self.nodes[neighbors[i]],
                Node.get_centroid, metric_fn, **metric_args)
            for i, node in enumerate(tqdm(self.nodes,
                desc="Ascertaining Universe Filaments (Finding Supernodes)",
                disable=(not show_progress)))
            if (i == neighbors[neighbors[i]]
                and i < neighbors[i])]

    # Don't need to override vectors_to_clusters; if needed, parent would have.

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        printer = kwargs["printer_fn"]
        space = kwargs["space"]

        printer("Measuring their Magnitude \
            (Calculating Supernode Span)")
        self.add_generic_node_stats()

        if len(self.clusters) > 0:
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