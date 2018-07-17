import numpy as np
import scipy.spatial as sp
from tqdm import tqdm

from ..clustertypes.node import Node
from .node_clusterizer import NodeClusterizer
#from analyst import Distances
import analyst

class SupernodeClusterizer(NodeClusterizer, object):

    def __init__(self, category="Supernodes", starred=None,
            node_category="Nodes"):
        super(SupernodeClusterizer, self).__init__(
            category=category, starred=starred)
        self.nodes = None
        self.D = None

    def compute_clusters(self, space, show_progress=True, **kwargs):
        printer            = kwargs["printer_fn"]
        metric_str         = kwargs["metric_str"]
        metric             = kwargs["metric_fn"]
        clusterizer_getter = kwargs["find_evaluator_fn"]
        metric_args        = kwargs["metric_args"]
        auto_print         = kwargs["draw_progress"]
        make_dist_matrix   = kwargs["make_dist_matrix"]
        make_kth_neighbors = kwargs["make_kth_neighbors"]
        parallel_count     = kwargs["parallel_count"]
        metric_args        = kwargs["metric_args"]

        # No need to make sure Nodes are computed before Supernodes,
        #   since get_nodes ensures this for us:
        node_clusterizer = clusterizer_getter(
            self.node_category, force_creation=True)
        self.nodes = node_clusterizer.get_clusters(**kwargs)

        if len(self.nodes) > 0:
            # Compute distance matrix and nearest neighbors for node centroids:
            centroids = [n.centroid for n in self.nodes]
            printer("Fracturing the Empire", "Computing Nodal Distance Matrix")
            # Assume to compute Nodal distance matrix if we did the spatial one:
            self.D = analyst.Distances(centroids, metric_str, metric, printer,
                auto_print=auto_print, make_distance_matrix=make_dist_matrix,
                make_kth_neighbors=make_kth_neighbors,
                parallel_count=parallel_count, **metric_args)
            neighbors = self.D.kth_neighbors(1)
            # node_dist_matrix = sp.distance.squareform(
            #     sp.distance.pdist(
            #         centroids,
            #         metric_str if metric_str != None else metric,
            #         **metric_args))
            # printer("Establishing a Hierocracy",
            #     "Computing Nearest Neighbor Nodes")
            # neighbors = np.argpartition(node_dist_matrix, 1, axis=1)[:,1]
            
        # Compute the Supernodes:
        printer("Ascertaining Universe Filaments", "Finding Supernodes")
        self.clusters = [
            Node(node,
                self.nodes[neighbors[i]],
                Node.get_centroid, metric, **metric_args)
            for i, node in enumerate(tqdm(self.nodes,
                disable=(not show_progress)))
            if (i == neighbors[neighbors[i]]
                and i < neighbors[i])]

    # Don't need to override vectors_to_clusters; if needed, parent would have.

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        printer = kwargs["printer_fn"]
        space = kwargs["embeddings"]

        printer("Measuring their Magnitude", "Calculating Supernode Span")
        self.add_generic_node_stats()

        if len(self.clusters) > 0:
            # Island Factor
            printer("Minding the Macrocosm", "Calculating Island Factor")
            self.data_dict["Island Factor"] = (
                len(self.clusters)*4.0/float(len(space)))
            self.add_star("Island Factor")

            # Hierarchical Factor
            printer("Deliberating over Dominions",
                "Calculating Hierarchical Factor")
            self.data_dict["Hierarchical Factor"] = (
                len(self.clusters)*2.0/float(len(self.nodes)))
            self.add_star("Hierarchical Factor")
        
        self.add_star("Span Min")