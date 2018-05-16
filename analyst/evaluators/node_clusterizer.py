from tqdm import tqdm
import numpy as np
import scipy.spatial as sp

from ..clustertypes.node import Node
from .clusterizer import Clusterizer

class NodeClusterizer(Clusterizer, object):

    # UNNEEDED BECAUSE ACCESS TO CLUSTERIZER LIST
    # While not forcing a singleton, this provides the capability.
    #   Also allows for different types:
    # instances = {}

    # @staticmethod
    # def get_instance(category="Nodes"):
    #     if category not in NodeClusterizer.instances:
    #         NodeClusterizer.instances[category] = NodeClusterizer(category)
    #     return NodeClusterizer.instances[category]

    def __init__(self, category="Nodes", starred=None):
        super(NodeClusterizer, self).__init__(
            category=category, starred=starred)

    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        strings     = kwargs["strings"]
        encode      = kwargs["encoder_fn"]
        metric      = kwargs["metric_fn"]
        neighbors   = kwargs["kth_neighbors_ix_fn"]
        metric_args = kwargs["metric_args"]
        printer     = kwargs["printer_fn"]

        # Nearest neighbor indeces array:
        nearest     = neighbors(1)

        # Compute Nodes:
        printer("Watching the Galaxies Coelesce", "Computing Nodes")
        self.clusters = [
            Node(strings[i], strings[nearest[i]], encode, metric, **metric_args)
            for i in tqdm(range(len(strings)), disable=(not show_progress))
            if (i == nearest[nearest[i]]
                and i < nearest[i])]

        # Compute the Relative Alignment of each Node:
        if len(self.clusters) > 1:
            for node in self.clusters:
                node.alignment = np.mean([
                    abs(np.dot(node.alignment_vec, n.alignment_vec)) \
                    for n in self.clusters if n != node])
        elif len(self.clusters) == 1: self.clusters[0].alignment = 1.0

        # Useful data to store:
        self._string_node_dict = {}
        for node in self.clusters:
            self._string_node_dict[node[0]] = node
            self._string_node_dict[node[1]] = node

    # Would override to prevent filling in, but it checks, so we needn't worry:
    # def vectors_to_clusters(self, **kwargs):
    #     # self.clusters is already filled in,
    #     #   so we override this function with an empty version.
    #     return

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        printer = kwargs["printer_fn"]
        space = kwargs["embeddings"]

        self.add_generic_node_stats()

        if len(self.clusters) > 0:
            # Nodal Factor
            printer("Comparing the Cosmos", "Calculating Nodal Factor")
            self.data_dict["Nodal Factor"] = (
                len(self.clusters)*2.0/float(len(space)))
            self.add_star("Nodal Factor")
            #   I tend to think this is important.

            # Alignment Factor
            printer("Musing over Magnetic Moments",
                "Calculating Alignment Factor")
            self.data_dict["Alignment Factor"] = np.mean([
                n.alignment for n in self.clusters])
            # avg_align = np.mean(
            #     [n.alignment_vec for n in self.clusters], axis=0)
            #     # Note: this only works because all on one side of the space,
            #     #   and all are normalized.
            # self.data_dict["Alignment Factor"] = np.linalg.norm(avg_align)
            # # avg_align /= np.linalg.norm(avg_align)
            # # self.data_dict["Alignment Factor"] = \
            # #     np.mean([
            # #         np.abs(np.dot(avg_align, n.alignment_vec) \
            # #             if np.linalg.norm(n.alignment_vec) != 0 else 0.0)
            # #         for n in self.clusters)] \
            # #     if np.linalg.norm(avg_align) != 0 else 0.0
            # # self.add_star("Alignment Factor")
            # #   I tend to think this is important.

    # No problem adding functions, as well. This one useful for Node inheriters.
    def add_generic_node_stats(self):
        # Node Count
        self.data_dict["Count"] = len(self.clusters)

        if len(self.clusters) > 0:
            # Span Stats
            self._compute_list_stats([n.distance for n in self.clusters],
                "Span", self.data_dict)

    # These exist to allow getting of node-specific information with the
    #   assurance that it has been filled in.
    def get_string_node_dict(self, **kwargs):
        self.calculate(recalculate_all=False, **kwargs)
        return self._string_node_dict

    def find_nodes_from_string_list(self, string_list, **kwargs):
        # Finds nodes this list completely or even partially intersects.
        self.calculate(recalculate_all=False, **kwargs)
        node_strings = set(string_list).intersection(self._string_node_dict)
        return [self._string_node_dict[string] for string in node_strings]