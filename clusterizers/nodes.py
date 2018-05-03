from tqdm import tqdm
import numpy as np
import scipy.spatial as sp

from ..clustertypes.node import Node
from .clusterizer import Clusterizer

class NodeClusterizer(Clusterizer, object):

    def __init__(self, category="Nodes", generic_stats=False):
        super(NodeClusterizer, self).__init__(category, generic_stats)
        # NOTE: We don't need the generic stats for simple pairs of objects.
        self.computed = False
        #   Built to prevent recalculation (if you must, then make a new one).

    def compute_clusters(
            self, vectors, show_progress=True, **kwargs):
        if self.computed: return

        strings = kwargs["strings"]
        encode = kwargs["encoder_fn"]
        metric = kwargs["metric_fn"]
        nearest = kwargs["nearest_neighbors_ix"]

        # Compute Nodes:
        self.clusters = [
            Node(strings[i], strings[nearest[i]], encode, metric)
            for i in tqdm(
                range(len(strings)),
                desc="Watching the Galaxies Coelesce (Computing Nodes)",
                disable=(not show_progress))
            if (i == nearest[nearest[i]]
                and i < nearest[i])]

        # Useful data to store:
        self.string_node_dict = {}
        for node in self.clusters:
            self.string_node_dict[node[0]] = node
            self.string_node_dict[node[1]] = node

        # Don't repeat calculations
        self.computed = True

    # Overriding
    def vectors_to_clusters(self, **kwargs):
        # self.clusters is already filled in,
        #   so we override this function with an empty version.
        return

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        printer = kwargs["printer_fn"]
        space = kwargs["space"]

        if self.generic_stats:
            # Node Count
            self.data_dict["Count"] = len(self.clusters)

            if len(self.clusters) > 0:
                # Span Stats
                self._compute_list_stats([n.span for n in self.clusters],
                    "Span", self.data_dict)

                # Nodal Factor
                printer("Comparing the Cosmos (Calculating Nodal Factor)")
                self.data_dict["Nodal Factor"] = (
                    len(self.clusters)*2.0/float(len(space)))
                self.add_star("Nodal Factor")
                #   I think this is important.

                # Alignment Factor
                printer("Musing over Magnetic Moments \
                    (Calculating Alignment Factor)")
                avg_align = np.mean(
                    [n.alignment for n in self.clusters], axis=0)
                avg_align /= np.linalg.norm(avg_align)
                self.data_dict["Alignment Factor"] = (
                    np.mean([
                        np.abs(sp.distance.cosine(avg_align, n.alignment))
                        for n in self.clusters]))
                self.add_star("Alignment Factor")
                #   I think this is important.