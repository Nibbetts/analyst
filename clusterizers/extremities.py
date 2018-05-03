from tqdm import tqdm

from ..clustertypes.node import Node
from .nodes import NodeClusterizer

class ExtremityClusterizer(NodeClusterizer, object):

    def __init__(self, category="Extremities", generic_stats=False):
        super(ExtremityClusterizer, self).__init__(category, generic_stats)
        # NOTE: We don't need the generic stats for simple pairs of objects.

    # Overriding
    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        # The use of **kwargs allows you to pull many pre-computed types
        #   of data from the analyst to save on computation/programming time:
        strings = kwargs["strings"]
        encode = kwargs["encoder_fn"]
        metric = kwargs["metric_fn"]
        furthest = kwargs["furthest_neighbors_ix"]

        # Compute Extremities:
        self.clusters = [
            Node(strings[i], strings[furthest[i]], encode, metric)
            for i in tqdm(
                range(len(space)),
                desc="Measuring the Reaches (Computing Extremities)",
                disable=(not show_progress))
            if (i == furthest[furthest[i]]
                and i < furthest[i])]

        # Get their vectors (Though unneeded in built-in functionality):
        self.vector_groups = [
            [node.vec_a, node.vec_b] for node in self.clusters]

    # Don't need to override vectors_to_clusters, since NodeClusterizer does,
    #   and it is parent.

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        if self.generic_stats:
            # Extremity Count
            self.data_dict["Count"] = len(self.clusters)
            self.add_star("Count")

            # Span Stats
            if len(self.clusters) > 0:
                self._compute_list_stats([n.span for n in self.clusters],
                    "Span", self.data_dict)
                
                # We can add stars to things we think are important:
                self.add_star("Span Min")
                self.add_star("Span Max")