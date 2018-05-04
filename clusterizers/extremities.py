from tqdm import tqdm

from ..clustertypes.node import Node
from .nodes import NodeClusterizer

class ExtremityClusterizer(NodeClusterizer, object):

    def __init__(self, category="Extremities"):
        super(ExtremityClusterizer, self).__init__(category)
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

    # Don't need to override vectors_to_clusters; if needed, parent would have.

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        self.add_generic_node_stats()
        # We can add stars to things we think are important:
        self.add_star("Count")
        self.add_star("Span Min")
        self.add_star("Span Max")