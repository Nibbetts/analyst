from tqdm import tqdm

from ..clustertypes.node import Node
from .node_clusterizer import NodeClusterizer

class ExtremityClusterizer(NodeClusterizer, object):

    def __init__(self, category="Extremities", starred=None):
        super(ExtremityClusterizer, self).__init__(
            category=category, starred=starred)
        # NOTE: We don't need the generic stats for simple pairs of objects.

    # Overriding
    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        # The use of **kwargs allows you to pull many pre-computed types
        #   of data from the analyst to save on computation/programming time:
        strings     = kwargs["strings"]
        encode      = kwargs["encoder_fn"]
        metric      = kwargs["metric_in_model_fn"]
        neighbors   = kwargs["kth_neighbors_ix_fn"]
        metric_args = kwargs["metric_args"]
        printer     = kwargs["printer_fn"]

        # This is an array of indeces for each object's furthest neighbor.
        furthest  = neighbors(-1)

        # Compute Extremities:
        printer("Measuring the Reaches", "Computing Extremities")
        self.clusters = [
            Node(strings[i], strings[furthest[i]],
                encode, metric, **metric_args)
            for i in tqdm(range(len(space)), disable=(not show_progress))
            if (i == furthest[furthest[i]]
                and i < furthest[i])]

        # Get their vectors (Though unneeded in built-in functionality):
        self.vector_groups = [
            [node.vec_a, node.vec_b] for node in self.clusters]

    # Don't need to override vectors_to_clusters; if needed, parent would have.

    # Overriding (because nodes only have two vectors, need different stats)
    def compute_stats(self, **kwargs):
        space   = kwargs["embeddings"]
        printer = kwargs["printer_fn"]

        self.add_generic_node_stats()

        if len(self.clusters) > 0:
            # Nodal Factor
            printer("Placating the Extremists", "Calculating Extremity Factor")
            self.data_dict["Extremity Factor"] = (
                len(self.clusters)*2.0/float(len(space)))
            self.add_star("Extremity Factor")
            #   I tend to think this is important.

        # We can add stars to things we think are important:
        self.add_star("Count")
        self.add_star("Span Min")
        self.add_star("Span Max")
        self.add_star("Extremity Factor")
        # NOTE: These could also have been added by initializing the object with
        #   starred=["Count","Span Min","Span Max","Extremity Factor"]