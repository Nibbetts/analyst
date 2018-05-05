from tqdm import tqdm

from ..clustertypes.cluster import Cluster
from .clusterizer import Clusterizer

class HubClusterizer(Clusterizer, object):

    def __init__(self, threshold=4, nodal=True, category="Nodal 4-Hubs",
            node_category="Nodes"):
        #   Notice we can add custom parameters.

        # threshold: how big a hub has to be to be counted.
        #   Ex: threshold=4 means the object plus 3 more for whom it is nearest.
        # nodal: whether or not to require a node to be present in the hub.
        #   If False, hubs may include branching points (Thinking about
        #   the data as a nearest-neighbor directed graph), and not just
        #   potential cluster centers.

        super(HubClusterizer, self).__init__(category, node_category)
        self.threshold = threshold
        self.nodal = nodal

    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        strings = kwargs["strings"]
        neighbors = kwargs["kth_neighbors_ix_fn"]
        nearest = kwargs["generic_nearest_fn"]
        metric = kwargs["metric_fn"]
        encoder = kwargs["encoder_fn"]
        evaluator_getter = kwargs["find_evaluator_fn"]

        # No need to make sure Nodes are computed before Hubs,
        #   since get_nodes ensures this for us, without repeating calculation:
        node_clusterizer = evaluator_getter(self.node_category,
            force_creation=self.nodal)
        s_to_node = node_clusterizer.get_string_node_dict() \
            if node_clusterizer != None else None

        # Calculate potential hubs:
        temp_hubs = []
        for i in tqdm(range(len(space)),
                desc="Finding Galactic Hubs (Finding Potential Hubs)",
                disable=(not show_progress)):
            temp_hubs.append(Cluster(
                encoder, metric, nearest=nearest,
                objects=[strings[i]], nodes=[], auto=False, name=strings[i]))
                # Its name is the original object's decoded string.
            for index, neighbor in enumerate(neighbors):
                if neighbor == i:
                    temp_hubs[i].add_objects([strings[index]])
                # The 0th index in the hub's list of objects is also
                #   it's original object (is included in hub).

        # Find the real, neighbor-limited hubs:
        j = 0
        for h in tqdm(temp_hubs,
                desc="Erecting Centers of Commerce (Finding Hubs)",
                disable=(not show_progress)):
            is_nodal = self.nodal and h.name in s_to_node
            if len(h) >= self.threshold and (is_nodal or not self.nodal):
                self.clusters.append(h)
                h.ID = j
                h.nodes = ([s_to_node[h.name]] if is_nodal else [])
                h.calculate()
                j += 1

    # Even though we have already filled in self.clusters, we needn't override
    #   vectors_to_clusters which does so, since it checks for this.

    # Needn't override compute_stats, since generic cluster stats OK for hubs.