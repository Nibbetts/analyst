from tqdm import tqdm

from ..clustertypes.cluster import Cluster
from .clusterizer import Clusterizer

class HubClusterizer(Clusterizer, object):

    def __init__(self, threshold=4, category="Hubs"):
        #   Notice we can add custom parameters.

        # node_clusterizer: requires node info to work. Don't make a new
        #   node clusterizer either, feed it the same instance, or everything
        #   will be recomputed.
        # threshold: how big a hub has to be to be counted.
        #   Ex: threshold=4 means the object plus 3 more for whom it is nearest.

        super(HubClusterizer, self).__init__(category)
        self.threshold = threshold
        self.s_to_node = None

    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        strings = kwargs["strings"]
        neighbors = kwargs["nearest_neighbors_ix"]
        nearest = kwargs["nearest_fn"]
        metric = kwargs["metric_fn"]
        encoder = kwargs["encoder_fn"]
        clusterizer_getter = kwargs["find_clusterizer_by_category_fn"]

        # No need to make sure Nodes are computed before Supernodes,
        #   since get_nodes ensures this for us, without repeating calculation:
        node_clusterizer = clusterizer_getter("Nodes")
        self.s_to_node = node_clusterizer.get_string_node_dict()

        # Calculate potential hubs:
        temp_hubs = []
        for i in tqdm(range(len(space)),
                desc="Finding Galactic Hubs (Finding Potential Hubs)",
                disable=(not show_progress)):
            temp_hubs.append(Cluster(
                encoder, metric, nearest=nearest,
                objects=[strings[i]], nodes=[], auto=False,
                name=strings[i]))
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
            if len(h) >= self.threshold:
                self.clusters.append(h)
                h.ID = j
                h.nodes = ([self.s_to_node[h.name]]
                    if h.name in self.s_to_node else [])
                h.calculate()
                j += 1

    # Even though we have already filled in self.clusters, we needn't override
    #   vectors_to_clusters which does so, since it checks for this.

    # Needn't override compute_stats, since generic cluster stats OK for hubs.