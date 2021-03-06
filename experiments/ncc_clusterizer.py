from tqdm import tqdm

from ..clustertypes.cluster import Cluster
from .clusterizer import Clusterizer
"""
class NCCClusterizer(Clusterizer, object):

    def __init__(self, category="NCC", starred=None,
            node_category="Nodes", hub_category_nodal="Hubs",
            hub_category_branching="Branching Hubs"):
        #   Notice we can add custom parameters.

        super(NCCClusterizer, self).__init__(
            category=category, starred=starred, node_category=node_category)

    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        strings          = kwargs["strings"]
        neighbors_fn     = kwargs["kth_neighbors_ix_fn"]
        nearest          = kwargs["generic_nearest_fn"]
        metric           = kwargs["metric_fn"]
        encoder          = kwargs["encoder_fn"]
        evaluator_getter = kwargs["find_evaluator_fn"]
        metric_args      = kwargs["metric_args"]
        printer          = kwargs["printer_fn"]

        # No need to make sure Nodes are computed before Hubs,
        #   since get_nodes ensures this for us, without repeating calculation:
        self.node_clusterizer = evaluator_getter(self.node_category,
            force_creation=True)
        self.hub_clusterizer_nodal = evaluator_getter(self.hub_category_nodal,
            force_creation=True)
        self.hub_clusterizer_branching = evaluator_getter(
            self.hub_category_branching, force_creation=False)
        s_to_node = self.node_clusterizer.get_string_node_dict() #\
            #if self.node_clusterizer != None else None
        neighbors = neighbors_fn(1)

        # # Calculate potential hubs:
        # temp_hubs = []
        # printer("Finding Galactic Hubs", "Finding Potential Hubs")
        # for i in tqdm(range(len(space)), disable=(not show_progress)):
        #     temp_hubs.append(Cluster(
        #         self.CATEGORY, encoder, metric, nearest=nearest,
        #         objects=[strings[i]], nodes=[], auto=False, name=strings[i],
        #         **metric_args))
        #         # Its name is the original object's decoded string.
        #     for index, neighbor in enumerate(neighbors):
        #         if neighbor == i:
        #             temp_hubs[i].add_objects([strings[index]])
        #         # The 0th index in the hub's list of objects is also
        #         #   it's original object (is included in hub).

        # # Find the real, neighbor-limited hubs:
        # j = 0
        # printer("Erecting Centers of Commerce", "Culling for Actual Hubs")
        # for h in tqdm(temp_hubs, disable=(not show_progress)):
        #     is_nodal = self.nodal and h.name in s_to_node
        #     if len(h) >= self.threshold and (is_nodal or not self.nodal):
        #         self.clusters.append(h)
        #         h.ID = j
        #         h.nodes = ([s_to_node[h.name]] if is_nodal else [])
        #         h.calculate()
        #         j += 1

    # Even though we have already filled in self.clusters, we needn't override
    #   vectors_to_clusters which does so, since it checks for this.

    def compute_stats(self, **kwargs):
        # Run the basic stats first:
        super(NCCClusterizer, self).compute_stats(**kwargs)

        # Then add our own:
        #!!!!add uniformity of density, by comparing hub dispersion range to space dispersion range, and maybe ln or sqrt to invert relationship?

"""