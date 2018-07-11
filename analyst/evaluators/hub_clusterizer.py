from tqdm import tqdm
import ray
import psutil

from ..clustertypes.cluster import Cluster
from .clusterizer import Clusterizer

class HubClusterizer(Clusterizer, object):

    def __init__(self, threshold=4, nodal=True, category="Nodal 4-Hubs",
            starred=None, node_category="Nodes"):
        #   Notice we can add custom parameters.

        # threshold: how big a hub has to be to be counted.
        #   Ex: threshold=4 means the object plus 3 more for whom it is nearest.
        # nodal: whether or not to require a node to be present in the hub.
        #   If False, hubs may include branching points (Thinking about
        #   the data as a nearest-neighbor directed graph), and not just
        #   potential cluster centers.

        super(HubClusterizer, self).__init__(
            category=category, starred=starred, node_category=node_category)
        self.threshold = threshold
        self.nodal = nodal

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
            force_creation=self.nodal)
        s_to_node = self.node_clusterizer.get_string_node_dict(**kwargs) \
            if self.node_clusterizer != None else None
        assert self.node_clusterizer.calculated == True
        neighbors = neighbors_fn(1)

        # Calculate potential hubs:

        # PARALLELIZATION:
        printer("Finding Galactic Hubs", "Finding Potential Hubs")

        try: ray.init()
        except: pass

        @ray.remote
        def find_hub(i, strings, neighbors):
            objects = [strings[i]]
            for index, neighbor in enumerate(neighbors):
                if neighbor == i:
                    objects.append(strings[index])
                # The 0th index in the hub's list of objects is also
                #   its original object (is included in hub).
            return objects

        strings_id = ray.put(strings)
        neighbors_id = ray.put(neighbors)

        cpus = psutil.cpu_count()
        remaining_ids = [find_hub.remote(i, strings_id, neighbors_id) for i in
            range(min(len(space), cpus))]

        temp_hubs = []
        for i in tqdm(range(len(space)), disable=not show_progress):
            ready_ids, remaining_ids = ray.wait(remaining_ids)
            objects = ray.get(ready_ids[0])
            if i + cpus < len(space):
                remaining_ids.append(find_hub.remote(
                    i + cpus, strings_id, neighbors_id))
            temp_hubs.append(Cluster(
                self.CATEGORY, encoder, metric, objects, nearest=nearest,
                nodes=[], auto=False, name=objects[0], **metric_args))

        # NON-PARALLELIZED:
        # temp_hubs = []
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

        # Find the real, neighbor-limited hubs:
        j = 0
        printer("Erecting Centers of Commerce", "Culling for Actual Hubs")
        for h in tqdm(temp_hubs, disable=(not show_progress)):
            is_nodal = self.nodal and h.name in s_to_node
            if len(h) >= self.threshold and (is_nodal or not self.nodal):
                self.clusters.append(h)
                h.ID = j
                h.nodes = ([s_to_node[h.name]] if is_nodal else [])
                h.calculate()
                j += 1

    # Even though we have already filled in self.clusters, we needn't override
    #   vectors_to_clusters which does so, since it checks for this.

    def compute_stats(self, **kwargs):
        # Run the basic stats first:
        super(HubClusterizer, self).compute_stats(**kwargs)

        # TODO:
        # Then add our own:
        #!!!!add uniformity of density, by comparing hub dispersion range to space dispersion range, and maybe ln or sqrt to invert relationship?