from tqdm import tqdm

from ..clustertypes.cluster import Cluster
from .clusterizer import Clusterizer

class NucleusClusterizer(Clusterizer, object):

    def __init__(self, category="Nuclei", starred=None,
            node_category="Nodes", hub_category="Nodal 4-Hubs"):

        super(NucleusClusterizer, self).__init__(
            category=category, starred=starred, node_category=node_category)

        self.hub_category = hub_category
        self.node_category = node_category

    def compute_clusters(
            self, space, show_progress=True, **kwargs):
        metric_fn        = kwargs["metric_fn"]
        evaluator_getter = kwargs["find_evaluator_fn"]
        metric_args      = kwargs["metric_args"]
        printer          = kwargs["printer_fn"]

        # No need to make sure Hubs are computed before Nuclei,
        #   since get_clusters ensures this for us:
        self.hub_clusterizer = evaluator_getter(self.hub_category,
            force_creation=True)
        hubs = self.hub_clusterizer.get_clusters()

        # Calculate Nuclei:
        hi_to_ni = {}
        printer("Melding Collided Galaxies", "Finding Nuclei")
        for i in tqdm(range(len(hubs)), disable=(not show_progress)):
            for j in range(i):
                # If interior nodes are close enough:
                if metric_fn(
                            hubs[i].nodes[0].centroid,
                            hubs[j].nodes[0].centroid) \
                        <= max(
                            hubs[i].stats_dict["Dispersion"],
                            hubs[j].stats_dict["Dispersion"]):
                    # if we've already added one of the hubs to a cluster:
                    if i in hi_to_ni:
                        # if we've actually already added both:
                        if j in hi_to_ni and hi_to_ni[i] != hi_to_ni[j]:
                            # combine them, delete the 2nd, re-key the dict
                            self.clusters[hi_to_ni[i]] += \
                                self.clusters[hi_to_ni[j]]
                            del self.clusters[hi_to_ni[j]]
                            hi_to_ni[j] = hi_to_ni[i]
                        # else if only added the first:
                        else: # add the second
                            hi_to_ni[j] = hi_to_ni[i]
                            self.clusters[hi_to_ni[i]] += hubs[j]
                    # or if we've only added the other, similar to the first:
                    elif j in hi_to_ni: # add the first
                        hi_to_ni[i] = hi_to_ni[j]
                        self.clusters[hi_to_ni[j]] += hubs[i]
                    # if both are new:
                    else: # add both
                        hi_to_ni[i] = len(self.clusters)
                        hi_to_ni[j] = len(self.clusters)
                        self.clusters.append(hubs[i] + hubs[j])

    # Even though we have already filled in self.clusters, we needn't override
    #   vectors_to_clusters which does so, since it checks for this.

    def compute_stats(self, **kwargs):
        # Run the basic stats first:
        super(NucleusClusterizer, self).compute_stats(**kwargs)

        # TODO:
        # Then add our own:
        #!!!!add uniformity of density, by comparing hub dispersion range to space dispersion range, and maybe ln or sqrt to invert relationship?