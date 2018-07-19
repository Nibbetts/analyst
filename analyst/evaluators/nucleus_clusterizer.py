from tqdm import tqdm

from .clusterizer import Clusterizer

class NucleusClusterizer(Clusterizer, object):

    def __init__(self, category="Nuclei", starred=None,
            node_category="Nodes", hub_category="Hubs"):

        super(NucleusClusterizer, self).__init__(
            category=category, starred=starred, node_category=node_category)

        self.hub_category = hub_category

    def compute_clusters(self, space, show_progress=True, **kwargs):
        metric           = kwargs["metric_fn"]
        evaluator_getter = kwargs["find_evaluator_fn"]
        printer          = kwargs["printer_fn"]
        #metric_args      = kwargs["metric_args"]

        # No need to make sure Hubs are computed before Nuclei,
        #   since get_clusters ensures this for us:
        self.hub_clusterizer = evaluator_getter(self.hub_category,
            force_creation=True)
        # get_clusters needs **kwargs to force calculation!
        hubs = self.hub_clusterizer.get_clusters(**kwargs)

        # Calculate Nuclei:
        hi_to_ni = {}
        printer("Melding Collided Galaxies", "Finding Nuclei")
        for i in tqdm(range(len(hubs)), disable=(not show_progress)):
            
            # Make a nucleus for hub i and record it:
            hi_to_ni[i] = len(self.clusters)
            # making a copy of the object, 
            new = hubs[i].modifiable_copy(auto=False)
            # Note: subcluster_ids are empty for hubs already.
            self.clusters.append(new)

            # for j in range(i):
            #     if hi_to_ni[i] != hi_to_ni[j]:
            #         threshold = max(
            #             hubs[i].stats_dict["Dispersion"],
            #             hubs[j].stats_dict["Dispersion"])
            #         if metric(hubs[i].nodes[0].centroid, 
            #                     hubs[j].nodes[0].centroid) <= threshold \
            #                 and metric(hubs[i].centroid, hubs[j].centroid) \
            #                 <= threshold:
            #             self.clusters[hi_to_ni[i]] += self.clusters[hi_to_ni[j]]
            #             self.clusters[hi_to_ni[i]].stats_dict["Hub Count"] += \
            #                 self.clusters[hi_to_ni[j]].stats_dict["Hub Count"]
            #             self.clusters[hi_to_ni[j]] = self.clusters[hi_to_ni[i]]
            #             hi_to_ni[j] = hi_to_ni[i]

            for j in range(i):
                # if not already combined, else we don't care:
                if hi_to_ni[i] != hi_to_ni[j]:
                    # # If interior nodes are close enough:
                    #threshold = max(
                    #    hubs[i].stats_dict["Dispersion"],
                    #    hubs[j].stats_dict["Dispersion"])
                    if hubs[i].nodes[0] == hubs[j].nodes[0]: #\
                            #or (metric(hubs[i].centroid, hubs[j].centroid) <= \
                            #threshold and metric(hubs[i].nodes[0].centroid,
                            #hubs[j].nodes[0].centroid) <= threshold):
                        # (remember, hubs have only one node)
                        # combine them, delete the 2nd, re-key the dict
                        self.clusters[hi_to_ni[j]] += self.clusters[hi_to_ni[i]]
                        self.clusters[hi_to_ni[i]] = self.clusters[hi_to_ni[j]]
                        hi_to_ni[i] = hi_to_ni[j]


        # Tell the nuclei which hubs they were built from:
        for h in hi_to_ni:
            self.clusters[hi_to_ni[h]].subcluster_ids.append(h)
        # Remove duplicate clusters from conglomeration algorithm:
        self.clusters = list(set(self.clusters)) # changes order, but OK, else:

        # Compute cluster stats:
        #   (not done when added, since conglomeration is iterative)
        for i, c in enumerate(self.clusters):
            c.SUBCLUSTER_CATEGORY = self.hub_category
            c.CATEGORY = self.CATEGORY
            c.ID = i
            c.calculate()

        self.add_star("Population Standard Dev")
        self.add_star("Dispersion Avg")
        self.add_star("Dispersion Range")
        self.add_star("Repulsion Avg")
        self.add_star("Count")

    # Even though we have already filled in self.clusters, we needn't override
    #   vectors_to_clusters which does so, since it checks for this.

    # def compute_stats(self, **kwargs):
    #     # Run the basic stats first:
    #     super(NucleusClusterizer, self).compute_stats(**kwargs)

    #     # TODO:
    #     # Then add our own:
    #     #!!!!add uniformity of density, by comparing hub dispersion range to space dispersion range, and maybe ln or sqrt to invert relationship?