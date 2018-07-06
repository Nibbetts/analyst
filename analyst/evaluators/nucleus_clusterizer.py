from tqdm import tqdm
import copy

from .clusterizer import Clusterizer

class NucleusClusterizer(Clusterizer, object):

    def __init__(self, category="Nuclei", starred=None,
            node_category="Nodes", hub_category="Nodal 4-Hubs"):

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
            for j in range(i):
                # If interior nodes are close enough:
                threshold = max(
                        hubs[i].stats_dict["Dispersion"],
                        hubs[j].stats_dict["Dispersion"])
                # Currently either from nodes or from centroids will work:
                if metric(hubs[i].nodes[0].centroid, # These have exactly 1 Node
                            hubs[j].nodes[0].centroid) <= threshold \
                        or metric(hubs[i].centroid, hubs[j].centroid) \
                        <= threshold:
                    # if we've already added one of the hubs to a cluster:
                    if i in hi_to_ni:
                        # if we've actually already added both:
                        if j in hi_to_ni:
                            # if not already combined, else we don't care:
                            if hi_to_ni[i] != hi_to_ni[j]:
                                # combine them, delete the 2nd, re-key the dict
                                self.clusters[hi_to_ni[i]] += \
                                    self.clusters[hi_to_ni[j]]
                                self.clusters[hi_to_ni[i]] \
                                    .stats_dict["Hub Count"] \
                                    += self.clusters[hi_to_ni[j]] \
                                    .stats_dict["Hub Count"]
                                self.clusters[hi_to_ni[j]] = \
                                    self.clusters[hi_to_ni[i]]
                                #hi_to_ni[j] = hi_to_ni[i]
                        # else if only added the first:
                        else: # add the second
                            hi_to_ni[j] = hi_to_ni[i]
                            self.clusters[hi_to_ni[i]] += hubs[j]
                            self.clusters[hi_to_ni[i]] \
                                .stats_dict["Hub Count"] += 1
                    # or if we've only added the other, similar to the first:
                    elif j in hi_to_ni: # add the first
                        hi_to_ni[i] = hi_to_ni[j]
                        self.clusters[hi_to_ni[j]] += hubs[i]
                        self.clusters[hi_to_ni[j]].stats_dict["Hub Count"] += 1
                    # if both are new:
                    else: # add both
                        hi_to_ni[i] = len(self.clusters)
                        hi_to_ni[j] = len(self.clusters)
                        new = hubs[i] + hubs[j]
                        new.stats_dict["Hub Count"] = 2
                        self.clusters.append(new)
            # If i didn't get added, add it, because this will only happen if
            #   i and j were too far apart or i is 0, and j will come up later.
            if i not in hi_to_ni: # if i is new:
                hi_to_ni[i] = len(self.clusters)
                # making a copy of the object, 
                new = copy.copy(hubs[i])
                new.stats_dict["Hub Count"] = 1
                self.clusters.append(new)
        # Remove duplicate clusters from conglomeration algorithm:
        self.clusters = list(set(self.clusters)) # changes order, but OK, else:
        # filtered = []
        # for c in self.clusters:
        #     if c not in filtered: filtered.append(c)
        # self.clusters = filtered

        # Compute cluster stats:
        #   (not done when added, since conglomeration is iterative)
        for c in self.clusters:
            c.calculate()

    # Even though we have already filled in self.clusters, we needn't override
    #   vectors_to_clusters which does so, since it checks for this.

    # def compute_stats(self, **kwargs):
    #     # Run the basic stats first:
    #     super(NucleusClusterizer, self).compute_stats(**kwargs)

    #     # TODO:
    #     # Then add our own:
    #     #!!!!add uniformity of density, by comparing hub dispersion range to space dispersion range, and maybe ln or sqrt to invert relationship?