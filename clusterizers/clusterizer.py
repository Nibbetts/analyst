import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import abstractmethod

from ..clustertypes.cluster import Cluster

"""
        .          .     .
      .  ..  .        .            .
     .. :...   .         .            ..            .
      . .::  :    ..  ..  
    .    ..   .      .:: ..     .
        .    .     . .: .                 
                :      .   .               .
      ..            .       .
            . :                     =-o                  *
           . . : .        .             .
     .       .           .  .
          .           .        "Space, the Final Frontier..."
"""

class Clusterizer:
    """
    What: A parent class for customizeable clusterizer objects which do the
        heavy lifting for the Analyst's spatial analyses.
    Why: Clusterizers exist to allow you to do custom analyses on an
        embedding space, or to perform custom clustering algorithms on them.
        Unless overridden, the built-in functions will perform basic analyses,
        and the Analyst will then collect and display the data for you.
    How: Create a custom clusterizer class which inherits from Clusterizer,
        or from a class performing things you want. An instance of your class,
        initialized with your parameters, can then be fed into a new Analyst
        instance. Built-ins that take parameters can also be fed in with custom
        parameters, instead of using the defaults that the Analyst would.
    Where: Now, really. You can't possibly think I'll tell you where to find the
        buried treasure, now can you?
    Who: Author and programmer is Nathan Tibbetts.
        Research Mentors: Nancy Fulda, David Wingate.
    """
    # An abstract class for clusterizers to inherit from.

    def __init__(self, category="Cluster"):
        self.category = category
        #   String, name of this cluster type, typically in plural form.
        #   This is allowed to be overridden per instance, since instances may
        #   be given different starting parameters, thus needing new labels.
        self.clusters = []
        #   Should contain clusters after vectors_to_clusters is called.
        #   However, since the Analyst only collects stats, it is only
        #   for detailed inspection through the Analyst.
        self.vector_groups = []
        #   Should contain a list of lists of vectors after compute_clusters is
        #   called. Also only used for detailed inspection, or is used if
        #   vectors_to_clusters is not overridden.
        self.data_dict = {}
        #   Additional information to be included in the printed report.
        #   Keys are attribute name strings, vals are ints, floats, strings, or
        #   lists if "Histogram Key" in key.
        #   NOTE: treat attribute names as case sensitive.
        self.starred = []
        #   Names of attributes which should be marked with a star in report.
    
    # TO BE OVERRIDDEN IN EVERY CASE - SUPER CALL UNNEEDED
    @abstractmethod
    def compute_clusters(self, space, show_progress=True, **kwargs):
        # space: the entire embedding space, a vector of vectors
        # show_progress: whether or not to show a progress bar etc.
        # Available kwargs (always given in case needed):
        #   space, show_progress, strings, metric_str, printer_fn, metric_fn,
        #   encoder_fn, decoder_fn, nearest_neighbors_ix_getter_fn,
        #   second_nearest_neighbors_ix_getter_fn,
        #   furthest_neighbors_ix_getter_fn, nearest_neighbors_dist_getter_fn,
        #   second_nearest_neighbors_dist_getter_fn,
        #   furthest_neighbors_dist_getter_fn, distance_matrix_getter_fn,
        #   clusterizer_list, find_clusterizer_by_category_fn.
        # NOTE: Excuse the long names. They should be fairly clear.
        #   We work with getters because the Analyst ensures 0 or 1 time
        #   calculations of things on a need-to-know basis.
        # NOTE: The distance_matrix_getter_fn exists to prevent recalculation
        #   of the distance matrix, which is computationally intensive.
        #   It will not be calculated unless you or built-ins use it.
        #   Unless you can't unpack your cluster algorithm, you should not
        #   need to calculate this yourself. Remember, if you are using and
        #   comparing multiple distance metrics, each should be run in a
        #   different analyst instance.
        # NOTE: By the time this function finishes, self.vector_groups must be
        #   filled in, but self.clusters doesn't need to be.
        pass

    # TO BE OVERRIDDEN IF NEEDED - DO NOT CALL SUPER
    def vectors_to_clusters(self, **kwargs):
        # kwargs: includes all the same as above, in case needed.
        # This function will automatically turn groups of vectors into clusters,
        #   unless you have already filled in self.clusters.
        if self.clusters == []:
            encoder    = kwargs["encoder_fn"]
            decoder    = kwargs["decoder_fn"]
            metric     = kwargs["metric_fn"]
            nearest    = kwargs["nearest_fn"]
            printer    = kwargs["printer_fn"]
            clusterizer_getter = kwargs["find_clusterizer_by_category_fn"]

            nodes = None
            self.node_clusterizer = clusterizer_getter(
                "Nodes", force_creation=False)

            # Allowing that clusterizer_getter may return None, in case user
            #   doesn't want Nodes calculated.
            if self.node_clusterizer != None:
                string_node_dict = self.node_clusterizer.get_string_node_dict()

                for i, group in enumerate(self.vector_groups):
                    objects = map(decoder, group)
                    nodes = map(string_node_dict, objects)
                    self.clusters.append(Cluster(encoder, metric, objects,
                        nearest=nearest, vectors=group, nodes=nodes, auto=True,
                        ID=i, name=None))
            else:
                printer("WARNING: 'Nodes' category not found! Clusters will \
                    have no information on contained Nodes.")
                nodes = []
                for i, group in enumerate(self.vector_groups):
                    objects = map(decoder, group)
                    self.clusters.append(Cluster(encoder, metric, objects,
                        nearest=nearest, vectors=group, nodes=nodes, auto=True,
                        ID=i, name=None))

    # TO BE OVERRIDDEN IF NEEDED - SHOULD CALL SUPER
    def compute_stats(self, **kwargs):
        # kwargs: includes all the same as above in case needed.
        # Do generic cluster stat stuff here; override if unwanted,
        #   or if you want both then override and call super.
        self.data_dict["Count"] = len(self.clusters)
        if len(self.clusters) > 0:
            self._compute_list_stats(map(len, self.clusters),
                "Population",  self.data_dict)
            self._compute_list_stats([c.centroid_length \
                for c in self.clusters], "Centroid Norm", self.data_dict)
            self._compute_list_stats([c.dispersion for c in self.clusters],
                "Dispersion",  self.data_dict)
            self._compute_list_stats([c.std_dev    for c in self.clusters],
                "Standard Dev", self.data_dict)
            self._compute_list_stats([c.repulsion  for c in self.clusters],
                "Repulsion",   self.data_dict)
            self._compute_list_stats([c.skew       for c in self.clusters],
                "Skew",        self.data_dict)
            if self.node_clusterizer != None:
                self._compute_list_stats([len(c.nodes) for c in self.clusters],
                    "Node Count",  self.data_dict)

    # Use this to indicate importance of a particular attribute in report;
    #   to be called from compute_stats preferrably.
    def add_star(self, attribute):
        # attribute: string name of attribute
        self.starred.append(attribute)

    # The Analyst will call this function, which pulls it all together.
    #   You shouldn't have to override this function:
    def calculate(
            self, space, show_progress=True, **kwargs):
        kwargs["space"] = space
        kwargs["show_progress"] = show_progress

        self.compute_clusters(space, show_progress, **kwargs)
        self.vectors_to_clusters(**kwargs)
        self.compute_stats(**kwargs)

        # Returning these means the Analyst need only access datamembers
        #   directly if we are doing specific inspections.
        return self.data_dict, self.starred, self.category

    @staticmethod
    def _compute_list_stats(vals, attribute, dictionary=None):
        # Used for a single stat across multiple clusters
        # vals: array-like containing the given attribute for each cluster.
        # attribute: name string, ie. "Dispersion".
        # dictionary: optional dictionary to add to instead of making a new one.

        stats = {} if dictionary == None else dictionary
        
        val_max = np.max(vals)
        val_min = np.min(vals)
        stats[attribute + " Avg"          ] = np.mean(vals)
        stats[attribute + " Min"          ] = val_min
        stats[attribute + " Max"          ] = val_max
        stats[attribute + " Range"        ] = val_max-val_min
        stats[attribute + " Standard Dev" ] = np.std(vals)
        stats[attribute + " Histogram Key"] = vals

        return stats


# def compute_hubs(metric_fn, encoder_fn, nearest_fn, nearest_neighbors_ix,
#                  strings, string_node_map, show_progress=True):
#     hubs = []
#     temp_hubs = []
#     for i in tqdm(range(len(strings)),
#             desc="Finding Galactic Hubs",
#             disable=(not show_progress)):
#         temp_hubs.append(clusters.Cluster(
#             encoder_fn, metric_fn, nearest=nearest_fn,
#             objects=[strings[i]], nodes=[], auto=False,
#             name=strings[i]))
#             # Its name is the original object's decoded string.
#         for index, neighbor in enumerate(nearest_neighbors_ix):
#             if neighbor == i:
#                 temp_hubs[i].add_objects([strings[index]])
#             # The 0th index in the hub's list of objects
#             #   is also it's original object (is included in hub).
#     j = 0
#     for h in tqdm(temp_hubs, desc="Erecting Centers of Commerce",
#             disable=(not show_progress)):
#         if len(h) >= 4: # obj plus 3 or more for whom it is nearest.
#             hubs.append(h)
#             h.ID = j
#             h.nodes = ([string_node_map[h.name]]
#                 if h.name in string_node_map else [])
#             h.calculate()
#             j += 1
#     return hubs


# def compute_supernodes(nodes, printer_fn, metric_str, metric_fn,
#                        show_progress=True):
#     centroids = [n.centroid for n in nodes]
#     printer_fn("Fracturing the Empire")
#     dist_matrix = sp.distance.squareform(
#         sp.distance.pdist(
#             centroids,
#             metric_str if metric_str != None else metric_fn))
#     printer_fn("Establishing a Hierocracy")
#     neighbors = np.argmax(dist_matrix, axis=1)
#     #neighbors_dist = dist_matrix[range(len(dist_matrix)), neighbors]

#     # Compute the Supernodes:
#     return [
#         clusters.Node(node,
#             nodes[neighbors[i]],
#             clusters.Node.get_centroid, metric_fn)
#         for i, node in enumerate(tqdm(nodes,
#             desc="Ascertaining Universe Filaments",
#             disable=(not show_progress)))
#         if (i == neighbors[neighbors[i]]
#             and i < neighbors[i])]

# def compute_nuclei():
#     pass

# def compute_chains():
#     pass

# def compute_NCC():
#     pass

# def compute_LNCC():
#     pass

# def compute_anti_hubs():
#     pass


# def compute_nodes(metric_fn, encoder_fn, nearest_neighbors_ix,
#                   strings, show_progress=True):
#     return [
#         clusters.Node(strings[i],
#             strings[nearest_neighbors_ix[i]],
#             encoder_fn,
#             metric_fn)
#         for i in tqdm(
#             range(len(strings)),
#             desc="Watching the Galaxies Coelesce",
#             disable=(not show_progress))
#         if (i == nearest_neighbors_ix[nearest_neighbors_ix[i]]
#             and i < nearest_neighbors_ix[i])
#     ]