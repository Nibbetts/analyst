from abc import abstractmethod
from tqdm import tqdm

from ..clustertypes.cluster import Cluster
from .evaluator import Evaluator


"""
        .          .     .
      .  ..  .        .            .
     .. :...   .         .            ..            .
      . .::  :    ..  ..    :   .
    .    ..   .      .:: ..     .
        .    .     . .: .                 
                :      .   .               .
      ..            .       .
            . :                     =-o                  *
           . . : .        .             .
     .       .           .  .
          .           .        "Space, the Final Frontier..."
"""


class Spatializer(Evaluator, object):
    """
    What: An class for general spatial analysis, to analyze the embedding space
        properties as a whole.
    Why: It seems to me that unless we better understand some quantitative
        properties of the embedding spaces we work with, it will be hard to
        know what to do to improve them, or why improvements we make work.
    How: You can override this if you want, to have custom stats, but otherwise
        just use it as a built-in. Instead of having to make a Spatializer
        instance, you can just put "Spatial" or "All" in as a string, and the
        Analyst will do it for you, as with all built-in evaluators.
    """

    def __init__(self, category="Spatial", node_category="Nodes", starred=None,
            neighbors_to_stat=None):
        super(Spatializer, self).__init__(category=category, starred=starred)
        #   To inherit, must call parent init.
        self.node_category = node_category
        self.neighbors_to_stat = neighbors_to_stat
        #   Can override the automatic behavior to stat all neighbors computed.

        # PARENT MEMBERS WE NEED:
        # self.CATEGORY
        # self.stats_dict
        # self.starred
        # self.calculated


    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # kwargs: see Evaluator class.
        # POST: self.stats_dict, self.starred filled in.
        encoder           = kwargs["encoder_fn"]
        metric            = kwargs["metric_fn"]
        metric_in_model   = kwargs["metric_in_model_fn"]
        metric_str        = kwargs["metric_str"]
        nearest           = kwargs["generic_nearest_fn"]
        printer           = kwargs["printer_fn"]
        find_evaluator    = kwargs["find_evaluator_fn"]
        metric_args       = kwargs["metric_args"]
        objects           = kwargs["strings"]
        space             = kwargs["embeddings"]
        scaler            = kwargs["scale_invariant_fn"]
        auto_print        = kwargs["draw_progress"]
        downstream        = kwargs["downstream_fn"]
        neighbors_dist    = kwargs["kth_neighbors_dist_fn"]
        neighbors_to_stat = kwargs["make_kth_neighbors"] \
            if self.neighbors_to_stat is None else self.neighbors_to_stat

        self.stats_dict["Distance Metric"] = metric_str

        # It is acceptable and useful to make one clusterizer depend on
        #   results from another. It is a BAD idea to try to make two
        #   depend on each other!
        self.node_clusterizer = find_evaluator(
            self.node_category, force_creation=False)

        # Allowing that find_evaluator may return None, in case user
        #   doesn't want Nodes calculated.
        nodes = []
        if self.node_clusterizer == None:
            printer("WARNING: " + self.node_category + " category not \
                found! " + self.CATEGORY + " will have no information on \
                contained Nodes")
        else:
            nodes = self.node_clusterizer.get_clusters(**kwargs)
        
        # Use the Cluster class to compute the main stats for us:
        printer("Balancing the Continuum", "Computing Common Spatial Stats")
        self.cluster = Cluster(self.CATEGORY, encoder, metric, objects,
            nearest=nearest, vectors=space, nodes=nodes, auto=True,
            **metric_args)

        if len(space) > 0:
            # Overall Info:
            self.stats_dict["Dimensionality"] = len(space[0])
            self.stats_dict["Population"] = len(space)
            printer("Electing a Ruler", "Getting Medoid, Etc.")
            self.stats_dict["Medoid - Obj Nearest to Centroid"] = \
                self.cluster.medoid
            
            skip = self.cluster.QUIET_STATS \
                if self.cluster.quiet_stats_override is None \
                else self.cluster.quiet_stats_override
            for key in self.cluster.stats_dict:
                if key not in skip:
                    self.stats_dict[key] = self.cluster.stats_dict[key]
                    if ("Avg" in key or "Max" in key or "Min" in key
                            or "Range" in key or "Std" in key or "Norm" in key
                            or "Dispersion" in key or "Dist" in key
                            or "Standard Dev" in key or "Skew" in key
                            or "Repulsion" in key):
                        self.stats_dict["SI " + key] = scaler(
                            self.cluster.stats_dict[key], si="dispersion")
                    elif key == "Node Count":
                        self.stats_dict["SI Nodal Factor"] = scaler(
                            self.cluster.stats_dict[key], si="nodal")

            # Centroid Info:
            printer("Setting Priorities", "Centroid Stats")
            self._compute_list_stats(self.cluster.centroid_distances,
                "Centroid Dist", self.stats_dict, si="dispersion", **kwargs)
            
            # Then re-key one entry:
            dispersion = self.stats_dict.pop("Centroid Dist Avg")
            self.stats_dict.pop("Dispersion")
            self.stats_dict["Dispersion - Centroid Dist Avg"] = dispersion
            self.stats_dict["SI Dispersion"] = \
                self.stats_dict.pop("SI Centroid Dist Avg")

            self._compute_list_stats(
                self.cluster.norms, "Norms", self.stats_dict, si="dispersion",
                **kwargs)
            
            # kth-Neighbors Distance Info:
            for n in neighbors_to_stat:
                if n == 1:  # Added here because this is an OrderedDict
                    printer("Building Trade Routes", "Nearest Neighbor Stats")
                    printer("Practicing Diplomacy")
                    self.stats_dict["Repulsion - Nearest Dist Avg"] = 0
                    self.stats_dict["SI Repulsion"] = 0
                if n == 2:  # For fun
                    printer("Coming up with Excuses", "Second Neighbor Stats")
                if n == -1: # Added here because this is an OrderedDict
                    printer("Making Enemies", "Furthest Neighbor Stats")
                    self.stats_dict["Broadness - Furthest Dist Max"] = 0
                    self.stats_dict["SI Broadness"] = 0
                self._compute_list_stats(neighbors_dist(n),
                    "Nghbr " + str(n) + " Dist", self.stats_dict,
                    si="dispersion", **kwargs)

            # Some Re-keying for Specificity:
            repulsion = self.stats_dict.pop("Nghbr 1 Dist Avg", None)
            broadness = self.stats_dict.pop("Nghbr -1 Dist Max", None)
            si_repulsion = self.stats_dict.pop("SI Nghbr 1 Dist Avg", None)
            si_broadness = self.stats_dict.pop("SI Nghbr -1 Dist Max", None)
            self.stats_dict.pop("Repulsion") # Because in cluster?
            printer("Claiming Frontiers", "Re-Keying Stuff")
            if repulsion is not None:
                self.stats_dict["Repulsion - Nearest Dist Avg"] = repulsion
                self.stats_dict["SI Repulsion"] = si_repulsion
            if broadness is not None:
                self.stats_dict["Broadness - Furthest Dist Max"] = broadness
                self.stats_dict["SI Broadness"] = si_broadness

            # Downstream Path Length stats:
            printer("Building a Caste-System", "Downstream Stats")
            downstreamness = []
            downstream_dist = []
            downstream_gradients = []
            for o in tqdm(objects, disable=not auto_print):
                path = downstream(o, give_path=True)
                downstreamness.append(len(path) - 2)
                # -2 because the last two make the node we come to, so that
                #   we get 0 if we are part of the node.
                downstream_dist.append(sum([
                        metric_in_model(path[i], path[i+1]) / repulsion \
                        for i in range(len(path)-2)])
                    if len(path) > 2 else 0.0)
                # This one is the actual distances added up to get there,
                #   divided by repulsion to generalize it across spaces.
                downstream_gradients.append(
                        (metric_in_model(path[0], path[1]) /
                        metric_in_model(path[-1], path[-2])) - 1 \
                    if downstreamness[-1] != 0 and len(path) >= 2 else 0.0)
                # This one is in some sense the slope of the gradient of
                #   expansion of space in the local region;
                #   it is a ratio of dist to nearest and node's width, -1 to
                #   make it so zero means no change in density as we go out.

            self._compute_list_stats(
                downstreamness, "Downstream Count", self.stats_dict)
            self._compute_list_stats(
                downstream_dist, "Downstream Distance", self.stats_dict,
                si="dispersion", **kwargs)
            self._compute_list_stats(
                downstream_gradients, "Downstream Density Gradient",
                self.stats_dict)

            # Add stars to things we think are important or more interesting:
            printer("Counting the Lightyears", "Adding Stars")
            self.add_star("Medoid - Obj Nearest to Centroid")
            self.add_star("Dispersion - Centroid Dist Avg")
            self.add_star("Repulsion - Nearest Dist Avg")
            self.add_star("Nearest Dist Range")
            self.add_star("Broadness - Furthest Dist Max")
            self.add_star("SI Dispersion - Centroid Dist Avg")
            self.add_star("SI Repulsion - Nearest Dist Avg")
            self.add_star("SI Nearest Dist Range")
            self.add_star("SI Broadness - Furthest Dist Max")



# class ScaleInvariantSpatializer(Spatializer, object):
#     """
#     Takes stats from the Spatializer and makes them scale invariant,
#     so that they are valid across multiple types of embedding spaces.
#     """

#     def __init__(self, category="Scale Invariant Spatial",
#             node_category="Nodes", starred=None, neighbors_to_stat=None,
#             spatializer_category="Spatial"):
#         super(ScaleInvariantSpatializer, self).__init__(
#             category=category, starred=starred)

#         self.spatializer_category=spatializer_category


#     # OVERRIDEABLE
#     def compute_stats(self, **kwargs):
#         # kwargs: see Evaluator class.
#         # POST: self.stats_dict, self.starred filled in.
#         find_evaluator = kwargs["find_evaluator_fn"]
#         printer        = kwargs["printer_fn"]
#         scaler         = kwargs["scale_invariant_fn"]

#         printer("Traversing the Multiverse",
#             "Scale-Invariant Spatial Stats")
#         spatializer = find_evaluator(self.spatializer_category,
#             force_creation=True)
#         spatial_stats = spatializer.get_stats_dict(**kwargs)
#         dispersion = spatial_stats["Dispersion - Centroid Dist Avg"]

#         # Keep all stats, and make conversions where necessary:
#         for key, value in spatial_stats.items():
#             if ("Avg" in key or "Max" in key or "Min" in key or "Range" in key
#                     or "Std" in key or "Norm" in key or "Dispersion" in key
#                     or "Dist" in key or "Standard Dev" in key or "Skew" in key
#                     or "Repulsion" in key)\
#                     and "Downstream Count" not in key \
#                     and "Downstream Density" not in key \
#                     and "Histogram Key" not in key \
#                     and "SI " not in key and "PI " not in key:
#                 # The last 2 are already invariant.
#                 self.stats_dict["SI " + key] = value / dispersion
#             elif key == "Node Count":
#                 self.stats_dict["SI Nodal Factor"] = scaler(value, si="nodal")
#             else:
#                 self.stats_dict[key] = value

#         printer("Looking to the Stars", "Adding Stars")
#         self.add_star("Medoid - Obj Nearest to Centroid")
#         self.add_star("SI Dispersion - Centroid Dist Avg")
#         self.add_star("SI Repulsion - Nearest Dist Avg")
#         self.add_star("SI Nearest Dist Range")
#         self.add_star("SI Broadness - Furthest Dist Max")