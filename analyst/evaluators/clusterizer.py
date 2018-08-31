from abc import abstractmethod
import numpy as np

from ..clustertypes.cluster import Cluster
from .evaluator import Evaluator


class Clusterizer(Evaluator, object):
    """
    What: An abstract, parent class for customizeable clusterizer objects which
        do the heavy lifting for the Analyst's spatial analyses.
    Why: Clusterizers exist to allow you to invoke custom clustering algorithms
        on your embedding spaces. Unless overridden, the built-in functions
        will perform basic analyses.
    How: Create a custom clusterizer class which inherits from Clusterizer,
        or from a derived class doing things you want. An instance of your
        class, initialized with your parameters, can then be fed into a new
        Analyst instance. Built-ins that take parameters can also be fed in with
        custom parameters, instead of using the defaults that the Analyst would.
    """

    def __init__(self, category, starred=None, node_category="Nodes"):
        super(Clusterizer, self).__init__(category=category, starred=starred)
        #   To inherit, must call parent init.
        self.node_category = node_category
        #   Name by which to search for the proper node clusterizer to be used
        #   to fill in info on which nodes are contained in each cluster.
        # self.CATEGORY = category
        #   String, name of this cluster type, typically in plural form.
        #   This is allowed to be overridden per class or per instance,
        #   since instances may be given different starting parameters,
        #   thus needing new labels to disambiguate.
        #   NOTE: Treat categories as case-sensitive.
        self.clusters = []
        #   Should contain clusters after vectors_to_clusters is called.
        #   However, since the Analyst only collects stats, it is only
        #   for detailed inspection through the Analyst.
        self.vector_groups = []
        #   Should contain a list of lists of vectors after compute_clusters is
        #   called. Also only used for detailed inspection, or is used if
        #   vectors_to_clusters is not overridden.
        # self.stats_dict = OrderedDict()
        #   Additional information to be included in the printed report.
        #   Keys are attribute name strings, vals are ints, floats, strings, or
        #   lists if "Histogram Key" in key.
        #   NOTE: treat attribute names as case-sensitive.
        # self.starred = []
        #   Names of attributes which should be marked with a star in report.
        # self.calculated = False
        #   Whether or not we have already calculated everything.
        #   Prevents unintentional repeat calculations.
    

    # TO BE OVERRIDDEN IN EVERY CASE - SUPER CALL UNNEEDED
    @abstractmethod
    def compute_clusters(self, space, show_progress=True, **kwargs):
        # space: the entire embedding space, a vector of vectors
        # show_progress: whether or not to show a progress bar etc.
        # (These two given are also included in kwargs, but with the names
        #   embeddings and draw_progress. Duplicated as real arguments because
        #   they are assumed to be very commonly used.)
        # Available kwargs (always given in case needed) listed in parent class,
        #   Evaluator.
        # NOTE: Excuse the long names. They should be fairly clear.
        #   We work with getters because the Analyst ensures something will be
        #   calculated only 0 or 1 times, on a need-to-know basis.
        # NOTE: Remember, if you are using and comparing multiple distance
        #   metrics, or multiple embedding spaces, each should be run in a
        #   different analyst instance.
        # POST: By the time this function finishes, self.vector_groups must be
        #   filled in, but self.clusters doesn't need to be.
        pass


    # TO BE OVERRIDDEN IF NEEDED - DO NOT CALL SUPER
    def vectors_to_clusters(self, **kwargs):
        # kwargs: includes all the same as above, in case needed.
        # This function will automatically turn groups of vectors into clusters,
        #   unless you have already filled in self.clusters.
        #   I anticipate most clustering algorithms will simply produce groups
        #   of vectors, hence vectors_to_clusters converts them for you.
        # PRE: self.compute_clusters must have been called.
        # POST: self.clusters must be filled in.
        if self.clusters == []:
            encoder        = kwargs["encoder_fn"]
            decoder        = kwargs["decoder_fn"]
            metric         = kwargs["metric_fn"]
            nearest        = kwargs["generic_nearest_fn"]
            printer        = kwargs["printer_fn"]
            find_evaluator = kwargs["find_evaluator_fn"]
            metric_args    = kwargs["metric_args"]

            # It is acceptable and useful to make one clusterizer depend on
            #   results from another. It is a VERY BAD idea to try to make two
            #   depend on each other!
            self.node_clusterizer = find_evaluator(
                self.node_category, force_creation=False)

            # Allowing that find_evaluator may return None, in case user
            #   doesn't want Nodes calculated.
            s_to_node = None
            if self.node_clusterizer == None:
                printer("WARNING: " + self.node_category + " category not \
                    found! " + self.CATEGORY + " will have no information on \
                    contained Nodes.")
            else:
                s_to_node = self.node_clusterizer.get_string_node_dict(**kwargs)
                
            nodes = []
            for i, group in enumerate(self.vector_groups):
                objects = [decoder(v) for v in group]
                nodes = [s_to_node[o] for o in objects if o in s_to_node] \
                    if s_to_node != None else []
                self.clusters.append(Cluster(
                    self.CATEGORY, encoder, metric,
                    objects, nearest=nearest, vectors=group, nodes=nodes,
                    auto=True, ID=i, name=None, **metric_args))


    # TO BE OVERRIDDEN IF NEEDED - SHOULD CALL SUPER
    def compute_stats(self, **kwargs):
        # kwargs: includes all the same as above in case needed.
        # Do generic cluster stat stuff here; override if unwanted,
        #   or if you want both then override and call super.
        # PRE: self.clusters must be filled in.
        # POST: self.stats_dict will contain whatever you want reported,
        #   and self.starred likewise.
        scaler         = kwargs["scale_invariance_fn"]

        self.stats_dict["Count"] = len(self.clusters)
        
        if len(self.clusters) > 0:
            self.stats_dict["Subcluster Category"] = \
                self.clusters[0].SUBCLUSTER_CATEGORY
            skip = self.clusters[0].QUIET_STATS if \
                self.clusters[0].quiet_stats_override == None else \
                self.clusters[0].quiet_stats_override
            for key in self.clusters[0].stats_dict.keys():
                if key not in skip:
                    try:
                        l = [c.stats_dict[key] for c in self.clusters]
                        self._compute_list_stats(
                            l, key, self.stats_dict, si=(
                                "dispersion" if ("Avg" in key or "Max" in key
                                    or "Min" in key or "Range" in key
                                    or "Std" in key or "Norm" in key
                                    or "Dispersion" in key or "Dist" in key
                                    or "Standard Dev" in key or "Skew" in key
                                    or "Repulsion" in key) else None
                            ), **kwargs)
                        if "Node Count" in key:
                            l = np.array(l)
                            nodality = l * 2.0 / np.array(
                                [c.stats_dict["Population"] \
                                for c in self.clusters])
                            self._compute_list_stats(nodality, "PI Nodality",
                                self.stats_dict)
                        if "Population" in key:
                            l = np.array(l)
                            pop_ratio = [scaler(i, si="population") for i in l]
                            self._compute_list_stats(pop_ratio,
                                "PI Population Ratio", self.stats_dict)
                    except:
                        pass


    # The Analyst will call this function, which pulls it all together.
    #   You shouldn't have to override this function:
    def calculate(self, recalculate_all=False, **kwargs):
        # For explanations of parameters, see parent Evaluator class.
        if not self.calculated or recalculate_all:
            if kwargs == {}:
                print("NOT YET CALCULATED AND NO KWARGS GIVEN!")
            space = kwargs["embeddings"]
            show_progress = kwargs["draw_progress"]
            printer = kwargs["printer_fn"]
            
            if show_progress:
                printer("Evaluating " + self.CATEGORY)

            self.compute_clusters(space, show_progress, **kwargs)
            self.vectors_to_clusters(**kwargs)
            self.compute_stats(**kwargs)

            # Override default stars if user gave any:
            if self.given_stars != None:
                self.starred = self.given_stars

            self.calculated = True

        return self.stats_dict, self.starred, self.CATEGORY


    # These allow the retrieval of cluster information without having to worry
    #   about whether or not it has been filled in.
    #   No getter needed for CATEGORY since it should never change.
    #   Parent also has get_stats_dict and get_starred.
    def get_clusters(self, **kwargs):
        self.calculate(recalculate_all=False, **kwargs)
        return self.clusters

    def get_vector_groups(self, **kwargs):
        self.calculate(recalculate_all=False, **kwargs)
        return self.vector_groups
