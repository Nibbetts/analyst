import numpy as np
from abc import abstractmethod
from collections import OrderedDict

from ..clustertypes.cluster import Cluster


class Evaluator:
    """
    What: An abstract, parent class for an Evaluator, which you can think of as
        an Analyst operator. Each Evaluator performs a specific analysis on the
        space, given various data, as well as access to the other Evaluators,
        and thus potentially to the results of their analyses. It is through
        Evaluators that the Analyst does all of the heavy lifting.
    Why: Evaluators exist to allow you to do built-in or custom analyses on an
        embedding space, or to invoke custom clustering algorithms on them.
        Unless overridden, the built-in functions will perform basic analyses,
        and the Analyst will then collect and display the data for you.
    How: Create a custom Evaluator class which inherits from this one,
        or from a class performing things you want. For example, if you want to
        use a new clustering algorithm, you would ideally inherit from the
        Clusterizer class or one of its subclasses. For analogical analyses, you
        would probably inherit from Analogizer. An instance of your class,
        initialized with your parameters, can then be fed into a new Analyst
        instance. Built-ins that take parameters can also be fed in with custom
        parameters, instead of using the defaults that the Analyst would.
    Where: Now, really. You can't possibly think I'll tell you where to find the
        buried treasure, now do you?
    Who: Author and programmer is Nathan Tibbetts, a student at BYU.
        Research Advisors/Mentors: David Wingate, Nancy Fulda.
    """

    def __init__(self, category, starred=None):
        self.CATEGORY = category
        #   String, name of this analysis or recorded stat category, ie:
        #       "Analogy Set Scores". Typically in plural form.
        #   This is allowed to be overridden per class or per instance,
        #   since instances may be given different starting parameters,
        #   thus needing new labels to disambiguate.
        #   NOTE: Treat categories as case-sensitive.
        self.data_dict = OrderedDict()
        #   Additional information to be included in the printed report.
        #   Keys are attribute name strings, vals are ints, floats, strings, or
        #   lists if "Histogram Key" in key.
        #   NOTE: treat attribute names as case-sensitive.
        self.given_stars = starred
        #   For overriding the default stars placed by compute_stats.
        self.starred = []
        #   Names of attributes which should be marked with a star in report.
        self.calculated = False
        #   Whether or not we have already calculated everything.
        #   Prevents unintentional repeat calculations.


    # TO BE OVERRIDDEN
    @abstractmethod
    def compute_stats(self, **kwargs):
        # Do generic cluster stat stuff here; override if unwanted,
        #   or if you want both then override and call super.
        # kwargs: includes all the same as the calculate function below.
        # POST: self.data_dict will contain whatever you want reported,
        #   and self.starred likewise.
        pass


    # Use this to indicate importance of a particular attribute in report;
    #   to be called from compute_stats preferrably.
    def add_star(self, attribute):
        # attribute: string name of attribute
        if self.given_stars == None:
            self.starred.append(attribute)


    # The Analyst will call this function, which pulls it all together.
    #   You shouldn't ever have to override this function, though it becomes an
    #   option if you want to do a very different form of analysis,
    #   (like how cluster analysis is different from analogy analysis).
    #   Even then, however, you should still be able to fit it all in the
    #   compute_stats function.
    def calculate(self, recalculate_all=False, **kwargs):
        # recalculate_all: whether or not to force it to recompute everything.
        #   Should be rarely, if ever, needed.

        # Available kwargs (always given in case needed):
        #
        #   embeddings                : 2D matrix, the whole vector space.
        #   draw_progress             : bool, whether user wants printouts.
        #   strings                   : list, objects in space, indeces match.
        #   printer_fn                : callable, Analyst's printout function.
        #   metric_str                : None|string, metric name, for scipy.
        #   metric_fn                 : callable, the one Analyst was given.
        #   metric_args               : dict, arguments for metric function.
        #   encoder_fn                : callable, string to vector.
        #   decoder_fn                : callable, vector to string.
        #   string_ix_map             : dict, maps strings to indeces.
        #   as_index_fn               : callable, any of three to index.
        #   as_string_fn              : callable, any of three to string.
        #   as_vector_fn              : callable, any of three to vector.
        #   exists_fn                 : callable, check any of three exists.
        #   is_string_fn              : callable, True if is str or bytes.
        #   angle_fn                  : callable, angle metric.
        #
        #   metric_in_model_fn        : getter, fast metric for in-model.
        
        #   generic_nearest_fn        : getter, nearest, keeps same type.
        #   generic_neighbor_k_fn     : getter, kth neighbor, keeps same type.
        #   kth_neighbors_ix_fn       : getter, vec. of ix of kth, or all.
        #   kth_neighbors_dist_fn     : getter, vec. of dist of kth, or all.
        #   arbitrary_dist_fn         : callable, dist to each obj in space.
        #   arbitrary_neighbors_fn    : callable, all ix ordered by proximity.
        #   distances_from_ix_fn      : callable, dist from one to all; SLOW?
        #   neighbors_of_ix_fn        : callable, all neighbors of ix; SLOW.
        #   condensed_dist_matrix_fn  : getter, 1D array of dist, if computed.
        #   condensed_ix_fn           : callable, convert ix to condensed.
        #
        #   downstream_fn             : callable, nearest-neighbor chain.
        #   evaluator_list            : list, all evaluators the Analyst has.
        #   find_evaluator_fn         : getter, finds evaluator by category.
        #   get_clusters_fn           : getter, finds cluster list by category.
        #   make_kth_neighbors        : list, k's for neighbors to compute.
        #   simulate_cluster_fn       : callable, Analyst's cluster simulator.
        #   parallel_count            : int >= 1, num threads user wants used.
        #   make_dist_matrix          : bool, whether OK to compute full matrix.

        # NOTE: Generic functions (non-type-specific) will be slower than
        #   direct mapping or indexing, so avoid them en masse.
        # NOTE: for details on the above, see their definitions in the Analyst.
        # NOTE: Analyst's spatial simulator has not been included because
        #   the Analyst is built to compare one analyst to another, instead of
        #   having an Evaluator compare one space to another. Hence even the
        #   cluster simulator is included only with feelings of reservation.

        if not self.calculated or recalculate_all:
            if kwargs == {}:
                print("NOT YET CALCULATED AND NO KWARGS GIVEN!")
            printer = kwargs["printer_fn"]
            show_progress = kwargs["draw_progress"]

            if show_progress:
                printer("Thinking Thoughts", "Evaluating " + self.CATEGORY)

            self.compute_stats(**kwargs)

            # Override default stars if user gave any:
            if self.given_stars != None:
                self.starred = self.given_stars

            self.calculated = True

        # Returning these means the Analyst need only access datamembers
        #   directly if we are doing specifics inspection, later,
        #   or when searching by category.
        return self.data_dict, self.starred, self.CATEGORY


    @staticmethod
    def _compute_list_stats(vals, attribute, dictionary=None):
        # Used for a single stat across multiple clusters or test sets.
        # vals: array-like containing the given attribute for each cluster.
        # attribute: name string, ie. "Dispersion".
        # dictionary: optional dictionary to add to instead of making a new one.

        stats = {} if dictionary == None else dictionary
        
        if len(vals) > 0:
            val_max = np.max(vals)
            val_min = np.min(vals)
            stats[attribute + " Avg"          ] = np.mean(vals)
            stats[attribute + " Min"          ] = val_min
            stats[attribute + " Max"          ] = val_max
            stats[attribute + " Range"        ] = val_max-val_min
            stats[attribute + " Standard Dev" ] = np.std(vals)
            stats[attribute + " Histogram Key"] = vals
        else:
            stats[attribute + " Stats"] = None

        return stats


    # These allow the retrieval of information without having to worry
    #   about whether or not it has been filled in.
    #   No getter needed for CATEGORY since it should never change.
    def get_data_dict(self, **kwargs):
        self.calculate(recalculate_all=False, **kwargs)
        return self.data_dict

    def get_starred(self, **kwargs):
        self.calculate(recalculate_all=False, **kwargs)
        return self.starred
