# Compatibility for python 2 and 3:
from __future__ import print_function
from __future__ import absolute_import
from builtins import str, bytes
from io import open

# Normal Packages:
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
#import cPickle as pickle
import dill as pickle
import os
#from sys import version_info
#import multiprocessing
import traceback

# Own files:
from .evaluators import *
#import evaluators
#from test_set_2d import TestSet2D


class Analyst:
    """
        Description:
            A toolset for embedding space analytics.


        Use:
                Generally you would initialize one Analyst instance per one
            embedding space, and perform analyses and access tools in the
            toolset through that analyst. However, with differing metrics you
            would use multiple analysts initialized with the same embeddings.

        Tools:
            (See README.md for details)

            General:
                NOTE: These are all static methods.
                Analyst.save(obj, path) -- Returns True if successful.
                Analyst.load(path) -- returns unpickled object, None if failed
                Analyst.unsave(path) -- deletes a saved file. Returns success

            Spatial:
                count
                centroid
                * medoid
                * dispersion
                * repulsion -- avg dist to nearest
                * broadness -- max dist to furthest
                various avg, min, max, range, graph of distribution of.

            Clustering:
                Extremities: (Mutual Furthest-Neighbor Pairs)
                Nodes: (Mutual Nearest-Neighbor Pairs)
                Hubs: (Nodal Proximity Groups)
                Supernodes: (Hierarchical Node Pairs)
                Nuclei: (Multi-Nodal Proximity Groups)
                Chains: (Nearest-Neighbor-Path Partition Groups)
                Clusters: (Nearest-Neighbor NODAL Conglomerate CLUSTERS)
                Strong Clusters: (Dispersion and Dual LIMITED Nearest-Neighbor
                    NODAL Conglomerate CLUSTERS)
                Anti-hubs: (Common Futhest-Neighbor Groups)

            Analogical:
                run_analogies()
                member_of(object) -- displays containing cluster for object
                cluster([list of objects]) -- a new cluster made from list
                seeded_cluster([list of objects]) -- a new seeded cluster
                inspect_clustering([list of objects]) -- pretend list is cluster
                circular_walk_graph(obj1, obj2)

            Comparative:
                compare_difference(analyst2, simple_diff=False)
                Analyst.compare([list_of_analysts])

            Specifics / Inspection:
                rank_outliers() -- by how often this one is furthest neighbor
                rank_clusters() -- by size; lists the indeces of the clusters
                rank_hubs() -- by how often this one is nearest neighbor
                graph(graph_key, bins) -- produce graph from key in report
                centroid -- accessible vector; can be used externally
                * clusters -- accessible variable; a list of the clusters
                * strong clusters -- ''
                nodes -- ''
                supernodes -- ''
                nuclei -- ''
                chains -- ''
                extremities -- ''
                anti_hubs -- dict keyed to outliers, contains anti-hubs
                as_string(obj) -- generic type converters for individual objects
                as_index(obj)
                as_vector(obj)

            Simulation:
                Analyst.simulate_space() -- generates a fake embedding space
                Analyst.simulate_cluster() -- generates generic test clusters
                TestSet2D -- small fake 2D embedding space class for testing
    """

    BUILT_IN_CATEGORIES = [
        u"Spatial",
        u"Nodes",
        u"Extremities",
        u"Supernodes",
        u"Hubs",
        u"Nuclei",
        u"Chains",
        u"NCC",
        u"LNCC",
        u"Anti-hubs"]

    def __init__(self, embeddings=None, strings=None,
        encoder=None, decoder=None, metric=u"cosine", evaluators=[u"All"],
        auto_print=True, desc=None, calculate=True, **metric_args):
        """
        Parameters:
            embeddings -- list of vectors populating the space.
                Must have static indeces. (ie: not dict or set) (semi-optional)
            strings -- a list of strings. (semi-optional)
            encoder -- a callable to convert strings to vectors. (semi-optional)
            decoder -- a callable to convert vectors to strings. (semi-optional)
            # NOTE: Put in as many of the above four as you can to reduce
            #   initial computation time. It WILL build the rest for rapid
            #   access, if it can, or will crash and tell you it can't.
            metric -- the distance metric used throughout,
                accepts any string accepted by scipy.spatial.distance.pdist,
                or any callable accepting vectors as the first two parameters
                and returning a scalar. This allows for custom distance metrics.
            evaluators -- FILL IN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # cluster_algorithms -- list of tuples (callable, "Description").
            #     Each callable must take an array-like list of vectors and return
            #     a list of array-like lists of vectors, each representing a
            #     cluster. They do not have to partition or even cover the space;
            #     ie, the can contain duplicates and do not have to contain all.
            #     If the left entry is None or is not callable, will expect a
            #     recognized tag in the right instead of a label, indicating to
            #     use a built-in function.
            #     Recognized tags:
            #         "Spatial" -- basic analysis of the space - must occur even
            #             if not included, but not including it will cause it not
            #             to display results.
            #         "Extremities" -- also generally include.
            #         "Nodes" -- also generally include. Many built-in clustering
            #             algorithms will cause these to be computed anyway.
            #         "Hubs"
            #         "Supernodes"
            #         "Nuclei"
            #         "Chains"
            #         "NNNCC" -- basic built-in experimental clustering algorithm
            #         "LNNNCC" -- advanced, or strong version of the above
            #         "Anti-clusters"
            #         "All" -- will automatically include all of the above.
            #     NOTE: As this variable contains functions, it will be altered
            #         to no longer contain functions when an Analyst is pickled.
            # analogy_algorithms -- list of tuples (callable, "Description").
            #     Each callable must take word_A, is_to_B, as_C, and must return
            #     is_to_D.
            #     NOTE: As this variable contains functions, it will be altered
            #         to no longer contain functions when an Analyst is pickled.
            # analogy_sets -- list of tuples: (list_of_quadruples, "Description),
            #     where each quadruple is a list of four strings, D being the
            #     correct answer to compare against. Example:
            #     [
            #         (
            #             [
            #                 ["wordA", "wordB", "wordC", "wordD"],
            #                 ["wordA", "wordB", "wordC", "wordD"]
            #             ],
            #             "Analogy Test Set 1"
            #         ),
            #         (
            #             [
            #                 ["wordA", "wordB", "wordC", "wordD"],
            #                 ["wordA", "wordB", "wordC", "wordD"]
            #             ],
            #             "Analogy Test Set 2"
            #         )
            #     ]
            auto_print -- whether to print reports automatically after analyses.
            desc -- optional short description/title for this analyst instance.
            calculate -- whether or not to run the analysis.
                Typically always True.
            metric_args -- these are extra arguments to be given to metric.
        """

        self.auto_print = auto_print
        print(u"")
        self._print(u"Asking the Grand Question", u"What is the Purpose of this Space?")
        self.description = str(desc)
        
        # Find and store a callable version of the given metric:
        self._print(u"Laying the Laws of Physics", u"Setting the Metric")
        if callable(metric):
            self.metric = metric
            self.metric_str = None
        else:
            try:
                self.metric_str = str(sp.distance._METRIC_ALIAS[metric])
                self.metric = sp.distance._TEST_METRICS[
                    u"test_" + self.metric_str]
            except Exception as e:
                print(e)
                raise ValueError(u"FATAL ERROR: %s PARAMETER UNRECOGNIZED AND UNCALLABLE!"
                    % str(metric))
        self.metric_args = metric_args

        # Encoder/Decoder Initializations:
        # OK if: encoder and strings, decoder and vectors, strings and vectors:
        #   encoder + strings -> vectors; vectors + strings -> decoder.
        #   decoder + vectors -> strings; strings + vectors -> encoder.
        #   strings + vectors -> encoder & decoder.
        self._print("Enumerating the Dimensions", "Making Internal Converters")
        # Find embeddings:
        if embeddings is None:
            if encoder is None or strings is None:
                raise ValueError(u"FATAL ERROR: Without embeddings, you must give both an encoder and strings!")
            else:
                self._print(u"Stretching the Fabric of Space and Time", u"Finding Embeddings")
                self.space = map(encoder,
                    tqdm(strings, disable=(not self.auto_print)))
        else: self.space = embeddings
        # Find strings:
        if strings is None:
            if embeddings is None and decoder is None:
                raise ValueError(u"FATAL ERROR: Without strings, you must give both a decoder and embeddings!")
            else:
                self._print(u"Naming Stars and Drawing a Star Map", u"Collecting Strings")
                self.strings = map(decoder,
                    tqdm(embeddings, disable=(not self.auto_print)))
        else: self.strings = strings
        # Now we have both embeddings and strings.
        assert len(self.space) == len(self.strings)
        # Make encoder:
        if encoder is None:
            self._print(u"Filling the Void", u"Mapping New Encoder")
            self.s_to_vec = {}
            for i in trange(len(self.strings), disable=(not self.auto_print)):
                self.s_to_vec[self.strings[i]] = self.space[i]
            self.encode = self.s_to_vec.__getitem__
        else:
            assert callable(encoder)
            self.encode = encoder
        # Make decoder:
        if decoder is None:
            self._print(u"Mapping the Emptiness", u"Mapping New Decoder")
            self.vec_to_s = {}
            for i in trange(len(self.space), disable=(not self.auto_print)):
                self.vec_to_s[self.space[i]] = self.strings[i]
            self.decode = self.vec_to_s.__getitem__
        else:
            assert callable(decoder)
            self.decode = decoder

        # Separate Map for quick string indexing:
        self.s_to_ix = {}
        self._print(u"Indexing Planets", u"Making String-Index Mappings")
        for ix, s in enumerate(
                tqdm(self.strings, disable=(not self.auto_print))):
            self.s_to_ix[s] = ix
        # NOTE: I'm not making vec_to_ix because I assume that the length of the
        #   vectors makes it more efficient to use s_to_ix[decode(vec)].

        # These are explained in the neighbors and distance matrix getters:
        self.distance_matrix = None
        self.neighbors = None
        self.neighbors_dist = None

        # Data
        self.graph_info = []
        self.categories = []
        self.evaluators = []
        self.category_lists = []
        self.add_evaluators(*evaluators)

        # Run Analyses:
        if calculate:
            self.analysis()


    # Generic type converters & tools for inputs and outputs:
    # NOTE: These will be slower than type-specific functions.
    #
    def exists(self, obj):
        if obj in self.s_to_ix: return True
        try:
            self.decode(obj)
            return True
        except: pass
        if obj // len(self.space) == 0: return True
        return False
    #
    # These work on objects not in the model, but this requires in_model=False,
    #   and of course only returns the nearest object.
    def as_index(self, obj, in_model=True):
        if in_model:
            if isinstance(obj, str) or isinstance(obj, bytes):
                return self.s_to_ix[obj]
            try: return self.s_to_ix[self.decode(obj)]
            except: return int(obj)
        else:
            return self.s_to_ix[self.decode(self.nearest(obj, in_model=False))]
    #
    def as_vector(self, obj, in_model=True):
        if in_model:
            if isinstance(obj, str) or isinstance(obj, bytes):
                return self.encode(obj)
            try: return self.space[obj]
            except: return obj
        else:
            return self.nearest(obj, in_model=False)
    #
    def as_string(self, obj, in_model=True):
        if in_model:
            if isinstance(obj, str) or isinstance(obj, bytes):
                return obj
            try: return self.strings[obj]
            except: return self.decode(obj)
        else:
            return self.decode(self.nearest(obj, in_model=False))


    # Gets the kth neighbor of obj. Negatives for furthest, 0 for self, positive
    #   for nearest. If not in_model, 0 is nearest.
    # Attempts to return the same type given, ie: index, string, or vector.
    # Ensures neighbors will be calculated before, without being recalculated.
    def neighbor_k(self, obj, k, in_model=True):
        if in_model:
            assert -1 <= k <= 2
            if k == 0: return obj
            n = k - 1 if k > 0 else k
            i = self.neighbors_getter()[0][self.as_index(obj)][n]
            if isinstance(obj, str) or isinstance(obj, bytes):
                return self.strings[i]
            try:
                int(obj)
                return i
            except: return self.space[i]
        else:
            # Note that if not in_model, we require obj to be a vector.
            self.space[self.arbitrary_vector_neighbors(obj)[k]]
    #
    def nearest(self, obj, in_model=True):
        if in_model:
            return self.neighbor_k(obj, 1)
        else:
            # Note that if not in_model, we require obj to be a vector.
            if self.metric_str != u"cosine":
                return self.space[np.argmin(self.arbitrary_vector_dist(obj))]
            else: # Optimization for cosine similarity:
                return self.space[np.argmax(np.dot(
                    self.space, np.array([obj]).T.squeeze()))]


    # Distance and Neighbor Computation:
    def distance_matrix_getter(self):
        # Allows us to only have to compute the distance matrix once, if needed.
        # Makes use of scipy in the fastest way possible. Any metric string
        #   recognized by scipy will work, or any valid function.
        #   For speed, prefer to use string, if one exists for desired metric.
        if self.distance_matrix == None:
            # Distance Matrix Calculation
            self._print(u"Acquainting the Species",
                u"Computing Distance Matrix")
            self.distance_matrix = sp.distance.squareform(sp.distance.pdist(
                self.space,
                self.metric_str if self.metric_str != None else self.metric,
                **self.metric_args))
        return self.distance_matrix

    def neighbors_getter(self):
        # Allows us to only have to compute neighbors etc once, if needed.
        if self.neighbors is None: # Must use 'is' here, else elementwise!
            # Need distance matrix filled in:
            self.distance_matrix_getter()

            # Initialize Empty Arrays:
            self.neighbors = np.empty((len(self.space),3), dtype=np.uint64)
                # Indeces correspond to indeces of vectors in the space.
                #   For each:
                #   [index of nearest, index of 2nd-nearest, index of furthest]
            self.neighbors_dist = np.empty(
                (len(self.space),3))
                # Same format as above, except distances to those indexed above.

            # Finding Furthest Neighbors
            self._print(u"Misconstruing Relations",
                u"Finding Furthest Neighbors")
            self.neighbors[:,2] = np.argmax(self.distance_matrix, axis=1)
            self.neighbors_dist[:,2] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,2]]
            self.distance_matrix[
                range(len(self.space)), range(len(self.space))] = np.inf
            # Finding Nearest Neighbors
            self._print(u"Forming Alliances", u"Finding Nearest Neighbors")
            self.neighbors[:,0] = np.argmin(self.distance_matrix, axis=1)
            self.neighbors_dist[:,0] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,0]]
            self.distance_matrix[
                range(len(self.space)), self.neighbors[:,0]] = np.inf
            # Finding Second-Nearest Neighbors
            self._print(u"Obfuscating Dynastic Ties",
                u"Finding 2nd-Nearest Neighbors")
            self.neighbors[:,1] = np.argmin(self.distance_matrix, axis=1)
            self.neighbors_dist[:,1] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,1]]
            # Put back the numbers we removed:
            self._print(u"Resetting the Ship's Computer",
                u"Repairing Distance Matrix")
            self.distance_matrix[
                range(len(self.space)), self.neighbors[:,0]
            ] = self.neighbors_dist[:,0]
            self.distance_matrix[
                range(len(self.space)), range(len(self.space))] = 0.0

        return self.neighbors, self.neighbors_dist

    def arbitrary_vector_dist(self, vector):
        # Takes a vector not in the model and finds its distance to every obj
        #   in the model, taking advantage of scipy's optimizations.
        # NOTE: results are not stored, so recomputes every time.
        return sp.distance.cdist(
                np.atleast_2d(vector), self.space,
                self.metric_str if self.metric_str != None else self.metric,
                **self.metric_args
            ).squeeze()
            
    def arbitrary_vector_neighbors(self, vector):
        # Takes a vector not in the model and finds its distance to every obj
        #   in the model, returning a 1D array of indeces (not vectors!)
        # Includes an extra optimization for the common case
        #   that metric is cosine similarity.
        # NOTE: results are not stored, so recomputes every time.
        if self.metric_str != u"cosine":
            return np.argsort(self.arbitrary_vector_dist(vector))
        else:
            distances = np.dot(self.space, np.array([vector]).T.squeeze())
            return distances.argsort()[::-1]

    # These return the kth neighbor of all objects in the space, index-to-
    #   index | distance, respectively.
    #   Use negative for furthest, 0 for self, positive for nearest.
    #   Default None will return the whole matrix.
    def kth_neighbors(self, k=None):
        if k == None: return self.neighbors_getter()[0]
        assert -1 <= k <= 2
        if k == 0: return range(len(self.space))
        n = k - 1 if k > 0 else k
        return self.neighbors_getter()[0][:,n]

    def kth_neighbors_dist(self, k=None):
        if k == None: return self.neighbors_getter()[1]
        assert -1 <= k <= 2
        if k == 0: return np.zeros((len(self.space)), dtype=np.uint64)
        n = k - 1 if k > 0 else k
        return self.neighbors_getter()[1][:,n]

    # Computes the downstream nearest neighbor, and lists the path if asked,
    #   starting from obj's kth-nearest neighbor,
    #   then going from one NEAREST neighbor to another until we start to repeat
    #   (having reached a node). Thus the last two in the list make a node. If
    #   you start from k=0, obj will be included. k=1 is same, but without obj.
    # Note: non type-specific, and returns same type as given.
    # Note: should be faster if path not kept.
    # Note: probably less efficient if you use straight vectors, because of
    #   equality checks. Maybe most efficient for indeces?
    def downstream(self, obj, start_neighbor_k=0, give_path=False):
        path = None
        if give_path: path=[]
        current = self.neighbor_k(obj, start_neighbor_k, in_model=True)
        while current not in path: # More efficient than overhead for a set?
            if give_path: path.append(current)
            current = self.nearest(current)
        if give_path: return path
        else: return current

    # Superfast metric function for objects within the model only, using dicts.
    # Note: generic types.
    def metric_in_model(self, obj1, obj2):
        return self.distance_matrix_getter()[
            self.as_index(obj1, in_model=True),
            self.as_index(obj2, in_model=True)]


    # NOTE: This function does NOT force the evaluator to pre-calculate!
    # NOTE: Since categories SHOULD be unique among evaluators,
    #   this function will only return the first match it finds. Or None.
    def find_evaluator(self, category, force_creation=False):
        # force_creation: whether or not to create a default evaluator for
        #   built-ins.
        for e in self.evaluators:
            if str(e.CATEGORY) == category: return e
        if force_creation: 
            return Analyst.make_default_evaluator(str(category))

    # Makes Built-in Clusterizers with Default Values:
    # Note: Can take some parameterization, such as "Nodal 10-Hubs", or "2Hubs".
    #   "Hubs" with no number defaults to "Nodal 4-Hubs".
    @staticmethod
    def make_default_evaluator(category):
        cat = category.lower()
        if cat == u"spatial":
            return Spatializer()
        elif cat == u"nodes":
            return NodeClusterizer()
        elif cat == u"extremities":
            return ExtremityClusterizer()
        elif cat == u"supernodes":
            return SupernodeClusterizer()
        elif u"hubs" in cat and u"anti" not in cat:
            thresh = 4
            nodal = False
            if u"nodal " in cat:
                nodal = True
            try:
                thresh = int(cat[6 if nodal else 0:-5]) if u"-hubs" in cat \
                    else int(cat[6 if nodal else 0:-4])
            except: pass
            if cat == u"hubs":
                nodal = True
                cat = u"Nodal 4-Hubs"
            else: cat = category
            return HubClusterizer(threshold=thresh, nodal=nodal, category=cat)
        # ADD STUFF!!
        else:
            print(u"UNRECOGNIZED BUILT-IN EVALUTATOR '"
                + category + u"'! SKIPPING IT.")
            return None

    def add_evaluators(self, *args):
        # Helper function
        def rename_evaluator(evaluator):
            version = 2
            category = evaluator.CATEGORY
            while category in self.categories:
                category = evaluator.CATEGORY + u" (" + str(version) + u")"
            evaluator.CATEGORY = category

        warning = u"WARNING: Evaluator category '{}' already exists! {} it."

        # Add evaluators and categories
        for e in args:
            if isinstance(e, str) or isinstance(e, bytes): # If keyword
                if str(e.lower()) == u"all": # If keyword 'All'
                    for cat in Analyst.BUILT_IN_CATEGORIES:
                        evaluator = Analyst.make_default_evaluator(cat)
                        if evaluator != None:
                            if evaluator.CATEGORY not in self.categories:
                                self.categories.append(evaluator.CATEGORY)
                                self.evaluators.append(evaluator)
                            else:
                                self._print(warning.format(
                                    evaluator.CATEGORY, u"SKIPPING"))
                else: # If keyword matching a built-in
                    evaluator = Analyst.make_default_evaluator(e)
                    if evaluator != None:
                        if evaluator.CATEGORY in self.categories:
                            self._print(warning.format(
                                evaluator.CATEGORY, u"RENAMING"))
                            rename_evaluator(evaluator)
                        self.categories.append(evaluator.CATEGORY)
                        self.evaluators.append(evaluator)
            else: # If actual evaluator object
                if e != None:
                    if e.CATEGORY in self.categories:
                        self._print(warning.format(e.CATEGORY, u"RENAMING"))
                        rename_evaluator(e)
                    self.categories.append(e.CATEGORY)
                    self.evaluators.append(e)

        # Modify category_lists to match categories
        diff = len(self.categories) - len(self.category_lists)
        self.category_lists += np.empty(shape=(diff,0)).tolist()


    #--------------------------------------------------------------------------#
    # General Analyses:                                                        #
    #--------------------------------------------------------------------------#

    def analysis(self, recalculate=[]):
        # Won't recalculate any but those whose categories are listed.
        # Even those it doesn't recalculate, it will still get their data and
        #   update its own in case it has changed.

        # Run the Evaluations:
        for evaluator in self.evaluators:
            try:
                data_dict, starred, category = evaluator.calculate(
                    recalculate_all=False, # Only does those not yet done.

                    # NOTE: The rest are the kwargs:
                    embeddings=self.space,        draw_progress=self.auto_print,
                    strings=self.strings,         metric_str=self.metric_str,
                    printer_fn=self._print,       metric_fn=self.metric,
                    as_string_fn=self.as_string,  metric_args=self.metric_args,
                    as_index_fn=self.as_index,    encoder_fn=self.encode,
                    as_vector_fn=self.as_vector,  decoder_fn=self.decode,
                    string_ix_map=self.s_to_ix,   exists_fn=self.exists,

                    generic_neighbor_k_fn=self.neighbor_k,
                    generic_nearest_fn=self.nearest,
                    kth_neighbors_ix_fn=self.kth_neighbors,
                    kth_neighbors_dist_fn=self.kth_neighbors_dist,
                    distance_matrix_getter_fn=self.distance_matrix_getter,
                    arbitrary_dist_fn=self.arbitrary_vector_dist,
                    arbitrary_neighbors_fn=self.arbitrary_vector_neighbors,
                    metric_in_model_fn=self.metric_in_model,
                    downstream_fn=self.downstream,
                    evaluator_list=self.evaluators,
                    find_evaluator_fn=self.find_evaluator,
                    simulate_cluster_fn=Analyst.simulate_cluster)

                # The below compatibilities should be unnecessary because both
                #   keys and starred come from same source, thus same version.
                #   Also, they're just going to be printed later, so no others
                #   need it either.
                #starred = map(str, starred)
                #key = str(key)
                #category = str(category)

                for (key, value) in data_dict.items():
                    self._add_info(value, category, key, key in starred)

            except Exception: # as e:
                #print(e)
                traceback.print_exc()
                print(u"ERROR IN CALCULATION OF %s. DOES YOUR EVALUATOR INHERIT FROM Evaluator CLASS?"
                    % evaluator.CATEGORY)
        
        if self.auto_print: self.print_report()


    # SPECIFICS INSPECTION:

    def graph(self, hist_key, bins=16):
        """
            Description: creates a histogram according to key printed in report.
            NOTE: Keys/Information from comparisons made are accessible only in
                the analyst which was leftward when the comparison was made.
        """

        #in self.graph_info, singles are directly the list of data, while
        #   comparison graphs are tuples:
        #       ("2", datalist_from_self, datalist_from_other)
        pass

    # COMPARATIVE:

    @staticmethod
    def _formatit(data, width=10, parentheses=False, start_at=0):
        #if parentheses: w = max(9, width)
        #else: w = max(7, width)
        w = max(9, width)
        result = ""
        if data is None:
            result = " " * w
        elif isinstance(data, str) or isinstance(data, bytes) or parentheses:
            # Strings or Bytestrings
            result = " " # For negatives on others
            if parentheses: result += "(" + str(data) + ")"
            else: result += data
            format_str = "{:" + str(w) + "}"
            result = format_str.format(result)
            if len(result) > w:
                result += "\n" + " " * (start_at + w)
        else:
            if (abs(data) >= 1e4 and not parentheses) \
                    or (abs(data) < 1e-4 and data != 0):
                # Large or Small
                format_str = "{: " + str(w) + "." + str(w - 7) + "e}"
                result = format_str.format(data)
            elif isinstance(data, int):
                # Integer
                format_str = "{:< " + str(w) + "d}"
                result = format_str.format(int(data))
            else:
                # Float
                format_str = "{: " + str(w - 1) + "." \
                    + str(w - 2 - len(str(int(abs(data))))) + "f}"
                result = format_str.format(data)
        return result
    
    # My own diff function, which emphasizes numbers further from zero:
    @staticmethod
    def weighted_difference(a, b):
        average = (abs(a) + abs(b))/2.0
        if average != 0: return (a - b)/average
        else: return np.nan

    # Compare this analyst with another, data per data.
    def compare_difference(self, analyst2, w=10, comparator=u"simple"):
        # Prints a full report with three numbers for each property
        #   instead of one - val_for_A, val_for_B, A_B_compared.
        # comparator:
        #   callable (not used on strings or None), or a built-in:
        #   "none" or None: don't print the third column
        #   "simple": A - B
        #   "weighted": (A - B) / avg(abs(A), abs(B))
        # w: Numbers will have space for w-2 digits, (w-2 because of . and - ).
        #   Total width will be: (3 + (w + 1)*num_cols + 2 + len(description))
        # Returns: a grapher object with double-histogram information
        #   from the comparison
        self._print(u"Bridging Two Universes",
            u"Building One-to-one Comparison")
        print(u"")
        
        # Descriptions to use:
        if self.description == None: desc = u"ANALYST 1"
        else: desc = self.description
        if analyst2.description == None: desc2 = u"ANALYST 2"
        else: desc2 = analyst2.description
        print(desc.upper() + u" vs. " + desc2.upper())

        # Comparator:
        if callable(comparator): comparison = comparator
        elif comparator == None or comparator.lower() == "none":
            comparator = None
            comparison = lambda a, b: ""
        elif comparator.lower() == u"simple": comparison = lambda a, b: a - b
        else: comparison = Analyst.weighted_difference

        # Combine and sort the Categories without losing any of them:
        all_categories = []
        category_indeces = {}
        # From left analyst (self):
        for i, category in enumerate(self.categories):
            if category not in all_categories:
                all_categories.append(category)
                category_indeces[category] = (i, None)
            else:
                t = category_indeces[category]
                if t[0] == None: category_indeces[category] = (i, t[1])
        # From right analyst (other):
        for i, category in enumerate(analyst2.categories):
            if category not in all_categories:
                all_categories.append(category)
                category_indeces[category] = (None, i)
            else:
                t = category_indeces[category]
                if t[1] == None: category_indeces[category] = (t[0], i)        

        # Combine and sort the info in each category without losing any:
        for category in all_categories:
            print(category + u": ")
            # Gather info from each:
            try:
                info_list_1 = self.category_lists[category_indeces[category][0]]
            except: info_list_1 = []
            try:
                info_list_2 = analyst2.category_lists[
                    category_indeces[category][1]]
            except: info_list_2 = []

            all_info = []
            info_indeces = {}
            # From left analyst (self):
            for i, info in enumerate(info_list_1):
                if info[0] not in all_info:
                    all_info.append(info[0])
                    info_indeces[info[0]] = (i, None)
                else:
                    t = info_indeces[info[0]]
                    if t[0] == None: info_indeces[info[0]] = (i, t[1])
            # From right analyst (other):
            for i, info in enumerate(info_list_2):
                if info[0] not in all_info:
                    all_info.append(info[0])
                    info_indeces[info[0]] = (None, i)
                else:
                    t = info_indeces[info[0]]
                    if t[1] == None: info_indeces[info[0]] = (t[0], i)    

            # Then for the combined info from that category:
            for info in all_info:
                # Combine each line of info, checking for None:
                # comb = [description, var1, var2, comparison, star]
                info1 = (info_list_1[info_indeces[info][0]] if
                    info_indeces[info][0] != None else None)
                info2 = (info_list_2[info_indeces[info][1]] if
                    info_indeces[info][1] != None else None)
                if info1 == info2 == None:
                    comb = [u"???", None, None, u"", u" "]
                elif info1 == None:
                    comb = [info2[0], None, info2[1], u"",
                        u"*" if info2[2] else u" "]
                elif info2 == None:
                    comb = [info1[0], info1[1], None, u"",
                        u"*" if info1[2] else u" "]
                else: comb = [info1[0], info1[1], info2[1], u"",
                    u"*" if (info1[2] or info2[2]) else u" "]
                
                is_hist = u"Histogram Key" in comb[0]

                # Compute comparison:
                if comb[1] != None and comb[2] != None:
                    if u"Histogram Key" in comb[0]:
                        # Add a new key for a combined histogram:
                        self.graph_info.append((u"c", info1[1], info2[1]))
                        comb[3] = len(self.graph_info) - 1
                    elif not (isinstance(comb[1], str) or \
                            isinstance(comb[1], bytes) or \
                            isinstance(comb[2], str) or \
                            isinstance(comb[2], bytes)):
                        comb[3] = comparison(comb[1], comb[2])
                
                # Formatting:
                comb[1] = Analyst._formatit(comb[1], w, is_hist, 3)
                comb[2] = Analyst._formatit(comb[2], w, is_hist, 3+(w+1))
                comb[3] = Analyst._formatit(comb[3], w, is_hist, 3+(w+1)*2)

                # And finally print a line:
                if comparator == None:
                    print(u"  {}{} {} {} {}".format(
                        comb[4], comb[1], comb[2], comb[4], comb[0]))
                else: print(u"  {}{} {} {} {} {}".format(
                    comb[4], comb[1], comb[2], comb[3], comb[4], comb[0]))

    @staticmethod
    def compare(ana_list, w=10, comparators=[u"all"]):
        # Lists side by side the values for each analyst in the list,
        #   as well as a column for each comparator, run on that stat.
        # w: Numbers will have space for w-2 digits, (w-2 because of . and - ).
        #   Total width will be: (6 + (w + 1)*(an + comp) + len(description))
        # comparator:
        #   empty list: no comparison columns
        #   callable (Takes list of available values; does NOT need to handle
        #       strings or None), or a built-in:
        #   "all": all builtins
        #   "std": standard deviation across the stat
        #   "avg" or "average": average across the stat
        #   "max" or "maximum": maximum value
        #   "min" or "minimum": minimum value
        #   "rng" or "range": max value minus min value
        # ana_list: a list of analysts. Kinda thought that was clear... :)
        # Returns: a grapher object with multi-histogram information from the!!!!!!!!!!!!!!!!!!!!!!!!!!
        #   the comparison.
        assert len(ana_list) > 0
        ana_list[0]._print(u"Bridging Two Universes",
            u"Building One-to-one Comparison")
        print(u"")

        # Descriptions to use:
        title = u"Comparison:"
        descriptions = []
        for i, a in enumerate(ana_list):
            if a.description == None:
                title += u" ANALYST " + str(i + 1)
                descriptions.append(u"ANALYST " + str(i + 1))
            else:
                title += " " + a.description.upper()
                descriptions.append(a.description.upper())
            if i < len(ana_list) - 1: title += ","
        print(title)

        # Comparator:
        comparisons = []
        def rng(l): return np.max(l) - np.min(l)
        for i, c in enumerate(comparators):
            if callable(c): comparisons.append(c)
            else: 
                word = c.lower()
                if word == u"all":
                    if np.std not in comparisons: comparisons.append(np.std)
                    if np.mean not in comparisons: comparisons.append(np.mean)
                    if np.max not in comparisons: comparisons.append(np.max)
                    if np.min not in comparisons: comparisons.append(np.min)
                    if rng not in comparisons: comparisons.append(rng)
                elif word == u"std" and np.std not in comparisons:
                    comparisons.append(np.std)
                elif (word == u"avg" or word == u"average") \
                    and np.mean not in comparisons: comparisons.append(np.mean)
                elif (word == u"max" or word == u"maximum") \
                    and np.max not in comparisons: comparisons.append(np.max)
                elif (word == u"min" or word == u"minimum") \
                    and np.min not in comparisons: comparisons.append(np.min)
                elif (word == u"rng" or word == u"range") \
                    and rng not in comparisons: comparisons.append(rng)

        # Column Headers:
        title_string = u"   " + u"{} " * len(ana_list) + u"|" + \
            "{} " * len(comparisons) + u"  PROPERTY"
        titles = descriptions + [c.__name__.upper() for c in comparisons]
        s = 3
        for i, t in enumerate(titles):
            titles[i] = Analyst._formatit(t, w, False, s)
            s += w + 1
            if i == len(ana_list): s += 1
        print(title_string.format(*titles))

        # Line Template:
        line_string = u"  {}" + u"{} " * len(ana_list) + u"|" + \
            u"{} " * len(comparisons) + u"|{}{}"
        #graphs = []

        # Helper function; All the work to print one category:
        def print_category(c):
            print(c + u":")
            # Get the information from this category in each Analyst:
            category_lists = []
            for a in ana_list:
                try:
                    index = a.categories.index(c)
                    category_lists.append(a.category_lists[index])
                except: category_lists.append(None)
            
            # Collect data:
            descriptions = []
            desc_lower = set()
            stars = {}
            values = {}
            for a, l in enumerate(category_lists): # a is index of analyst
                if l is not None:
                    for t in l:
                        desc = t[0].lower()
                        values[(a, desc)] = t[1]
                        if desc not in desc_lower:
                            desc_lower.add(desc)
                            descriptions.append(t[0])
                            stars[desc] = t[2]
                        else: stars[desc] = stars[desc] or t[2]

            # Print lines:
            for desc in descriptions:
                d = desc.lower()
                is_hist = u"histogram key" in d

                # Gather Data:
                datalist = [values[(a, d)] if (a, d) in values else u"" \
                    for a in range(len(ana_list))]
                # Comparisons:
                numbers = filter(lambda a: not isinstance(a, str) \
                    and not isinstance(a, bytes), datalist)
                if numbers == [] or is_hist:
                    comps = [u""] * len(comparisons)
                else: comps = [comp(numbers) for comp in comparisons]
                # Histograms:
                #if is_hist:
                #    if comps == []!!: comps = [len(graphs)]
                #    else: comps[0] = len(graphs)
                #    #keys = [datalist[i] if data]!!!
                #    graphs.append(tuple("c") + tuple([a.graph_info[...]]))
                # Formatting:
                data = []
                s = 3
                for i, v in enumerate(datalist + comps):
                    data.append(Analyst._formatit(v, w, is_hist, s))
                    s += w + 1
                    if i == len(ana_list): s += 1
                star = u"*" if stars[d] else u" "
                data = [star] + data + [star, desc]
                print(line_string.format(*data))

        # Put it all together - Loop through and print each category in order:
        categories = []
        for a in ana_list:
            for c in a.categories:
                if c not in categories:
                    categories.append(c)
                    print_category(c)



    #--------------------------------------------------------------------------#
    # Information Gathering and Reporting Functions                            #
    #--------------------------------------------------------------------------#

    def _add_info(self, var, category, description, star=False):
        # Description and category must be strings.
        #variable = None
        #i = None
        if u"Histogram Key" in description:
            variable = len(self.graph_info)
            self.graph_info.append(var)
        else: variable = var
        try:
            i = self.categories.index(category)
        except:
            i = len(self.categories)
            self.categories.append(category)
            self.category_lists.append([])
        found = False
        for entry in self.category_lists[i]: # Replace if it exists:
            if entry[0] == description:
                entry = (description, variable, star)
                found = True
                break # NOTE: will only replace the first. So don't duplicate.
        if not found: # Add it if it doesn't exist:
            self.category_lists[i].append((description, variable, star))

    def _print(self, string=u"", report=None):
        if self.auto_print: print("{:<40}".format(u"\r" + str(string) + u"...")
            + ("" if report is None else u"(" + report + u")"))

    def print_report(self, w=10):
        self._print(u"Revealing the Grand Plan", u"Printing Report")
        print(u"")
        if self.description != None: print(self.description.upper())
        for i, category in enumerate(self.categories):
            print(category + u": ")
            for cat in self.category_lists[i]:
                print(u"  {}{} {}{}".format(
                    "*" if cat[2] else u" ", # Stars
                    Analyst._formatit(cat[1], w, u"Histogram Key" in cat[0], 3),
                    u"*" if cat[2] else u" ", # Stars
                    cat[0]))


    #--------------------------------------------------------------------------#
    # Simulation:                                                              #
    #--------------------------------------------------------------------------#

    @staticmethod
    def simulate_space(parameters):
        '''
        parameters:
            A list of lists, each of which follows the format:
                ["space_type", "cluster_type", num_clusters, space_radius,
                    space_dims (cluster_min_pop, cluster_max_pop),
                    (cluster_min_radius, cluster_max_radius),
                    cluster_occupied_dims, cluster_total_dims, randomize_dims,
                    noise, normalize]

                Types: (used for both cluster and space)
                    "shell" (circle if occupied_dims==2)
                    "ball"
                    "radial" (like ball only random direction and radius instead
                        of x,y,z,... Ends up concentrated in center.)
                    "cube" (random x,y,z,... but in plane or hypercube instead
                        of ball)
                    "even" (attempts amorphous semi-uniformity of distances btw.
                        points)
                    "grid" (attempts a gridlike uniformity)
                    "pairs" (generates points in pairs of close repulsion --
                        forces excessive node generation)
                    "line" (generates points in lines)
                    "snake" (generate points in curvy lines)
                    "oval" (like radial, but randomly varies size of each axis
                        within allowed radius sizes)
                    "hierarchy" (attempts to recursively make closer and closer
                        pairs of groupings.)

            NOTE: Multiple lists in the parameters can be used to fill the space
                    with varied types of data.
                occupied dimensions must be <= total dimensions.
                min_num <= max_num.
                randomize_dims: boolean.
                    If false, will use same set of dims for each cluster.
                noise: a float; how much to randomly vary locations.
                normalize: boolean. If true will afterward scale all vectors
                    to unit length of 1, creating a hypersphere.

        returns:
            A new analyst object, and
            A list of the clusters used to create the space before clustering
                was recalculated, for comparison. This will be different if
                clusters overlapped.
        '''
        pass
        #note, need to make it create a generic identity function for
        #   encode/decode. or use indeces.

    @staticmethod
    def simulate_cluster(type, population, radius, occupied_dims,
        total_dims, randomize_dims=True, noise=0, normalize=False):
        # Same usage as the cluster parameters in simulate_space().
        # NOTE: when this function is called by simulate_space(), normalize
        #   is never True here. That would be done after the fact,
        #   on the whole simulated space, not on any one cluster.
        pass


    #--------------------------------------------------------------------------#
    # General Functions:                                                       #
    #--------------------------------------------------------------------------#

    @staticmethod
    def _file_extension(f_name):
        return str(f_name) if u"." in str(f_name) else f_name + u".dill"

    @staticmethod
    def save(obj, f_name):
        try:
            #obj._serialize()
            with open(Analyst._file_extension(f_name), 'wb') as file:
                pickle.dump(obj, file)#, pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            print(u"ERROR: Save function expected Analyst object.")
            print(e)
            return False

    @staticmethod
    def load(f_name):
        name = Analyst._file_extension(f_name)
        try:
            with open(name, 'rb') as file:
                an = pickle.load(file)
                #an._deserialize(metric, encoder, decoder, cluster_algorithms, analogy_algorithms)
                return an
        except Exception as e:
            print(u"ERROR: Unable to load or deserialize Analyst object from file: '{}'".format(name))
            print(e)
            #raise e
            return None

    @staticmethod
    def unsave(f_name):
        try:
            os.remove(Analyst._file_extension(f_name))
            return True
        except:
            return False


# END OF ANALYST CLASS.
################################################################################



# Brief script-like behavior for development, debugging, and testing purposes:
if __name__ == "__main__":
    #import TestSet2D

    raise Exception("Analyst script behabior not defined.")
