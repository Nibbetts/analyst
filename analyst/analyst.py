import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
#import cPickle as pickle
import dill as pickle
import os
#import sys
#import multiprocessing

from .evaluators import *
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

    def __init__(self, embeddings, metric="cosine",
        encoder=None, decoder=None, #cluster_algorithms=[(None, "All")],
        #analogy_algorithms=[], analogy_sets=[],
        evaluators=[],
        auto_print=True, desc=None, calculate=True, **metric_args):
        """
        Parameters:
            embeddings -- list of vectors populating the space.
                Must have static indeces. (ie: not a dict or set)
            metric -- the distance metric used throughout,
                accepts any string accepted by scipy.spatial.distance.pdist,
                or any callable accepting vectors as the first two parameters
                and returning a scalar. This allows for custom distance metrics.
            encoder -- a callable to convert strings to vectors, or
                alternatively a list of strings to replace encoder and decoder.
            decoder -- a callable to convert vectors to strings.
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
        print("")
        self._print("Asking the Grand Question")
        self._print("Stretching the Fabric of Space and Time")
        self._print("Enumerating the Dimensions")
        self.space = embeddings
        
        # Find and store a callable version of the given metric:
        self._print("Laying the Laws of Physics (Setting the Metric)")
        if callable(metric):
            self.metric = metric
            self.metric_str = None
        else:
            try:
                self.metric_str = sp.distance._METRIC_ALIAS[metric]
                self.metric = sp.distance._TEST_METRICS[
                    "test_" + self.metric_str]
            except Exception as e:
                print(e)
                raise ValueError("'metric' parameter unrecognized and uncallable")
        self.metric_args = metric_args
        
        self.description = desc

        # Encoder/Decoder Initializations:
        #   While initializing these should, in theory, be unnecessary,
        #   failing to do so will limit all inputs to findable types.
        #   If an encoder does not exist, may be replaced with list of strings.
        self._print("Filling the Void (Setting Encoder/Decoder)")
        self.ix_to_s = None
        if callable(encoder) or encoder is None:
            self.encode = encoder # string to vector
            self.decode = decoder # vector to string
        else:
            assert len(encoder) == len(self.space)
            self.ix_to_s = encoder
            self.vec_to_s = {}
            self.s_to_vec = {}
            self._print("Mapping the Emptiness \
                (Making Encoder/Decoder Mappings)")
            for i in range(len(self.space)):
                self.vec_to_s[self.space[i]] = self.ix_to_s[i]
                self.s_to_vec[self.ix_to_s[i]] = self.space[i]
            self.encode = self.s_to_vec.__getitem__
            self.decode = self.vec_to_s.__getitem__

        #self.serialized = False

        # ix_to_s initialized previously
        self.s_to_ix = {}
        try:
            if self.ix_to_s == None:
                self.ix_to_s = [self.decode(vec) for vec in tqdm(self.space,
                    desc="Naming Stars and Drawing a Star Map",
                    disable=(not self.auto_print))]
            for ix, s in enumerate(
                    tqdm(self.ix_to_s, desc="Indexing Planets",
                    disable=(not self.auto_print))):
                self.s_to_ix[s] = ix
        except Exception as e:
            print("DECODER APPEARS TO HAVE MALFUNCTIONED...")
            print(e)
            return

        # These are explained in the neighbors_getter function:
        self.neighbors = None
        self.neighbors_dist = None

        # Run Analyses:
        self.graph_info = []
        #self.categories = []
        #self.analogy_algorithms = analogy_algorithms
        #self.analogy_sets = analogy_sets
        self.evaluators = evaluators
        self.categories = [e.CATEGORY for e in self.evaluators]
        # self.built_in_categories = [
        #     "Spatial",
        #     "Extremities",
        #     "Nodes",
        #     "Hubs",
        #     "Supernodes",
        #     "Nuclei",
        #     "Chains",
        #     "NCC",
        #     "LNCC",
        #     "Anti-hubs"]
        # self.cluster_algorithms = []
        # report_spatial = False
        # for t in cluster_algorithms:
        #     if callable(t[0]):
        #         self.cluster_algorithms.append(t)
        #     else:
        #         tag = t[1].lower()
        #         if tag == "spatial":
        #             report_spatial = True
        #             self.categories.append("Spatial")
        #         elif tag == "all":
        #             report_spatial = True
        #             map(self.categories.append, self.built_in_categories)
        #         else:
        #             self.categories.append(t[1])
        #self.categories = [e.category for e in self.evaluators]
        self.category_lists = np.empty(
            shape=(len(self.categories),0)).tolist()

        self.distance_matrix = None

        self.s_to_node = {}
        self.spatial_data = {}
        self.cluster_data = {}
        self.analogical_data = {}

        if calculate:
            self._spatial_analysis()
            self._cluster_analysis()
            self._analogical_analysis()
            if auto_print: self.print_report()


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
            if isinstance(obj, basestring): return self.s_to_ix[obj]
            try: return self.s_to_ix[self.decode(obj)]
            except: return int(obj)
        else:
            return self.s_to_ix[self.decode(self.nearest(obj, in_model=False))]
    #
    def as_vector(self, obj, in_model=True):
        if in_model:
            if isinstance(obj, basestring): return self.encode(obj)
            try: return self.space[obj]
            except: return obj
        else:
            return self.nearest(obj, in_model=False)
    #
    def as_string(self, obj, in_model=True):
        if in_model:
            if isinstance(obj, basestring): return obj
            try: return self.ix_to_s[obj]
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
            n = k + 1 if k > 0 else k
            i = self.neighbors_getter()[0][self.as_index(obj)][n]
            if isinstance(obj, basestring): return self.ix_to_s[i]
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
            if self.metric_str != "cosine":
                return self.space[np.argmin(self.arbitrary_vector_dist(obj))]
            else:
                return self.space[np.argmax(np.dot(
                    self.space, np.array([vector]).T.squeeze()))]


    # Distance and Neighbor Computation:
    def distance_matrix_getter(self):
        # Allows us to only have to compute the distance matrix once, if needed.
        # Makes use of scipy in the fastest way possible. Any metric string
        #   recognized by scipy will work, or any valid function.
        #   For speed, prefer to use string, if one exists for desired metric.
        if self.distance_matrix == None:
            # Distance Matrix Calculation
            self._print("Acquainting the Species (Calculating Distance Matrix)")
            self.distance_matrix = sp.distance.squareform(sp.distance.pdist(
                self.space,
                self.metric_str if self.metric_str != None else self.metric,
                **self.metric_args))
        return self.distance_matrix

    def neighbors_getter(self):
        # Allows us to only have to compute neighbors etc once, if needed.
        if self.neighbors == None:
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
            self._print("Misconstruing Relations (Finding Furthest Neighbors)")
            self.neighbors[:,2] = np.argmax(self.distance_matrix, axis=1)
            self.neighbors_dist[:,2] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,2]]
            self.distance_matrix[
                range(len(self.space)), range(len(self.space))] = np.inf
            # Finding Nearest Neighbors
            self._print("Forming Alliances (Finding Nearest Neighbors)")
            self.neighbors[:,0] = np.argmin(self.distance_matrix, axis=1)
            self.neighbors_dist[:,0] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,0]]
            self.distance_matrix[
                range(len(self.space)), self.neighbors[:,0]] = np.inf
            # Finding Second-Nearest Neighbors
            self._print("Obfuscating Dynastic Ties \
                (Finding 2nd-Nearest Neighbors)")
            self.neighbors[:,1] = np.argmin(self.distance_matrix, axis=1)
            self.neighbors_dist[:,1] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,1]]
            # Put back the numbers we removed:
            self._print("Resetting the Ship's Computer \
                (Repairing Distance Matrix)")
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
        if self.metric_str != "cosine":
            return np.argsort(self.arbitrary_vector_dist(vector))
        else:
            distances = np.dot(self.space, np.array([vector]).T.squeeze())
            return distances.argsort()[::-1]

    # These return the kth neighbor of all objects in the space, index to index.
    #   Use negative for furthest, 0 for self, positive for nearest.
    #   Default None will return the whole matrix.
    def kth_neighbors(self, k=None):
        if k == None: return self.neighbors_getter()[0]
        assert -1 <= k <= 2
        if k == 0: return range(len(self.space))
        n = k + 1 if k > 0 else k
        return self.neighbors_getter()[0][:,n]

    def kth_neighbors_dist(self, k=None):
        if k == None: return self.neighbors_getter()[1]
        assert -1 <= k <= 2
        if k == 0: return np.zeros((len(self.space)), dtype=np.uint64)
        n = k + 1 if k > 0 else k
        return self.neighbors_getter()[1][:,n]

    # NOTE: This function does NOT force the evaluator to pre-calculate!
    # NOTE: Since categories SHOULD be unique among evaluators,
    #   this function will only return the first match it finds. Or None.
    def find_evaluator(self, category, force_creation=True):
        # force_creation: whether or not to create a default evaluator for
        #   built-ins.
        for e in self.evaluators:
            if e.category == category: return e
        if force_creation: 
            return Analyst.make_default_evaluator(category)

    @staticmethod
    def make_default_evaluator(category):
        pass



    #--------------------------------------------------------------------------#
    # General Analyses:                                                        #
    #--------------------------------------------------------------------------#

    def _spatial_analysis(self):

        # MEASUREMENTS:
        #self.distance_matrix_getter()
        self.neighbors_getter()

        # Centroid, Dispersion, Std Dev, repulsion:
        self._print("Balancing the Continuum")
        self.centroid = np.mean(self.space, axis=0)
        self.centroid_length = np.linalg.norm(self.centroid)
        self.centroid_dist = [self.metric(self.centroid, v, **self.metric_args)
            for v in tqdm(self.space, desc="Counting the Lightyears",
                disable=(not self.auto_print))]
        self.dispersion = np.mean(self.centroid_dist, axis=0)
        self.std_dev = np.std(self.space)
        centr_min = np.min(self.centroid_dist, axis=0)
        centr_max = np.max(self.centroid_dist, axis=0)
        #if print_report:
        self._add_info(self.ix_to_s[np.argmin([
            self.metric(self.centroid, v, **self.metric_args) \
                for v in tqdm(self.space,
                desc="Electing a Ruler", disable=(not self.auto_print))])],
            "Spatial", "Medoid - Obj Nearest to Centroid", star=True)
        self._add_info(len(self.space), "Spatial", "Count")
        self._add_info(self.centroid_length, "Spatial", "Centroid Length")
        self._add_info(self.dispersion,
            "Spatial", "Dispersion - Centroid Dist Avg", star=True)
        self._add_info(centr_min, "Spatial", "Centroid Dist Min")
        self._add_info(centr_max, "Spatial", "Centroid Dist Max")
        self._add_info(centr_max - centr_min,
            "Spatial", "Centroid Dist Range")
        self._add_info(self.centroid_dist,
            "Spatial", "Centroid Dist Histogram Key")
        self._add_info(self.std_dev, "Spatial", "Standard Dev")
        #self.repulsion = np.mean(
        #    [self.metric(v, self.encoder(self.nearest(self.objects[i])),
        #         **self.metric_args)
        #     for i, v in self.vectors])

        # Nearest Neighbor Info:
        self._print("Building Trade Routes")
        self.nearest_avg = np.mean(self.neighbors_dist[:,0])
        self._print("Practicing Diplomacy")
        nearest_min = np.min(self.neighbors_dist[:,0])
        nearest_max = np.max(self.neighbors_dist[:,0])
        #if print_report:
        self._add_info(self.nearest_avg,
            "Spatial", "repulsion - Nearest Dist Avg", star=True)
        self._add_info(nearest_min, "Spatial", "Nearest Dist Min")
        self._add_info(nearest_max, "Spatial", "Nearest Dist Max")
        self._add_info(nearest_max-nearest_min,
            "Spatial", "Nearest Dist Range", star=True)
        self._add_info(self.neighbors_dist[:,0],
            "Spatial", "Nearest Dist Histogram Key")

        # Second-Nearest Neighbor Info:
        self._print("Setting Priorities")
        self.nearest2_avg = np.mean(self.neighbors_dist[:,1])
        self._print("Coming up with Excuses")
        nearest2_min = np.min(self.neighbors_dist[:,1])
        nearest2_max = np.max(self.neighbors_dist[:,1])
        #if print_report:
        self._add_info(self.nearest2_avg,
            "Spatial", "Second Nearest Dist Avg")
        self._add_info(nearest2_min, "Spatial", "Second Nearest Dist Min")
        self._add_info(nearest2_max, "Spatial", "Second Nearest Dist Max")
        self._add_info(nearest2_max-nearest2_min,
            "Spatial", "Second Nearest Dist Range")
        self._add_info(self.neighbors_dist[:,1],
            "Spatial", "Second Nearest Dist Histogram Key")

        #Furthest Neighbor Info:
        self._print("Making Enemies")
        self.furthest_avg = np.mean(self.neighbors_dist[:,2])
        self._print("Claiming Frontiers")
        furthest_min = np.min(self.neighbors_dist[:,2])
        furthest_max = np.max(self.neighbors_dist[:,2])
        #far = np.argmax(self.neighbors_dist[:,2])
        #if print_report:
        self._add_info(self.furthest_avg, "Spatial", "Furthest Dist Avg")
        self._add_info(furthest_min, "Spatial", "Furthest Dist Min")
        self._add_info(furthest_max,
            "Spatial", "Broadness - Furthest Dist Max", star=True)
        self._add_info(furthest_max - furthest_min,
            "Spatial", "Furthest Dist Range")
        self._add_info(self.neighbors_dist[:,2],
            "Spatial", "Furthest Dist Histogram Key")


    def _cluster_analysis(self):
        # # NOTE: The built-in algorithms are not included in the loop below
        # #   because some of them depend on each other, and must be in order.

        # Run the Evaluations:
        for evaluator in self.evaluators:
            data_dict, starred, category = evaluator.calculate(
                space=self.space,             show_progress=self.auto_print,
                strings=self.ix_to_s,         metric_str=self.metric_str,
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
                evaluator_list=self.evaluators,
                find_evaluator_by_category_fn=self.find_evaluator,
                simulate_cluster_fn=Analyst.simulate_cluster)

            for key, value in data_dict:
                self._add_info(value, category, key, key in starred)

        # # Cluster Algorithms:
        # for alg in self.cluster_algorithms:
        #     # function = alg[0]
        #     # description = alg[1]
        #     self._print("Analyzing " + alg[1])

        #     # Custom Callables:
        #     if callable(alg[0]):
        #         if alg[1] == "" or alg[1] == None:
        #             alg[1] = alg[0].__name__
        #         cluster_list = []
        #         clusterings = alg[0](self.space)
        #         for i, c in enumerate(clusterings):
        #             strings = map(self.decode, c)
        #             nodes = self.find_nodes_from_string_list(strings)
        #             cluster_list.append(clusters.Cluster(
        #                 self.encode, self.metric, self.nearest, strings,
        #                 vectors=c, nodes=nodes, auto=True, ID=i))
                
        #         self.cluster_data[alg[1]] = cluster_list
        #         self._add_cluster_type_attributes(cluster_list, alg[1])
            
        #     # Invalid Non-callables:
        #     elif (alg[0] not in self.built_in_categories
        #             and alg[1] not in self.built_in_categories):
        #         self._print(
        #             alg[0] + ", " + alg[1] +" UNRECOGNIZED and NOT CALLABLE!")

    # ANALOGICAL & SPECIFICS:

    def _analogical_analysis(self):
        pass

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

    def compare_difference(self, analyst2, simple_diff=False):
        # Prints a full report with three numbers for each property
        #   instead of one - val_for_A, val_for_B, A_B_compared
        #   where A_B_compared is A-B if simple_diff, else is A-B/avg(absA,absB)
        # Also adds double-histogram information from the comparison in self,
        #   but not in analyst2.
        self._print("Bridging Two Universes")
        print("")
        if self.description == None: self.description = "ANALYST 1"
        if analyst2.description == None: desc2 = "ANALYST 2"
        else: desc2 = analyst2.description
        print(self.description.upper() + " vs. " + desc2.upper())

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
            print(category + ": ")
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
                # Check for empty (None) info:
                info1 = (info_list_1[info_indeces[info][0]] if
                    info_indeces[info][0] != None else None)
                info2 = (info_list_2[info_indeces[info][1]] if
                    info_indeces[info][1] != None else None)
                if info1 == info2 == None:
                    comb = ["???", None, None, "", " "]
                elif info1 == None:
                    comb = [info2[0], None, info2[1], "",
                        "*" if info2[2] else " "]
                elif info2 == None:
                    comb = [info1[0], info1[1], None, "",
                        "*" if info1[2] else " "]
                # Gather and format the rest:
                else:
                    comb = [
                        info1[0], # Description
                        info1[1], # var1
                        info2[1], # var2
                        "",       # comparison vars 1 & 2, or combined hist key
                        "*" if (info1[2] or info2[2]) else " "] # Star
                    # Deal with histogram keys, putting () around them:
                    if "Histogram Key" in comb[0]:
                        # Add a new key for a combined histogram:
                        self.graph_info.append(("2", info1[1], info2[1]))
                        comb[1] = "(" + str(comb[1]) + ")"
                        comb[2] = "(" + str(comb[2]) + ")"
                        comb[3] = "(" + str(len(self.graph_info) - 1) + ")"
                # Then if not strings, more formatting:
                if not (isinstance(comb[1], basestring) or
                        isinstance(comb[2], basestring)):
                    if comb[1] != None and comb[2] != None and comb:
                        # Commented versions is just a-b;
                        #   New version is scaled difference, (a-b)/avg(a,b):
                        #if comb[1] % 1.0 != 0 and comb[2] % 1.0 != 0:
                            #comb[3] = "{:<11.8f}".format(comb[1] - comb[2])
                        #elif "Histogram Key" not in comb[0]:
                            #comb[3] = comb[1] - comb[2]
                        # For int or float pairs, not keys or strings or Nones:
                        if "Histogram Key" not in comb[0]:
                            if simple_diff:
                                comb[3] = "{:<11.8f}".format(comb[1] - comb[2])
                            else:
                                average = (abs(comb[1]) + abs(comb[2]))/2.0
                                if average != 0: comb[3] = "{:<11.8f}".format(
                                    (comb[1] - comb[2])/average)
                                else: comb[3] = "nan"
                    # Format floats differently:
                    if comb[1] != None and comb[1] % 1.0 != 0:
                        comb[1] = "{:<11.8f}".format(comb[1])
                    if comb[2] != None and comb[2] % 1.0 != 0:
                        comb[2] = "{:<11.8f}".format(comb[2])
                #else: # Add quotes around printed words:
                #    if isinstance(comb[1], basestring):
                #        comb[1] = "\"" + comb[1] + "\""
                #    if isinstance(comb[2], basestring):
                #        comb[2] = "\"" + comb[2] + "\""

                # And finally print a line:
                print("  {} {:<11}  {:<11}  {:<11} {}{}".format(
                    comb[4], comb[1], comb[2], comb[3], comb[4], comb[0]))
                '''
                elif isinstance(info1[1], basestring) or info1[1] % 1.0 == 0:
                    print("  {} {:<11}  {:<11}  {:<11} {}{}".format(
                        comb[4],
                        info1[1],
                        info2[1],
                        ("" if (isinstance(info1[1], basestring)
                                or info1[1] == None or info2[1] == None) else (
                            info1[1]-info2[1])),???????????????
                        comb[4],
                        info1[0]))
                else:
                    print("  {} {:<11.8f}  {:<11.8f}  {:<11.8f} {}{}".format(
                        comb[4],
                        info1[1],
                        info2[1],
                        (info1[1]-info2[1] if (
                                info1[1] != None and info2[1] != None)
                            else ""),???????????????
                        comb[4],
                        info1[0]))'''

    @staticmethod
    def compare(ana_list):
        # Lists side by side the values for each analyst in the list.
        pass


    #--------------------------------------------------------------------------#
    # Information Gathering and Reporting Functions                            #
    #--------------------------------------------------------------------------#

    def _add_info(self, var, category, description, star=False):
        # Description and category must be strings.
        #variable = None
        #i = None
        if "Histogram Key" in description:
            variable = len(self.graph_info)
            self.graph_info.append(var)
        else: variable = var
        try:
            i = self.categories.index(category)
        except:
            i = len(self.categories)
            self.categories.append(category)
            self.category_lists.append([])
        self.category_lists[i].append(
            (description, variable, star))

    def _print(self, string=""):
        if self.auto_print: print("\r" + string + "...")

    def print_report(self):
        self._print("Revealing the Grand Plan")
        print("")
        if self.description != None: print(self.description.upper())
        for i, category in enumerate(self.categories):
            print(category + ": ")
            for cat in self.category_lists[i]:
                #print("\t" + str(cat[1]) + "\t" + str(cat[0]))
                #print(cat[0],cat[1],sep="\t") #python3
                if isinstance(cat[1], basestring) or cat[1] % 1.0 == 0:
                    # Keep strings strings and ints ints... :
                    print("  {} {:<11} {}{}".format(
                        "*" if cat[2] else " ", # Stars
                        ("(" + str(cat[1]) + ")" if "Histogram Key" in cat[0]
                            #else ("\"" + cat[1] + "\"" if # Quotes around words
                            #    isinstance(cat[1], basestring) else cat[1])),
                            else cat[1]),
                        "*" if cat[2] else " ", # Stars
                        cat[0]))
                else:
                    # ...But format floats to look uniform:
                    print("  {} {:<11.8f} {}{}".format(
                        "*" if cat[2] else " ", # Stars
                        cat[1],
                        "*" if cat[2] else " ", # Stars
                        cat[0]))
                #cat_new = [
                #    cat[0],
                #    ("{:<11.8f}".format(cat[1]) if (
                #        cat[1] % 1.0 != 0 and not isinstance(cat[1], basestring)
                #        ) else (
                #            "(" + str(cat[1]) + ")" if "Histogram Key" in cat[0]
                #            else cat[1])),
                #    "*" if cat[2] else " "]
                #print("  {} {:<11} {}{}".format(
                #    cat_new[2], cat_new[1], cat_new[2], cat_new[0]))


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
        return f_name if "." in f_name else f_name + ".dill"

    @staticmethod
    def save(obj, f_name):
        try:
            #obj._serialize()
            with open(Analyst._file_extension(f_name), 'wb') as file:
                pickle.dump(obj, file)#, pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            print("ERROR: Save function expected Analyst object.")
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
            print("ERROR: Unable to load or deserialize Analyst object from file: \"" + name + "\"")
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
