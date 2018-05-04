import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
#import cPickle as pickle
import dill as pickle
import os
#import sys
#import multiprocessing

from .clustertypes import *
from .clusterizers import *
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
        encoder=None, decoder=None, cluster_algorithms=[(None, "All")],
        analogy_algorithms=[], analogy_sets=[],
        auto_print=True, desc=None, calculate=True):
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
            cluster_algorithms -- list of tuples (callable, "Description").
                Each callable must take an array-like list of vectors and return
                a list of array-like lists of vectors, each representing a
                cluster. They do not have to partition or even cover the space;
                ie, the can contain duplicates and do not have to contain all.
                If the left entry is None or is not callable, will expect a
                recognized tag in the right instead of a label, indicating to
                use a built-in function.
                Recognized tags:
                    "Spatial" -- basic analysis of the space - must occur even
                        if not included, but not including it will cause it not
                        to display results.
                    "Extremities" -- also generally include.
                    "Nodes" -- also generally include. Many built-in clustering
                        algorithms will cause these to be computed anyway.
                    "Hubs"
                    "Supernodes"
                    "Nuclei"
                    "Chains"
                    "NNNCC" -- basic built-in experimental clustering algorithm
                    "LNNNCC" -- advanced, or strong version of the above
                    "Anti-clusters"
                    "All" -- will automatically include all of the above.
                NOTE: As this variable contains functions, it will be altered
                    to no longer contain functions when an Analyst is pickled.
            analogy_algorithms -- list of tuples (callable, "Description").
                Each callable must take word_A, is_to_B, as_C, and must return
                is_to_D.
                NOTE: As this variable contains functions, it will be altered
                    to no longer contain functions when an Analyst is pickled.
            analogy_sets -- list of tuples: (list_of_quadruples, "Description),
                where each quadruple is a list of four strings, D being the
                correct answer to compare against. Example:
                [
                    (
                        [
                            ["wordA", "wordB", "wordC", "wordD"],
                            ["wordA", "wordB", "wordC", "wordD"]
                        ],
                        "Analogy Test Set 1"
                    ),
                    (
                        [
                            ["wordA", "wordB", "wordC", "wordD"],
                            ["wordA", "wordB", "wordC", "wordD"]
                        ],
                        "Analogy Test Set 2"
                    )
                ]
            auto_print -- whether to print reports automatically after analyses.
            desc -- optional short description/title for this analyst instance.
            calculate -- whether or not to run the analysis.
                Typically always True.
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

        if calculate:
            # MAYBE PUT THIS OUTSIDE OF IF CALCULATE!
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
            self.categories = []
            self.built_in_categories = [
                "Spatial",
                "Extremities",
                "Nodes",
                "Hubs",
                "Supernodes",
                "Nuclei",
                "Chains",
                "NCC",
                "LNCC",
                "Anti-hubs"]
            self.cluster_algorithms = []
            report_spatial = False
            for t in cluster_algorithms:
                if callable(t[0]):
                    self.cluster_algorithms.append(t)
                else:
                    tag = t[1].lower()
                    if tag == "spatial":
                        report_spatial = True
                        self.categories.append("Spatial")
                    elif tag == "all":
                        report_spatial = True
                        map(self.categories.append, self.built_in_categories)
                    else:
                        self.categories.append(t[1])
            self.category_lists = np.empty(
                shape=(len(self.categories),0)).tolist()

            self.analogy_algorithms = analogy_algorithms
            self.analogy_sets = analogy_sets
            self.distance_matrix = None

            self.s_to_node = {}
            self.spatial_data = {}
            self.cluster_data = {}
            self.analogical_data = {}
            self._spatial_analysis(report_spatial)
            self._cluster_analysis()
            self._analogical_analysis()
            if auto_print: self.print_report()


    # Generic type converters for inputs and outputs:
    def as_index(self, obj):
        if isinstance(obj, basestring): return self.s_to_ix[obj]
        try: return self.s_to_ix[self.decode(obj)]
        except: return int(obj)

    def as_vector(self, obj):
        if isinstance(obj, basestring): return self.encode(obj)
        try: return self.space[obj]
        except: return obj

    def as_string(self, obj):
        if isinstance(obj, basestring): return obj
        try: return self.ix_to_s[obj]
        except: return self.decode(obj)


    # Nearest, 2nd-nearest, and futhest getters:
    #   Each attempts to return the same type given.
    #   Each ensures neighbors will be calculated before, but not recalculated.
    def nearest(self, obj):
        self.neighbors_getter()
        i = self.neighbors[self.as_index(obj)][0]
        if isinstance(obj, basestring): return self.ix_to_s[i]
        try:
            int(obj)
            return i
        except: return self.space[i]

    def second_nearest(self, obj):
        self.neighbors_getter()
        i = self.neighbors[self.as_index(obj)][1]
        if isinstance(obj, basestring): return self.ix_to_s[i]
        try:
            int(obj)
            return i
        except: return self.space[i]

    def furthest(self, obj):
        self.neighbors_getter()
        i = self.neighbors[self.as_index(obj)][2]
        if isinstance(obj, basestring): return self.ix_to_s[i]
        try:
            int(obj)
            return i
        except: return self.space[i]


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
                self.metric_str if self.metric_str != None else self.metric))
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

    def nearest_neighbors(self):
        return self.neighbors_getter()[0][:,0]

    def nearest_neighbors_dist(self):
        return self.neighbors_getter()[1][:,0]

    def second_nearest_neighbors(self):
        return self.neighbors_getter()[0][:,1]

    def second_nearest_neighbors_dist(self):
        return self.neighbors_getter()[1][:,1]

    def furthest_neighbors(self):
        return self.neighbors_getter()[0][:,2]

    def furthest_neighbors_dist(self):
        return self.neighbors_getter()[1][:,2]


    #--------------------------------------------------------------------------#
    # General Analyses:                                                        #
    #--------------------------------------------------------------------------#

    def _spatial_analysis(self, print_report=True):

        # MEASUREMENTS:

        # Centroid, Dispersion, Std Dev, repulsion:
        self._print("Balancing the Continuum")
        self.centroid = np.mean(self.space, axis=0)
        self.centroid_length = np.linalg.norm(self.centroid)
        self.centroid_dist = [self.metric(self.centroid, v)
            for v in tqdm(self.space, desc="Counting the Lightyears",
                disable=(not self.auto_print))]
        self.dispersion = np.mean(self.centroid_dist, axis=0)
        self.std_dev = np.std(self.space)
        centr_min = np.min(self.centroid_dist, axis=0)
        centr_max = np.max(self.centroid_dist, axis=0)
        if print_report:
            self._add_info(self.ix_to_s[np.argmin([
                self.metric(self.centroid, v) for v in tqdm(self.space,
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
        #    [self.metric(v, self.encoder(self.nearest(self.objects[i])))
        #     for i, v in self.vectors])

        # Nearest Neighbor Info:
        self._print("Building Trade Routes")
        self.nearest_avg = np.mean(self.neighbors_dist[:,0])
        self._print("Practicing Diplomacy")
        nearest_min = np.min(self.neighbors_dist[:,0])
        nearest_max = np.max(self.neighbors_dist[:,0])
        if print_report:
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
        if print_report:
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
        if print_report:
            self._add_info(self.furthest_avg, "Spatial", "Furthest Dist Avg")
            self._add_info(furthest_min, "Spatial", "Furthest Dist Min")
            self._add_info(furthest_max,
                "Spatial", "Broadness - Furthest Dist Max", star=True)
            self._add_info(furthest_max - furthest_min,
                "Spatial", "Furthest Dist Range")
            self._add_info(self.neighbors_dist[:,2],
                "Spatial", "Furthest Dist Histogram Key")


    def _cluster_analysis(self):
        # NOTE: The built-in algorithms are not included in the loop below
        #   because some of them depend on each other, and must be in order.

        # Extremities:
        if "Extremities" in self.categories:

            # Compute the Extremities:
            self.extremities = clusters.clusterizer.compute_extremities(
                self.metric, self.encode, self.neighbors[:,2],
                self.ix_to_s, self.auto_print)
            # Extremity Lengths and other info:
            self._print("Setting the Scopes")
            self._print("Puzzling Over the Star Charts")
            self._add_node_type_attributes(self.extremities, "Extremities",
                [1, 0, 1, 1, 0, 0, 0])

        # Nodes:
        print_node_info = "Nodes" in self.categories
        if (print_node_info or "Hubs" in self.categories or "Supernodes" in
                self.categories or "Nuclei" in self.categories or "Chains" in
                self.categories or "NCC" in self.categories or "LNCC" in
                self.categories or "Anti-hubs" in self.categories):
                # ...all dependent on Nodes.

            # Compute the Nodes:
            self.nodes = clusters.clusterizer.compute_nodes(
                self.metric, self.encode, self.neighbors[:,0],
                self.ix_to_s, self.auto_print)
            for node in self.nodes:
                self.s_to_node[node[0]] = node
                self.s_to_node[node[1]] = node
            # Node Length and other info:
            if print_node_info:
                self._print("Delineating the Quasars")
                self._add_node_type_attributes(self.nodes, "Nodes",
                [0, 0, 0, 0, 0, 0, 0])
                if len(self.nodes) > 0:
                    self._print("Comparing the Cosmos")
                    self._add_info(len(self.nodes)*2.0/float(len(self.space)),
                        "Nodes", "Nodal Factor", star=True)
                    avg_align = np.mean(
                        [n.alignment for n in self.nodes], axis=0)
                    avg_align /= np.linalg.norm(avg_align)
                    self._add_info(
                        np.mean([
                            np.abs(sp.distance.cosine(avg_align, n.alignment))
                            for n in self.nodes]),
                        "Nodes", "Alignment Factor", star=True)

        # Hubs:
        if "Hubs" in self.categories:

            # Compute the Hubs:
            self.hubs = clusters.clusterizer.compute_hubs(
                self.metric, self.encode, self.nearest, self.neighbors[:,0],
                self.ix_to_s, self.s_to_node, self.auto_print)

            # Hub count, populations, etc:
            self._add_cluster_type_attributes(self.hubs, "Hubs")

        # # Supernodes:
        # if "Supernodes" in self.categories and len(self.nodes) >= 2:

        #     # Nearest neighbor-node computation:
        #     self.supernodes = clusters.clusterizer.compute_supernodes(
        #         self.nodes, self._print, self.metric_str,
        #         self.metric, self.auto_print)

        #     # Supernode Length and other info:
        #     self._print("Measuring their Magnitude")
        #     self._add_node_type_attributes(self.supernodes, "Supernodes",
        #         [0, 0, 0, 0, 0, 0, 0])
        #     if len(self.supernodes) > 0:
        #         self._print("Minding the Macrocosm")
        #         self._add_info(len(self.supernodes)*4.0/float(len(self.space)),
        #             "Supernodes", "Island Factor", star=True)
        #         self._add_info(
        #             len(self.supernodes)*2.0/float(len(self.nodes)),
        #             "Supernodes", "Hierarchical Factor", star=True)

        # Nuclei:
        if "Nuclei" in self.categories:
            self.nuclei = clusters.clusterizer.compute_nuclei()
            self._print("Performing Cold Fusion")
            self._add_cluster_type_attributes(self.nuclei, "Nuclei")

        # Chains:
        pass

        # NCC:
        pass

        # LNCC:
        pass

        # Anti-hubs:
        if "Anti-hubs" in self.categories:
            self.anti_hubs = clusters.clusterizer.compute_anti_hubs()
            self._print("Unraveling the Secrets of Dark Matter")
            self._add_cluster_type_attributes(self.anti_hubs, "Anti-hubs")

        # Cluster Algorithms:
        for alg in self.cluster_algorithms:
            # function = alg[0]
            # description = alg[1]
            self._print("Analyzing " + alg[1])

            # Custom Callables:
            if callable(alg[0]):
                if alg[1] == "" or alg[1] == None:
                    alg[1] = alg[0].__name__
                cluster_list = []
                clusterings = alg[0](self.space)
                for i, c in enumerate(clusterings):
                    strings = map(self.decode, c)
                    nodes = self.find_nodes_from_string_list(strings)
                    cluster_list.append(clusters.Cluster(
                        self.encode, self.metric, self.nearest, strings,
                        vectors=c, nodes=nodes, auto=True, ID=i))
                
                self.cluster_data[alg[1]] = cluster_list
                self._add_cluster_type_attributes(cluster_list, alg[1])
            
            # Invalid Non-callables:
            elif (alg[0] not in self.built_in_categories
                    and alg[1] not in self.built_in_categories):
                self._print(
                    alg[0] + ", " + alg[1] +" UNRECOGNIZED and NOT CALLABLE!")

    # ANALOGICAL & SPECIFICS:

    def _analogical_analysis(self):
        pass

    """
        # Specific Functions:
        def rescale(self, theta, alpha=15, power=0.5):
            ''' Rescales based on observed distribution of angles between words
                in a postagged Wikipedia word embedding from BYU PCCL.
                Accepts theta in radians.'''
            return (0.5 + (math.atan((theta*180/np.pi - 90)/alpha)
                            / (2*math.atan(90/alpha))))**power

        def test_angles(self, n, alpha=15, power=0.5):
            dist = [self.rescale(self.s.angle(
                        self.s.get_vector(self.s.model.vocab[int(x)]),
                        self.s.get_vector(self.s.model.vocab[int(2*x)])),
                        alpha, power)
                    for x in (np.random.random(n)*len(self.s.model.vocab)/2.0)]
            plt.hist(dist, 90)
            plt.show()

        #def scale_bimodal(self, theta):
        #    deg = theta*180/np.pi
        #    return 0.5 + (self.cbrt((deg-90)) / (2*self.cbrt(90)))

        def cluster_analogy(self, A, B, C, AC_clustername, B_clustername,
                            num_words=1, exclude=True):
            ''' Follows form: A:B::C:D.
                Assumes that we know which cluster each word comes from.'''
            dist = self.s.get_angle(A, B)
            A_tighter = (self.clusters[AC_clustername][1]
                        <= self.clusters[B_clustername][1]
            C_vec = self.s.get_vector(C)
            dir_vec = self.clusters[AC_clustername][0] - C_vec
            if A_tighter: dir_vec = -dir_vec
            D_vec = self.s.yarax(C_vec, dir_vec, dist)
            D_vec /= np.linalg.norm(D_vec)

            if exclude:
                if self.s.slim == True: # This branch other part of patch:
                    results = self.s.wordify(
                        self.s.model.get_closest_words(D_vec, num_words+3))
                    trimmed = ([word for word in results[0]
                                if word not in [A, B, C]],
                            [results[1][i] for i in range(len(results[1]))
                                if results[0][i] not in [A, B, C]])
                    return (np.array(trimmed[0][:num_words:]),
                            np.array(trimmed[1][:num_words:]))
                else: # This branch is the original return:
                    return self.s.wordify(self.s.model.get_closest_words_excluding(
                        D_vec, [self.s.get_vector(A), self.s.get_vector(B), C_vec],
                        num_words))
            else: # The real original return...
                return self.s.wordify(
                    self.s.model.get_closest_words(D_vec, num_words))

        def divergence_analogy(self, A, B, C):
        ''' Automatically tries to find clusters around A and B,
            and then does a cluster analogy.'''
        raise NotImplementedError("Function not implemented.")
    """

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

    def _add_clusters_attribute(self, vals, cluster_type, attribute, stars=[]):
        # vals: a list containing the given attribute for each cluster.
        # cluster_type: ie. "Hubs".
        # attribute: ie. "Dispersion".
        hub_max = np.max(vals)
        hub_min = np.min(vals)
        self._add_info(np.mean(vals), cluster_type, attribute + " Avg", stars[0])
        self._add_info(hub_min, cluster_type, attribute + " Min", stars[1])
        self._add_info(hub_max, cluster_type, attribute + " Max", stars[2])
        self._add_info(hub_max-hub_min, cluster_type, attribute + " Range", stars[3])
        self._add_info(np.std(vals), cluster_type, attribute + " Standard Dev", stars[4])
        self._add_info(vals, cluster_type, attribute + " Histogram Key", stars[5])

    def _add_cluster_type_attributes(self, cluster_list, cluster_type):
        # cluster_list: a list of all clusters of the given type.
        # cluster_type: ie. "Hubs".
        self._add_info(len(cluster_list), cluster_type, "Count")
        if len(cluster_list) > 0:
            self._add_clusters_attribute(map(len, cluster_list),
                cluster_type, "Population")
            self._add_clusters_attribute([c.dispersion for c in cluster_list],
                cluster_type, "Dispersion")
            self._add_clusters_attribute([c.repulsion for c in cluster_list],
                cluster_type, "Repulsion")
            self._add_clusters_attribute([c.skew for c in cluster_list],
                cluster_type, "Skew")
            self._add_clusters_attribute([len(c.nodes) for c in cluster_list],
                cluster_type, "Node Count")

    def _add_node_type_attributes(self, node_list, node_type, stars):
        # node_list: list of all nodes of the given type.
        # node_type: ie. "Extremities".
        # stars: boolean list of length 7 for which attributes are important.
        self._add_info(len(node_list), node_type, "Count", stars[0])
        if len(node_list) > 0:
            lengths = [n.distance for n in node_list]
            self._add_clusters_attribute(lengths, node_type, "Span", stars[1:])

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

    """
    def _serialize(self):
        # method to prepare an analyst instance for saving.
        if not self.serialized:
            self.metric = self.metric.__name__ if self.metric != None else None
            self.encode = self.encode.__name__ if self.encode != None else None
            self.decode = self.decode.__name__ if self.decode != None else None
            self.cluster_algorithms = [
                (None, pair[1]) for pair in self.cluster_algorithms]
            self.analogy_algorithms = [
                (None, pair[1]) for pair in self.analogy_algorithms]

            # Serialize the Clusters (Nodes don't need it):
            for key in self.cluster_data:
                for cluster in self.cluster_data[key]:
                    cluster._serialize()

            self.serialized = True

    def _deserialize(self, metric, encoder, decoder,
                     cluster_algorithms, analogy_algorithms):
        assert self.serialized
        if callable(metric):
            assert metric.__name__ == self.metric
            self.metric = metric
        elif metric == "l2" or metric == "euclidean":
            self.metric = sp.distance.euclidean
        elif metric == "cosine_similarity":
            self.metric = sp.distance.cosine
        elif metric == "l1":
            self.metric = sp.distance.cityblock
        else: raise ValueError("'metric' parameter unrecognized and uncallable")            
            
        if encoder != None: assert encoder.__name__ == self.encode
        self.encode = encoder
        if decoder != None: assert decoder.__name__ == self.decode
        self.decode = decoder

        if cluster_algorithms != None:
            assert zip(*cluster_algorithms)[1] == zip(*self.cluster_algorithms)[1]
        self.cluster_algorithms = cluster_algorithms
        if analogy_algorithms != None:
            assert zip(*analogy_algorithms)[1] == zip(*self.analogy_algorithms)[1]
        self.analogy_algorithms = analogy_algorithms

        # Deserialize the Clusters:
        for key in self.cluster_data:
            for cluster in self.cluster_data[key]:
                cluster._deserialize(metric, encoder, decoder)

        self.serialized = False
    """

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
