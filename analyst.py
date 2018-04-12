import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
#import cPickle as pickle
import dill as pickle
import os
import sys
import multiprocessing

import clusters
#from test_set_2d import TestSet2D


class Analyst:
    """
    Description:
        A toolset for embedding space analytics.


    Use:
            Generally you would initialize one Analyst instance per one
        embedding space, and perform analyses and access tools in the toolset
        through that analyst. The exception is in experimentation with
        differing metrics; here you would use multiple analysts initialized
        with the same embeddings.
            The Analyst is designed to be abstract; it requires you to tell it
        what metric to use or to define your own. Likewise it requires a list of
        cluster types to compute - either callables or recognized tags - and
        it will iterate through and compute all of them. The built-in,
        experimental clustering algorithms will save each other compute time
        where possible by using already-processed parts from others, but custom
        input clustering functions will be processed entirely individually.
        Thereafter, the Analyst will run its same analyses on each of those
        cluster types no matter the source.
            The objects encoded in the space must be some sort of strings, or
        the internal conversion functions will not work.


    Definitions:
        node: a pair of obj in the space whose nearest neighbors are each other.
        supernode: a pair of nodes whose nearest neighbor-nodes are each other.
        extremity: a pair of objects in the space whose furthest neighbors are
            each other. (implemented with the same Node class as nodes are.)
        outlier: an object which is a member of an extremity.
        loner: an object which has been rejected when forming clusters,
            making it a cluster unto itself, of sorts.
        hub: an obj that is the nearest neigbor of three or more other objects.
        nodal factor: ratio of words belonging to nodes;
            a measure of the scale or impact of relationships in the space
        alignment factor: normalize mean of vectors from a to b in nodes, then
            measure average absolute value of cosine similarity of each to that.
            Think magnetic moments - how lined-up nodes are in the space.
        hierarchical factor: ratio of nodes belonging to supernodes;
            a further measure of relationships in the space.
        island factor: ratio of objects belonging to supernodes;
            another measure of relationships in the space.
        nucleus: grouping of relatively nearby objects. Starting with nodes and
            all obj whose nearest are one of those, then finding average dist
            to center, then combining all clusters whose central nodes are
            closer than one of their averages.
        chain: different clustering where each cluster has a sole node and
            recursively finds all whose nearest's nearest's etc. nearest is a
            member of that node.
        cluster: Find nuclei, then add chainable objects whose nearest is closer
            than the avg dist of those whose nearest are in one of the nodes,
            (those we started with only) and whose nearest is in the cluster.
        strong cluster: Same as cluster, but builds nuclei and clusters
            requiring objects to have both first and second nearest belonging to
            the same grouping in order for them to be added.
        unpartitioned clusters: strong clusters, except we find all contingent
            clusters, and add all objects from those that are closer than own
            dispersion (avg obj dist from centroid). //MAY CHANGE - USE A DROPOFF DISTANCE DEPENDING ON MAX DIST OF OBJECTS IN NUCLEUS?
        center: average location of all objects in a cluster.
        string factor: average cluster span divided by space dispersion
        regularity: find average cluster pop, then find average difference btw.
            cluster pop and average cluster pop. Regularity is 1/(1+this)
        remoteness: as used here, avg distance of nodes to their nearest
            neighbors. While not a real measure of spatial or volumentric
            density, this is a metric for relationships between encoded objects,
            and thus could potentially be used as a density metric if inverted.
        dispersion: avg distance of nodes to the center of their distribution
        focus: averaged location of nodes in a cluster; concentration center.
        skew: distance from focus to center.
        anti-cluster: list of objects whose furthest are all the same outlier.
        contingent cluster: other clusters whose centroid is closer than own //ADD STUFF!!
            dispersion, to which self is closer than other's dispersion.
        contingency: distance from a cluster to its nearest neighbor // ADD STUFF!
            cluster, minus its own dispersion


    Tools:
        NOTE: Those in parentheses cannot be used as measures of the properties
            of the embedding space since they depend directly on the number of
            embeddings, dimensionality, or some similar property.
            These could be seen as specific information.
        NOTE: Those properties with a * are of most import in measuring a space.

        General:
            NOTE: These are all static methods.
            Analyst.save(obj, path) -- Returns True if successful.
                General use case is for an already-processed analyst object,
                but should work on most objects in most cases.
                Will overwrite files with the same name.
                Detects/automatically adds .pickle extensions.
            Analyst.load(path) -- returns unpickled object, or None if failed.
            Analyst.unsave(path) -- deletes a saved file. Rtrns True if success.

        Spatial:
            centroid
            dist. to centroid avg, min, max, range, graph of distribution of.
            * medoid

            * dispersion

            * remoteness -- avg dist to nearest
            dist. to nearest min, max, *range, graph of distribution of.
            * broadness -- max dist to furthest
            dist. to furthest avg, min, range, graph of distr.

        Clustering:
            NOTE: For each property of a cluster type, these stats are always
                available: avg, min, max, range, distribution graph of.

            Extremities: (Mutual Furthest-Neighbor Pairs)
                * (num extremities -- probably depends strongly on
                    dimensionality, but shows the spherical-ness of the
                    distribution in the space)
                extremity length avg, * min, * max, range, graph of distr.

            Nodes: (Mutual Nearest-Neighbor Pairs)
                (num nodes)
                * nodal factor
                node length avg, min, max, range, graph of distr.
                * alignment factor

            Hubs: (Nodal Proximity Groups)
                (num hubs)
                hub num stats

            Supernodes: (Hierarchical Node Pairs)
                (num supernodes)
                * hierarchical factor, burst factor
                * island factor
                supernode length stats

            Nuclei: (Multi-Nodal Proximity Groups)
                (num nuclei)
                * nucleus factor (num nuclei divided by num objects)
                ratio in nuclei versus not
                nucleus population stats
                nuclei string factor, nucleus span stats
                nucleus regularity
                nucleus dispersion factor -- avg. nucleus disp. / space disp,
                    (nucleus dispersion stats)
                node count stats
                nucleus remoteness factor -- avg. nucleus remoteness divided by
                    overall space remoteness,
                    (nucleus remoteness stats)
                nucleus skew factor, (nucleus skew stats)

            Chains: (Nearest-Neighbor-Path Partitioning Groups)
                chain population stats
                chain string factor -- avg. chain span / space dispersion,
                    (chain span stats)
                chain dispersion factor -- avg. chain disp. / space disp,
                    (chain dispersion stats)
                chain remoteness factor -- avg. chain remoteness / overall space
                    remoteness, (chain remoteness stats)
                chain skew factor, (chain skew stats)
                NOTE: num chains is equal to num nodes
                NOTE: all objects in the space belong to a chain

            Clusters: (Nearest-Neighbor NODAL Conglomerate CLUSTERS)
                (num clusters)
                * cluster factor -- num clusters divided by num objects
                * string factor -- avg. cluster span / space dispersion
                * regularity, (cluster span stats)
                * ratio clustered versus loners
                * avg cluster population, cluster population stats
                * cluster dispersion factor -- avg. cluster disp. / space disp,
                    (cluster dispersion stats)
                * avg num nodes per cluster, node count stats
                * cluster remoteness factor -- avg cluster remoteness / overall
                    space remoteness, (cluster remoteness stats)
                * cluster skew factor -- avg. cluster skew / space dispersion,
                    (cluster skew stats)

            Strong Clusters: (Dispersion and Dual LIMITED Nearest-Neighbor NODAL
                Conglomerate CLUSTERS)
                Same info as for clusters.

            Anti-clusters: (Common Futhest-Neighbor Groups)
                More or less the same information as for clusters,
                but it WILL not mean the same things. Note that these clusters
                DO NOT include the word that is their farthest neighbor.

        Analogical:
            run_analogies() !!!CANT--UNKNOWN OUTPUT??!!!
            member_of(object) -- displays cluster this object is a member of.
            cluster([list of objects]) -- a new cluster composed solely of the
                given objects.
            seeded_cluster([list of objects]) -- a new cluster composed of all
                nearby objects likely to be clustered with these, if these were
                treated as being together.
            inspect_clustering([list of objects]) -- analysis on given objects,
                Returns:
                - number of unique clusters these words are found across
                - average ward dissimilarity of involved clusters
                - list of tuples containing: (object, cluster_index)
            circular_walk_graph(obj1, obj2) -- most useful in a normalized
                space, like word2vec.

        Comparative:
            compare_difference(analyst2, simple_diff=False)
                -- prints a full report with three numbers for each property:
                val_for_A, val_for_B, A_B_compared.
                The third number is a representation of how different A, B are,
                    either a simple difference or weighted by their scales,
                    depending on the value of simple_diff.
            Analyst.compare([list_of_analysts]) -- a @staticmethod which lists
                side by side the values for each analyst in the list.

        Specifics / Inspection:
            rank_outliers() -- by number of obj. for which this one is furthest
                neighbor. Resulting list contains exactly all objects which are
                members of an extremity.
            rank_clusters() -- by size; lists the indeces of the clusters.
            rank_hubs() -- by num of obj for which this one is nearest neighbor.
            graph(graph_key, bins) -- produce graph given key printed in report.
            centroid -- accessible vector; can be used externally.
            * clusters -- accessible variable; a list of the clusters.
                Further info is available in the internal vars of each cluster.
            * strong clusters -- ''
            nodes -- ''
            supernodes -- ''
            nuclei -- ''
            chains -- ''
            extremities -- ''
            anti-clusters -- dictionary keyed to outlier objects,
                containing anti-clusters.
            as_string(obj) -- generic type converters for individual objects
            as_index(obj)
            as_vector(obj)

        Simulation:
            Analyst.simulate_space() -- @staticmethod which generates an entire
                fake embedding space with the specified properties,
                and returns it wrapped in a new analyst.
                NOTE: Includes cluster generation. No need to add to it.
            Analyst.simulate_cluster() -- @staticmethod which generates generic
                test clusters to compare with, or to examine properties.
                Available cluster types listed in function comments.
            TestSet2D -- a class which can be treated like a small 2D embedding
                space and fed into an analyst for testing. Has encoder and
                decoder functions to be fed in also.
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
            encoder -- a callable to convert strings to vectors.
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
        self._print("Laying the Laws of Physics")
        if callable(metric): self.metric = metric
        else:
            try:
                self.metric = sp.distance._TEST_METRICS[
                    "test_" + sp.distance._METRIC_ALIAS[metric]]
            except e:
                print(e)
                raise ValueError("'metric' parameter unrecognized and uncallable")
        
        self.description = desc

        # Encoder/Decoder Initializations:
        #   While initializing these should, in theory, be unnecessary,
        #   failing to do so will limit all inputs to findable types.
        self._print("Filling the Void")
        self.encode = encoder # string to vector
        self.decode = decoder # vector to string
        self.serialized = False

        if calculate:
            self.s_to_ix = {}
            self.ix_to_s = []
            try:
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

            self.neighbors = np.empty((len(self.space),3), dtype=np.uint64)
                # Indeces correspond to indeces of vectors in the space.
                #   For each:
                #   [index of nearest, index of 2nd-nearest, index of furthest]
                #   These are filled in in the _spatial_analysis.
            self.neighbors_dist = np.empty(
                (len(self.space),3))
                # Same format as above, except distances to those indexed above.

            # Run Analyses:
            self.graph_info = []
            self.categories = []
            self.cluster_algorithms = []
            report_spatial = False
            for t in cluster_algorithms:
                if callable(t[0]):
                    self.cluster_algorithms.append(t)
                else:
                    tag = t[1].lower()
                    if tag == "spatial":
                        report_spatial = True
                        self.categories.append(t[1])
                    elif tag == "all":
                        report_spatial = True
                        self.categories.append("Spatial")
                        self.categories.append("Extremities")
                        self.categories.append("Nodes")
                        self.categories.append("Hubs")
                        self.categories.append("Supernodes")
                        self.categories.append("Nuclei")
                        self.categories.append("Chains")
                        self.categories.append("NNNCC")
                        self.categories.append("LNNNCC")
                        self.categories.append("Anti-clusters")
                    else:
                        self.categories.append(t[1])
            self.category_lists = np.empty(
                shape=(len(self.categories),0)).tolist()

            self.analogy_algorithms = analogy_algorithms
            self.analogy_sets = analogy_sets

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
    #   Each attempts to return the same type given
    def nearest(self, obj):
        i = self.neighbors[self.as_index(obj)][0]
        if isinstance(obj, basestring): return self.ix_to_s[i]
        try:
            int(obj)
            return i
        except: return self.space[i]

    def second_nearest(self, obj):
        i = self.neighbors[self.as_index(obj)][1]
        if isinstance(obj, basestring): return self.ix_to_s[i]
        try:
            int(obj)
            return i
        except: return self.space[i]

    def furthest(self, obj):
        i = self.neighbors[self.as_index(obj)][2]
        if isinstance(obj, basestring): return self.ix_to_s[i]
        try:
            int(obj)
            return i
        except: return self.space[i]

    #--------------------------------------------------------------------------#
    # General Analyses:                                                        #
    #--------------------------------------------------------------------------#

    def _spatial_analysis(self, print_report=True):

        # DISTANCE AND NEIGHBOR COMPUTATION:

        # Distance Matrix Calculation
        self._print("Acquanting the Species")
        self.distance_matrix = sp.distance.squareform(
            sp.distance.pdist(self.space, self.metric))
            
        # Finding Furthest Neighbors
        self._print("Misconstruing Relations")
        self.neighbors[:,2] = np.argmax(self.distance_matrix, axis=1)
        self.neighbors_dist[:,2] = self.distance_matrix[
            range(len(self.distance_matrix)),
            self.neighbors[:,2]]
        self.distance_matrix[
            range(len(self.space)), range(len(self.space))] = np.inf
        # Finding Nearest Neighbors
        self._print("Forming Alliances")
        self.neighbors[:,0] = np.argmin(self.distance_matrix, axis=1)
        self.neighbors_dist[:,0] = self.distance_matrix[
            range(len(self.distance_matrix)),
            self.neighbors[:,0]]
        self.distance_matrix[
            range(len(self.space)), self.neighbors[:,0]] = np.inf
        # Finding Second-Nearest Neighbors
        self._print("Obfuscating Dynastic Ties")
        self.neighbors[:,1] = np.argmin(self.distance_matrix, axis=1)
        self.neighbors_dist[:,1] = self.distance_matrix[
            range(len(self.distance_matrix)),
            self.neighbors[:,1]]
        # Put back the numbers we removed:
        self._print("Resetting the Ship's Computer")
        self.distance_matrix[
            range(len(self.space)), self.neighbors[:,0]
        ] = self.neighbors_dist[:,0]
        self.distance_matrix[
            range(len(self.space)), range(len(self.space))] = 0.0


        # MEASUREMENTS:

        # Centroid, Dispersion, Remoteness:
        self._print("Balancing the Continuum")
        self.centroid = np.mean(self.space, axis=0)
        #self._add_info(self.centroid,
        #    "Spatial", "Centroid - Coordinate Avg")
        self._add_info(self.ix_to_s[np.argmin([
            self.metric(self.centroid, v) for v in tqdm(self.space,
                desc="Electing a Ruler", disable=(not self.auto_print))])],
            "Spatial", "Medoid - Obj Nearest to Centroid", star=True)
        self.centroid_dist = [self.metric(self.centroid, v)
            for v in tqdm(self.space, desc="Counting the Lightyears",
                disable=(not self.auto_print))]
        self.dispersion = np.mean(self.centroid_dist, axis=0)
        self._add_info(self.dispersion,
            "Spatial", "Dispersion - Centroid Dist Avg", star=True)
        centr_min = np.min(self.centroid_dist, axis=0)
        centr_max = np.max(self.centroid_dist, axis=0)
        self._add_info(centr_min, "Spatial", "Centroid Dist Min")
        self._add_info(centr_max, "Spatial", "Centroid Dist Max")
        self._add_info(centr_max - centr_min,
            "Spatial", "Centroid Dist Range")
        self._add_info(self.centroid_dist,
            "Spatial", "Centroid Dist Histogram Key")
        #self.remoteness = np.mean(
        #    [self.metric(v, self.encoder(self.nearest(self.objects[i])))
        #     for i, v in self.vectors])


        """
        # Nearest, 2nd Nearest, and Futhest Computation:
        self._print("Ousting Nearly Empty Universes") #"Ousting the Flatlanders"
        if len(self.space) < 4:
            return
        #self._print("Acquainting the Species:")
        #pbar = tqdm(total=len(self.space), desc="Analyzing Stellar Adjacency",
        #    disable=(not self.auto_print)) #USE WITH STATEMENT INSTEAD
        self.counter = 0
        try: cpus = multiprocessing.cpu_count()
        except: cpus = 2
        pool = multiprocessing.Pool(cpus)
        pool.map(_compute_neighbors, range(len(self.space)))#[pbar]*len(self.space))
        #pbar.close()
        """


        # Nearest Neighbor Info:
        self._print("Building Trade Routes")
        self.nearest_avg = np.mean(self.neighbors_dist[:,0])
        self._add_info(self.nearest_avg,
            "Spatial", "Remoteness - Nearest Dist Avg", star=True)
        self._print("Practicing Diplomacy")
        nearest_min = np.min(self.neighbors_dist[:,0])
        near = np.argmin(self.neighbors_dist[:,0])
        nearest_max = np.max(self.neighbors_dist[:,0])
        self._add_info(nearest_min, "Spatial", "Nearest Dist Min")
        self._add_info(nearest_max, "Spatial", "Nearest Dist Max")
        self._add_info(nearest_max-nearest_min,
            "Spatial", "Nearest Dist Range", star=True)
        self._add_info(self.neighbors_dist[:,0],
            "Spatial", "Nearest Dist Histogram Key")

        # Second-Nearest Neighbor Info:
        self._print("Setting Priorities")
        self.nearest2_avg = np.mean(self.neighbors_dist[:,1])
        self._add_info(self.nearest2_avg,
            "Spatial", "Second Nearest Dist Avg")
        self._print("Coming up with Excuses")
        nearest2_min = np.min(self.neighbors_dist[:,1])
        nearest2_max = np.max(self.neighbors_dist[:,1])
        self._add_info(nearest2_min, "Spatial", "Second Nearest Dist Min")
        self._add_info(nearest2_max, "Spatial", "Second Nearest Dist Max")
        self._add_info(nearest2_max-nearest2_min,
            "Spatial", "Second Nearest Dist Range")
        self._add_info(self.neighbors_dist[:,1],
            "Spatial", "Second Nearest Dist Histogram Key")

        #Furthest Neighbor Info:
        self._print("Making Enemies")
        self.furthest_avg = np.mean(self.neighbors_dist[:,2])
        self._add_info(self.furthest_avg, "Spatial", "Furthest Dist Avg")
        self._print("Claiming Frontiers")
        furthest_min = np.min(self.neighbors_dist[:,2])
        furthest_max = np.max(self.neighbors_dist[:,2])
        far = np.argmax(self.neighbors_dist[:,2])
        self._add_info(furthest_min, "Spatial", "Furthest Dist Min")
        self._add_info(furthest_max,
            "Spatial", "Broadness - Furthest Dist Max", star=True)
        self._add_info(furthest_max - furthest_min,
            "Spatial", "Furthest Dist Range")
        self._add_info(self.neighbors_dist[:,2],
            "Spatial", "Furthest Dist Histogram Key")


    def _cluster_analysis(self):

        # Extremities:
        if "Extremities" in self.categories:

            # Compute the Extremities:
            self.extremities = [
                clusters.Node(self.ix_to_s[i],
                    self.ix_to_s[self.neighbors[i][2]],
                    self.encode, self.metric)
                for i in tqdm(range(len(self.space)),
                    desc="Measuring the Reaches",
                    disable=(not self.auto_print))
                if (i == self.neighbors[self.neighbors[i][2]][2]
                    and i < self.neighbors[i][2])]

            # Extremity Lengths and other info:
            self.extremity_lengths = [e.distance for e in tqdm(self.extremities,
                desc="Setting the scopes",
                disable=(not self.auto_print))]
            self._print("Puzzling Over the Star Charts")
            self._add_info(len(self.extremities),
                "Extremities", "Count", star=True)
            if len(self.extremities) > 0:
                extr_min = np.min(self.extremity_lengths)
                extr_max = np.max(self.extremity_lengths)
                self._add_info(np.mean(self.extremity_lengths),
                    "Extremities", "Span Avg")
                self._add_info(extr_min, "Extremities", "Span Min", star=True)
                self._add_info(extr_max, "Extremities", "Span Max", star=True)
                self._add_info(extr_max - extr_min, "Extremities", "Span Range")
                self._add_info(self.extremity_lengths,
                    "Extremities", "Span Histogram Key")

        # Nodes:
        print_node_info = "Nodes" in self.categories
        if (print_node_info or "Hubs" in self.categories or "Supernodes" in
                self.categories or "Nuclei" in self.categories or "Chains" in
                self.categories or "NNNCC" in self.categories or "LNNNCC" in
                self.categories or "Anti-clusters" in self.categories):
                # ...all dependent on Nodes.

            # Compute the Nodes:
            self.nodes = [
                clusters.Node(self.ix_to_s[i],
                    self.ix_to_s[self.neighbors[i][0]],
                    self.encode, self.metric)
                for i in tqdm(range(len(self.space)),
                    desc="Watching the Galaxies Coelesce",
                    disable=(not self.auto_print))
                if (i == self.neighbors[self.neighbors[i][0]][0]
                    and i < self.neighbors[i][0])]

            # Node Length and other info:
            self.node_lengths = [n.distance for n in tqdm(self.nodes,
                desc="Delineating the Quasars",
                disable=(not self.auto_print))]
            self._print("Comparing the Cosmos")
            self._add_info(len(self.nodes), "Nodes", "Count")
            if print_node_info and len(self.nodes) > 0:
                node_min = np.min(self.node_lengths)
                node_max = np.max(self.node_lengths)
                self._add_info(np.mean(self.node_lengths), "Nodes", "Span Avg")
                self._add_info(node_min, "Nodes", "Span Min")
                self._add_info(node_max, "Nodes", "Span Max")
                self._add_info(node_max - node_min, "Nodes", "Span Range")
                self._add_info(len(self.nodes)*2.0/float(len(self.space)),
                    "Nodes", "Nodal Factor", star=True)
                avg_align = np.mean([n.alignment for n in self.nodes], axis=0)
                avg_align /= np.linalg.norm(avg_align)
                self._add_info(
                    np.mean([
                        np.abs(sp.distance.cosine(avg_align, n.alignment))
                        for n in self.nodes]),
                    "Nodes", "Alignment Factor", star=True)
                self._add_info(self.node_lengths, "Nodes", "Span Histogram Key")

        # Hubs:
        if "Hubs" in self.categories:

            # Compute the Hubs:
            self.hubs = []
            temp_hubs = []
            for i in tqdm(range(len(self.space)),
                    desc="Finding Galactic Hubs",
                    disable=(not self.auto_print)):
                temp_hubs.append(clusters.Cluster(
                    self.encode, self.metric, nearest=self.nearest,
                    objects=[self.ix_to_s[i]], nodes=[], auto=False,
                    name=self.ix_to_s[i]))
                    # Its name is the original object's decoded string.
                for index, neighbor in enumerate(self.neighbors[:,0]):
                    if neighbor == i:
                        temp_hubs[i].add_objects([self.ix_to_s[index]])
                    # The 0th index in the hub's list of objects
                    #   is also it's original object (is included in hub).
            j = 0
            for h in tqdm(temp_hubs, desc="Erecting Centers of Commerce",
                    disable=(not self.auto_print)):
                if len(h) >= 4: # obj plus 3 or more for whom it is nearest.
                    self.hubs.append(h)
                    self.hubs[j].ID = j
                    self.hubs[j].calculate()
                    j += 1
            del(temp_hubs) # To save on memory.

            # Hub count, populations, etc:
            self._add_info(len(self.hubs), "Hubs", "Count")
            if len(self.hubs) > 0:
                hub_sizes = [len(h) for h in self.hubs]
                hub_min = np.min(hub_sizes)
                hub_max = np.max(hub_sizes)
                self._add_info(np.mean(hub_sizes), "Hubs", "Population Avg")
                self._add_info(hub_min, "Hubs", "Population Min")
                self._add_info(hub_max, "Hubs", "Population Max")
                self._add_info(hub_max-hub_min, "Hubs", "Population Range")
                self._add_info(hub_sizes, "Hubs", "Population Histogram Key")

        """
        # Supernodes:
        if "Supernodes" in self.categories:
            self.supernodes = []
            if len(self.nodes) >= 2:

                # Nearest neighbor-node computation:
                node_centroids = [n.centroid for n in self.nodes]
                node_neighbors = []
                node_neighbors_dist = []
                for i, vec in enumerate(tqdm(node_centroids,
                        disable=(not self.auto_print),
                        desc="Forming Coalitions")):
                    nearest_i = (0 if i != 0 else 1) # Can't start off on self!
                    nearest_dist = self.metric(vec, node_centroids[nearest_i])
                    for j, other in enumerate(node_centroids):
                        if j != i:
                            dist = self.metric(vec, other)
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest_i = j
                    node_neighbors.append(nearest_i)
                    node_neighbors_dist.append(nearest_dist)
                ?????????????
                # Compute the Supernodes:
        """

        #MAKE ANALYST ACCEPT LIST OF WORDS INSTEAD OF ONLY AN ENCODE FUNCTION!

        # Nuclei:
        pass

        # Chains:
        pass

        # NNNCC:
        pass

        # LNNNCC:
        pass

        # Anti-clusters:
        pass

        # Other Callables:
        pass
        #for algorithm in self.cluster_algorithms:
        #    clusters = []
        #    clusterings = algorithm[0](self.space)
        #    for c in clusterings:
        #        clusters.append(clusters.Cluster(c))
        #
        # or can use __name__ to get function name!

        # Invalid non-callables:
        pass

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
    """

    """
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
                    "pairs" (generates points in pairs of close remoteness --
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
        #if f_name[-7:] == ".pickle": return f_name
        #if f_name[-4:] == ".pck": return f_name
        #if f_name[-4:] == ".pcl": return f_name
        #if f_name[-3:] == ".db": return f_name
        #if f_name[-8:] == ".analyst": return f_name
        #if f_name[-3:] == ".an": return f_name
        #return f_name if f_name[-4:] == ".pkl" else f_name + ".pkl"

        #return f_name if "." in f_name else f_name + ".pkl"
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
            """assert obj.__class__.__name__ == "Analyst"
            data = {
                "cls"   :  obj.__class__.__name__,
                "space" :  obj.space,
                "metric": obj.metric.__name__ if obj.metric != None else None,
                "encode": obj.encode.__name__ if obj.encode != None else None,
                "decode": obj.decode.__name__ if obj.decide != None else None,
                "c_algs": [(None, pair[1]) for pair in obj.cluster_algorithms],
                "a_algs": [(None, pair[1]) for pair in obj.analogy_algorithms],
                "a_sets": obj.analogy_sets,
                "print" : obj.auto_print,
                "desc"  : obj.description,
                "calc"  : obj.calculate,
                "s_ix"  : obj.s_to_ix,
                "ix_s"  : obj.ix_to_s,
                "neigh" : obj.neighbors,
                "n_dist": obj.neighbors_dist,
                "g_info": obj.graph_info,
                "cats"  : obj.categories,
                "clists": obj.category_lists#,
                #"s_data": obj.spatial_data,????
                #"c_data": obj.cluster_data,????
                #"a_data": obj.analogical_data????
            }"""
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
