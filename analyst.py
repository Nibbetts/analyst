import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
#import cPickle as pickle
import dill as pickle
import os
#import sys
#import multiprocessing

import clusters
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
                * remoteness -- avg dist to nearest
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
        self._print("Laying the Laws of Physics")
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
        self._print("Filling the Void")
        self.ix_to_s = None
        if callable(encoder) or encoder is None:
            self.encode = encoder # string to vector
            self.decode = decoder # vector to string
        else:
            assert len(encoder) == len(self.space)
            self.ix_to_s = encoder
            self.vec_to_s = {}
            self.s_to_vec = {}
            self._print("Mapping the Emptiness")
            for i in range(len(self.space)):
                self.vec_to_s[self.space[i]] = self.ix_to_s[i]
                self.s_to_vec[self.ix_to_s[i]] = self.space[i]
            self.encode = self.s_to_vec.__getitem__
            self.decode = self.vec_to_s.__getitem__

        #self.serialized = False

        if calculate:
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
                        self.categories.append("NCC")
                        self.categories.append("LNCC")
                        self.categories.append("Anti-hubs")
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
        self._print("Acquainting the Species")
        self.distance_matrix = sp.distance.squareform(
            sp.distance.pdist(
                self.space,
                self.metric_str if self.metric_str != None else self.metric))
            
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
        self.centroid_dist = [self.metric(self.centroid, v)
            for v in tqdm(self.space, desc="Counting the Lightyears",
                disable=(not self.auto_print))]
        self.dispersion = np.mean(self.centroid_dist, axis=0)
        centr_min = np.min(self.centroid_dist, axis=0)
        centr_max = np.max(self.centroid_dist, axis=0)
        if print_report:
            self._add_info(self.ix_to_s[np.argmin([
                self.metric(self.centroid, v) for v in tqdm(self.space,
                    desc="Electing a Ruler", disable=(not self.auto_print))])],
                "Spatial", "Medoid - Obj Nearest to Centroid", star=True)
            self._add_info(len(self.space), "Spatial", "Count")
            self._add_info(self.dispersion,
                "Spatial", "Dispersion - Centroid Dist Avg", star=True)
            self._add_info(centr_min, "Spatial", "Centroid Dist Min")
            self._add_info(centr_max, "Spatial", "Centroid Dist Max")
            self._add_info(centr_max - centr_min,
                "Spatial", "Centroid Dist Range")
            self._add_info(self.centroid_dist,
                "Spatial", "Centroid Dist Histogram Key")
        #self.remoteness = np.mean(
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
                "Spatial", "Remoteness - Nearest Dist Avg", star=True)
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

        # Extremities:
        if "Extremities" in self.categories:

            # Compute the Extremities:
            self.extremities = clusters.clusterizer.compute_extremities(
                self.metric, self.encode, self.neighbors[:,2],
                self.ix_to_s, self.auto_print
            )
            #self.extremities = [
            #    clusters.Node(self.ix_to_s[i],
            #        self.ix_to_s[self.neighbors[i][2]],
            #        self.encode, self.metric)
            #    for i in tqdm(range(len(self.space)),
            #        desc="Measuring the Reaches",
            #        disable=(not self.auto_print))
            #    if (i == self.neighbors[self.neighbors[i][2]][2]
            #        and i < self.neighbors[i][2])]

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
                self.categories or "NCC" in self.categories or "LNCC" in
                self.categories or "Anti-hubs" in self.categories):
                # ...all dependent on Nodes.

            # Compute the Nodes:
            self.nodes = clusters.clusterizer.compute_nodes(
                self.metric, self.encode, self.neighbors[:,0],
                self.ix_to_s, self.auto_print
            )
            #self.nodes = [
            #    clusters.Node(self.ix_to_s[i],
            #        self.ix_to_s[self.neighbors[i][0]],
            #        self.encode, self.metric)
            #    for i in tqdm(range(len(self.space)),
            #        desc="Watching the Galaxies Coelesce",
            #        disable=(not self.auto_print))
            #    if (i == self.neighbors[self.neighbors[i][0]][0]
            #        and i < self.neighbors[i][0])]

            # Node Length and other info:
            self.node_lengths = [n.distance for n in tqdm(self.nodes,
                desc="Delineating the Quasars",
                disable=(not self.auto_print))]
            self._print("Comparing the Cosmos")
            if print_node_info:
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
            self.hubs = clusters.clusterizer.compute_hubs(
                self.metric, self.encode, self.nearest, self.neighbors[:,0],
                self.ix_to_s, self.auto_print
            )
            #self.hubs = []
            #temp_hubs = []
            #for i in tqdm(range(len(self.space)),
            #        desc="Finding Galactic Hubs",
            #        disable=(not self.auto_print)):
            #    temp_hubs.append(clusters.Cluster(
            #        self.encode, self.metric, nearest=self.nearest,
            #        objects=[self.ix_to_s[i]], nodes=[], auto=False,
            #        name=self.ix_to_s[i]))
            #        # Its name is the original object's decoded string.
            #    for index, neighbor in enumerate(self.neighbors[:,0]):
            #        if neighbor == i:
            #            temp_hubs[i].add_objects([self.ix_to_s[index]])
            #        # The 0th index in the hub's list of objects
            #        #   is also it's original object (is included in hub).
            #j = 0
            #for h in tqdm(temp_hubs, desc="Erecting Centers of Commerce",
            #        disable=(not self.auto_print)):
            #    if len(h) >= 4: # obj plus 3 or more for whom it is nearest.
            #        self.hubs.append(h)
            #        self.hubs[j].ID = j
            #        self.hubs[j].calculate()
            #        j += 1
            #del(temp_hubs) # To save on memory.

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

        # Supernodes:
        if "Supernodes" in self.categories and len(self.nodes) >= 2:

            # Nearest neighbor-node computation:
            self.supernodes = clusters.clusterizer.compute_supernodes(
                self.nodes, self._print, self.metric_str,
                self.metric, self.auto_print
            )
            #centroids = [n.centroid for n in self.nodes]
            #self._print("Fracturing the Empire")
            #dist_matrix = sp.distance.squareform(
            #    sp.distance.pdist(
            #        centroids,
            #        self.metric_str
            #            if self.metric_str != None else self.metric))
            #self._print("Establishing a Hierocracy")
            #neighbors = np.argmax(dist_matrix, axis=1)
            ##neighbors_dist = dist_matrix[range(len(dist_matrix)), neighbors]
            #
            ## Compute the Supernodes:
            #self.supernodes = [
            #    clusters.Node(node,
            #        self.nodes[neighbors[i]],
            #        clusters.Node.get_centroid, self.metric)
            #    for i, node in enumerate(tqdm(range(len(self.nodes)),
            #        desc="Ascertaining Universe Filaments",
            #        disable=(not self.auto_print)))
            #    if (i == neighbors[neighbors[i]]
            #        and i < neighbors[i])]

            # Supernode Length and other info:
            self._print("Measuring their Magnitude")
            self.supernode_lengths = [n.distance for n in self.supernodes]
            self._print("Minding the Macrocosm")
            self._add_info(len(self.supernodes), "Supernodes", "Count")
            if len(self.supernodes) > 0:
                node_min = np.min(self.supernode_lengths)
                node_max = np.max(self.supernode_lengths)
                self._add_info(np.mean(self.supernode_lengths),
                    "Supernodes", "Span Avg")
                self._add_info(node_min, "Supernodes", "Span Min")
                self._add_info(node_max, "Supernodes", "Span Max")
                self._add_info(node_max - node_min, "Supernodes", "Span Range")
                self._add_info(len(self.supernodes)*4.0/float(len(self.space)),
                    "Supernodes", "Island Factor", star=True)
                self._add_info(
                    len(self.supernodes)*2.0/float(len(self.nodes)),
                    "Supernodes", "Hierarchical Factor", star=True)
                self._add_info(self.supernode_lengths,
                    "Supernodes", "Span Histogram Key")

        # Nuclei:
        pass

        # Chains:
        pass

        # NCC:
        pass

        # LNCC:
        pass

        # Anti-hubs:
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
