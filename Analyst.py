import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import sys

import clusters


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
            dist. to nearest min, max, range, graph of distribution of.
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
            compare_with(analyst2) -- prints a full report with three numbers
                for each property instead of one - val_for_A, val_for_B, val_A-B
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

    def __init__(self, embeddings, metric="cosine_similarity",
        encoder=None, decoder=None, cluster_algorithms=[(None, "All")],
        auto_print=True, desc=None):
        """
        Parameters:
            embeddings -- list of vectors populating the space.
                Must have static indeces. (ie: not a dict or set)
            metric -- the distance metric used throughout,
                "l2" or "euclidean", "l1", "cosine_similarity",
                or a function object. Defaults to "cosine_similarity".
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
            auto_print -- whether to print reports automatically after analyses.
            desc -- optional short description/title for this analyst instance.
        """

        self.auto_print = auto_print
        print
        self._print("Asking the Grand Question")
        self._print("Stretching the Fabric of Space and Time")
        self._print("Enumerating the Dimensions")
        self.space = embeddings
        self._print("Laying the Laws of Physics")
        if callable(metric): self.metric = metric
        elif metric == "l2" or metric == "euclidean":
            self.metric = sp.distance.euclidean
        elif metric == "cosine_similarity": self.metric = sp.distance.cosine
        elif metric == "l1": self.metric = sp.distance.cityblock
        else: raise ValueError("'metric' parameter unrecognized and uncallable")
        self.description = desc

        # Encoder/Decoder Initializations:
        #   While initializing these should, in theory, be unnecessary,
        #   failing to do so will limit all inputs to findable types.
        self._print("Filling the Void")
        self.encode = encoder # string to vector
        self.decode = decoder # vector to string
        self.s_to_ix = {}
        self.ix_to_s = []
        try:
            self.ix_to_s = [self.decode(vec) for vec in tqdm(self.space,
                desc="Naming Stars and Drawing a Star Map",
                disable=(not self.auto_print))]
            for ix, s in enumerate(tqdm(self.ix_to_s, desc="Indexing Planets",
                    disable=(not self.auto_print))):
                self.s_to_ix[s] = ix
        except: print("You gone done broke it...")

        self.neighbors = np.zeros((len(self.space),3), dtype=np.uint8)
            # Indeces correspond to indeces of vectors in the space. For each:
            #   [index of nearest, index of second-nearest, index of furthest]
            #   These are filled in in the _spatial_analysis.
        self.neighbors_dist = np.zeros((len(self.space),3), dtype=np.float32)
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
        self.category_lists = np.empty(shape=(len(self.categories),0)).tolist()
        self._spatial_analysis(report_spatial)
        self._cluster_analysis()
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

        # Nearest, 2nd Nearest, and Futhest Computation:
        self._print("Ousting Nearly Empty Universes") #"Ousting the Flatlanders"
        if len(self.space) < 4:
            return
        for i, vec in enumerate(tqdm(self.space, disable=(not self.auto_print),
                                desc="Acquainting the Species")):
            nearest_i = (0 if i != 0 else 1) # Can't start off on self!
            nearest_2i = (2 if i != 2 else 3) # Can't start off same as nearest!
            furthest_i = i # Start off closest possible - self.
            nearest_dist = self.metric(vec, self.space[nearest_i])
            nearest_2dist = self.metric(vec, self.space[nearest_2i])
            furthest_dist = self.metric(vec, self.space[furthest_i])
            # In case we started them off switched:
            if nearest_2dist < nearest_dist:
                temp_i = nearest_i
                temp_dist = nearest_dist
                nearest_i = nearest_2i
                nearest_2i = temp_i
                #furthest_i = temp_i
                nearest_dist = nearest_2dist
                nearest_2dist = temp_dist
                #furthest_dist = temp_dist
            for j, other in enumerate(self.space):
                if j != i:
                    dist = self.metric(vec, other)
                    if dist < nearest_dist:
                        nearest_2dist = nearest_dist
                        nearest_2i = nearest_i
                        nearest_dist = dist
                        nearest_i = j
                    elif dist < nearest_2dist and j != nearest_i:
                        nearest_2dist = dist
                        nearest_2i = j
                    if dist > furthest_dist:
                        furthest_dist = dist
                        furthest_i = j
            self.neighbors[i][0] = nearest_i
            self.neighbors[i][1] = nearest_2i
            self.neighbors[i][2] = furthest_i
            self.neighbors_dist[i][0] = nearest_dist
            self.neighbors_dist[i][1] = nearest_2dist
            self.neighbors_dist[i][2] = furthest_dist

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

        # Nearest Neighbor Info:
        self._print("Building Trade Routes")
        self.nearest_avg = np.mean(self.neighbors_dist[:,0])
        self._add_info(self.nearest_avg,
            "Spatial", "Remoteness - Nearest Dist Avg", star=True)
        self._print("Practicing Diplomacy")
        nearest_min = np.min(self.neighbors_dist[:,0])
        nearest_max = np.max(self.neighbors_dist[:,0])
        self._add_info(nearest_min, "Spatial", "Nearest Dist Min")
        self._add_info(nearest_max, "Spatial", "Nearest Dist Max")
        self._add_info(nearest_max-nearest_min, "Spatial", "Nearest Dist Range")
        self._add_info(self.neighbors_dist[:,0],
            "Spatial", "Nearest Dist Histogram Key")

        #Furthest Neighbor Info:
        self._print("Making Enemies")
        self.furthest_avg = np.mean(self.neighbors_dist[:,2])
        self._add_info(self.furthest_avg, "Spatial", "Furthest Dist Avg")
        self._print("Claiming Frontiers")
        furthest_min = np.min(self.neighbors_dist[:,2])
        furthest_max = np.max(self.neighbors_dist[:,2])
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
            if print_node_info:
                node_min = np.min(self.node_lengths)
                node_max = np.max(self.node_lengths)
                self._add_info(len(self.nodes), "Nodes", "Count")
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
            hub_sizes = [len(h) for h in self.hubs]
            hub_min = np.min(hub_sizes)
            hub_max = np.max(hub_sizes)
            self._add_info(np.mean(hub_sizes), "Hubs", "Population Avg")
            self._add_info(hub_min, "Hubs", "Population Min")
            self._add_info(hub_max, "Hubs", "Population Max")
            self._add_info(hub_max-hub_min, "Hubs", "Population Range")
            self._add_info(hub_sizes, "Hubs", "Population Histogram Key")

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


    #--------------------------------------------------------------------------#
    # Information Gathering and Reporting Functions                            #
    #--------------------------------------------------------------------------#

    def _add_info(self, var, category, description, star=False):
        # Description and category must be strings.
        #variable = None
        #i = None
        if "Histogram" in description:
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
            (description, variable, "*" if star else " "))

    def _print(self, string=""):
        if self.auto_print: print("\r" + string + "...")

    def print_report(self):
        self._print("Revealing the Grand Plan")
        print
        if self.description != None: print(self.description.upper())
        for i, category in enumerate(self.categories):
            print(category + ": ")
            for cat in self.category_lists[i]:
                #print("\t" + str(cat[1]) + "\t" + str(cat[0]))
                #print(cat[0],cat[1],sep="\t") #python3
                print("  {} {:<16} {}{}".format(cat[2],cat[1],cat[2],cat[0]))


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
        if f_name[-7:] == ".pickle": return f_name
        if f_name[-4:] == ".pck": return f_name
        if f_name[-4:] == ".pcl": return f_name
        if f_name[-3:] == ".db": return f_name
        return f_name if f_name[-7:] == ".pkl" else f_name + ".pkl"

    @staticmethod
    def save(obj, f_name):
        try:
            with open(Analyst._file_extension(f_name), 'wb') as file:
                pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
            return True
        except:
            return False

    @staticmethod
    def load(f_name):
        try:
            with open(Analyst._file_extension(f_name), 'rb') as file:
                return pickle.load(file)
        except:
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



# Brief script-like behavior for development, debugging, and testing purposes.
if __name__ == "__main__":

    import TestSet2D

    t = TestSet2D.TestSet2D()
    a = Analyst(t, "euclidean", t.encode, t.decode, desc="2D Test Set")
