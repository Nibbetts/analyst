import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

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


    Definitions:
        node: a pair of obj in the space whose nearest neighbors are each other.
        supernode: a pair of nodes whose nearest neighbor-nodes are each other.
        extremity: a pair of objects in the space whose furthest neighbors are
            each other. (implemented with the same Node class as nodes are.)
        outlier: an object which is a member of an extremity.
        loner: an object which has been rejected when forming clusters,
            making it a cluster unto itself, of sorts.
        hub: an obj that is the nearest neigbor of three or more other objects. //DO SOMETHING!!
        nodal factor: ratio of words belonging to nodes;
            a measure of the scale or impact of relationships in the space
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
        proximity: as used here, 1 / avg distance of nodes to their nearest
            neighbors. This is not a measure of spatial or volumentric density,
            but is a metric for relationships between encoded objects,
            and thus could potentially be used as a density metric.
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
            * centroid
            medoid

            * dispersion

            * proximity -- 1 / avg dist to nearest
            dist. to nearest avg, min, max, range, graph of distribution of.

            Extremities:
                * (num extremities -- probably depends strongly on
                    dimensionality, but shows the spherical-ness of the
                    distribution in the space)
                extremity length avg, * min, * max, range, graph of distr.

            Nodes:
                (num nodes)
                * nodal factor
                node length avg, min, max, range, graph of distr.
                * alignment factor

        Clustering:
            NOTE: For each property of a cluster type, available stats are:
                avg, min, max, range, distribution graph of.

            Hubs:
                (num hubs)
                hub num stats

            Supernodes:
                (num supernodes)
                * hierarchical factor, burst factor
                * island factor
                supernode length stats

            Nuclei:
                (num nuclei)
                * nucleus factor (num nuclei divided by num objects)
                ratio in nuclei versus not
                nucleus population stats
                nuclei string factor, nucleus span stats
                nucleus regularity
                nucleus dispersion factor -- avg. nucleus disp. / space disp,
                    (nucleus dispersion stats)
                node count stats
                nucleus proximity factor -- avg. nucleus proximity divided by
                    overall space proximity,
                    (nucleus proximity stats)
                nucleus skew factor, (nucleus skew stats)

            Chains:
                chain population stats
                chain string factor -- avg. chain span / space dispersion,
                    (chain span stats)
                chain dispersion factor -- avg. chain disp. / space disp,
                    (chain dispersion stats)
                chain proximity factor -- avg. chain proximity / overall space
                    proximity, (chain proximity stats)
                chain skew factor, (chain skew stats)
                NOTE: num chains is equal to num nodes
                NOTE: all objects in the space belong to a chain

            Clusters:
                (num clusters)
                * cluster factor -- num clusters divided by num objects
                * string factor -- avg. cluster span / space dispersion
                * regularity, (cluster span stats)
                * ratio clustered versus loners
                * avg cluster population, cluster population stats
                * cluster dispersion factor -- avg. cluster disp. / space disp,
                    (cluster dispersion stats)
                * avg num nodes per cluster, node count stats
                * cluster proximity factor -- avg cluster proximity / overall
                    space proximity, (cluster proximity stats)
                * cluster skew factor -- avg. cluster skew / space dispersion,
                    (cluster skew stats)

            Strong Clusters:
                Same info as for clusters.

            Anti-clusters:
                More or less the same information as for clusters,
                but it WILL not mean the same things. Note that these clusters
                DO NOT include the word that is their farthest neighbor.

        Specifics / Inspection:
            rank_outliers() -- by number of obj. for which this one is furthest
                neighbor. Resulting list contains exactly all objects which are
                members of an extremity.
            rank_clusters() -- by size; lists the indeces of the clusters.
            rank_hubs() -- by num of obj for which this one is nearest neighbor.
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

        Simulation:
            Analyst.simulate_space() -- @staticmethod which generates an entire
                fake embedding space with the specified properties,
                and returns it wrapped in a new analyst.
                NOTE: Includes cluster generation. No need to add to it.
            Analyst.simulate_cluster() -- @staticmethod which generates generic
                test clusters to compare with, or to examine properties.
                Available cluster types listed in function comments.

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
    """

    def __init__(self, embeddings, metric="cosine_similarity",
        encoder=None, decoder=None, auto_print=True, desc=None):
        """
        Parameters:
            embeddings -- list of vectors populating the space.
                Must have static indeces. (ie: not a dict or set)
            metric -- the distance metric used throughout,
                "l2" or "euclidean", "l1", "cosine_similarity",
                or a function object. Defaults to "cosine_similarity".
            encoder -- a callable to convert strings to vectors.
            decoder -- a callable to convert vectors to strings.
            auto_print -- whether to print reports automatically after analyses.
            desc -- optional short description/title for this analyst instance.
        """

        self.auto_print = auto_print
        self._print()
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
        except: pass

        self.neighbors = np.zeros((len(self.space),3), dtype=np.uint8)
            # Indeces correspond to indeces of vectors in the space. For each:
            #   [index of nearest, index of second-nearest, index of furthest]
            #   These are filled in in the _spatial_analysis.
        self.neighbors_dist = np.zeros((len(self.space),3), dtype=np.float32)
            # Same format as above, except distances to those indexed above.

        # Run Analyses:
        self.categories = []
        self.category_lists = []
        self._spatial_analysis()
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


    # General Analyses:

    def _spatial_analysis(self):
        
        # Nearest and Futhest:
        self._print("Ousting Empty Universes") #"Ousting the Flatlanders"
        if len(self.space) < 3:
            return
        for i, vec in enumerate(tqdm(self.space, disable=(not self.auto_print),
                                desc="Acquainting the Species")):
            nearest_i = (0 if i != 0 else 1)
            nearest_2i = (1 if i != 1 else 2)
            furthest_i = (1 if i != 1 else 2)
            nearest_dist = self.metric(vec, self.space[nearest_i])
            nearest_2dist = self.metric(vec, self.space[nearest_2i])
            furthest_dist = self.metric(vec, self.space[furthest_i])
            if nearest_2dist < nearest_dist:
                temp_i = nearest_i
                temp_dist = nearest_dist
                nearest_i = nearest_2i
                nearest_2i = temp_i
                furthest_i = temp_i
                nearest_dist = nearest_2dist
                nearest_2dist = temp_dist
                furthest_dist = temp_dist
            for j, other in enumerate(self.space):
                if j != i:
                    dist = self.metric(vec, other)
                    if dist < nearest_dist:
                        nearest_2dist = nearest_dist
                        nearest_2i = nearest_i
                        nearest_dist = dist
                        nearest_i = j
                    if dist > furthest_dist:
                        furthest_dist = dist
                        furthest_i = j
            self.neighbors[i][0] = nearest_i
            self.neighbors[i][1] = nearest_2i
            self.neighbors[i][2] = furthest_i
            self.neighbors_dist[i][0] = nearest_dist
            self.neighbors_dist[i][1] = nearest_2dist
            self.neighbors_dist[i][2] = furthest_dist

        # Measurements:
        self._print("Balancing the Continuum")
        self.centroid = np.mean(self.space, axis=0)
        #self._add_info(self.centroid,
        #    "Spatial", "Centroid - Coordinate Avg")
        self._add_info(self.as_string(np.argmin([
            self.metric(self.centroid, v) for v in tqdm(self.space,
                desc="Electing a Ruler", disable=(not self.auto_print))])),
            "Spatial", "Medoid - Obj Nearest to Centroid")
        self.centroid_dist = [self.metric(self.centroid, v)
            for v in tqdm(self.space, desc="Counting the Lightyears",
                disable=(not self.auto_print))]
        self._add_info(np.mean(self.centroid_dist, axis=0),
            "Spatial", "Dispersion - Centroid Dist Avg")
        centr_min = np.min(self.centroid_dist, axis=0)
        centr_max = np.max(self.centroid_dist, axis=0)
        self._add_info(centr_min, "Spatial", "             Centroid Dist Min")
        self._add_info(centr_max, "Spatial", "             Centroid Dist Max")
        self._add_info(centr_max - centr_min,
            "Spatial", "             Centroid Dist Range")
        #self.proximity = np.mean(
        #    [self.metric(v, self.encoder(self.nearest(self.objects[i])))
        #     for i, v in self.vectors])
        self._print("Building Trade Routes")
        nearest_avg = np.mean(self.neighbors_dist[:,0])
        self._add_info(1.0 / nearest_avg, "Spatial", "Proximity")
        self._add_info(nearest_avg, "Spatial", "Nearest Dist Avg")
        self._print("Practicing Diplomacy")
        nearest_min = np.min(self.neighbors_dist[:,0])
        nearest_max = np.max(self.neighbors_dist[:,0])
        self._add_info(nearest_min, "Spatial", "Nearest Dist Min")
        self._add_info(nearest_max, "Spatial", "Nearest Dist Max")
        self._add_info(nearest_max-nearest_min, "Spatial", "Nearest Dist Range")

        # Extremities:
        self.extremities = [
            clusters.Node(self.as_string(i),
                self.as_string(self.neighbors[i][2]),
                self.encode, self.metric)
            for i in tqdm(range(len(self.space)),
                desc="Measuring the Reaches",
                disable=(not self.auto_print))
            if (i == self.neighbors[self.neighbors[i][2]][2]
                and i < self.neighbors[i][2])]
        self.extremity_lengths = [e.distance for e in tqdm(self.extremities,
            desc="Setting the scopes",
            disable=(not self.auto_print))]
        self._print("Puzzling Over the Star Charts")
        self._add_info(len(self.extremities), "Spatial", "Extremity Count")
        extr_min = np.min(self.extremity_lengths)
        extr_max = np.max(self.extremity_lengths)
        self._add_info(np.mean(self.extremity_lengths),
            "Spatial", "Extremity Span Avg")
        self._add_info(extr_min, "Spatial", "Extremity Span Min")
        self._add_info(extr_max, "Spatial", "Extremity Span Max")
        self._add_info(extr_max - extr_min, "Spatial", "Extremity Span Range")
        
        # Nodes:
        self.nodes = [
            clusters.Node(self.as_string(i),
                self.as_string(self.neighbors[i][0]),
                self.encode, self.metric)
            for i in tqdm(range(len(self.space)),
                desc="Watching the Galaxies Coelesce",
                disable=(not self.auto_print))
            if (i == self.neighbors[self.neighbors[i][0]][0]
                and i < self.neighbors[i][0])]
        self.node_lengths = [n.distance for n in tqdm(self.nodes,
            desc="Delineating the Quasars",
            disable=(not self.auto_print))]
        self._print("Comparing the Cosmos")
        self._add_info(len(self.nodes), "Spatial", "Node Count")
        node_min = np.min(self.node_lengths)
        node_max = np.max(self.node_lengths)
        self._add_info(np.mean(self.node_lengths), "Spatial", "Node Span Avg")
        self._add_info(node_min, "Spatial", "Node Span Min")
        self._add_info(node_max, "Spatial", "Node Span Max")
        self._add_info(node_max - node_min, "Spatial", "Node Span Range")
        self._add_info(len(self.nodes)*2.0/float(len(self.space)),
            "Spatial", "Nodal Factor")
        ###self._add_info(???, "Spatial", "Node Alignment Factor") //HOW??
        ###???HUBS???
        
        
    def _cluster_analysis(self):
        pass


    def _add_info(self, var, category, description):
        # Description and category must be strings.
        try:
            i = self.categories.index(category)
        except:
            i = len(self.categories)
            self.categories.append(category)
            self.category_lists.append([])
        self.category_lists[i].append((description, var))

    def _print(self, string=""):
        if self.auto_print: print(string)

    def print_report(self):
        self._print("Revealing the Grand Plan")
        if self.description != None: print(self.description.upper())
        for i, category in enumerate(self.categories):
            print(category + ": ")
            for cat in self.category_lists[i]:
                #print("\t" + str(cat[1]) + "\t" + str(cat[0]))
                #print(cat[0],cat[1],sep="\t") #python3
                print("    {}\t{}".format(cat[1],cat[0]))


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


    # Simulation:
    @classmethod
    def simulate_space(cls, parameters):
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
                    "pairs" (generates points in pairs of close proximity --
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


    # General:
    @staticmethod
    def _file_extension(f_name):
        return f_name if f_name[-7:] == ".pickle" else f_name + ".pickle"

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
