import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt

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
        node: a pair of objects in the space whose nearest neighbors are each other.
        supernode: a pair of nodes whose nearest neighbor-nodes are each other.
        extremity: a pair of objects in the space whose furthest neighbors are each other.
            (implemented with the same Node class as nodes are.)
        outlier: an object which is a member of an extremity.
        loner: an object which has been rejected when forming clusters,
            making it a cluster unto itself, of sorts.
        hub: an object that is the nearest neigbor of three or more other objects.
        nodal factor: ratio of words belonging to nodes;
            a measure of the scale or impact of relationships in the space
        hierarchical factor: ratio of nodes belonging to supernodes;
            a further measure of relationships in the space.
        island factor: ratio of objects belonging to supernodes;
            another measure of relationships in the space.
        nucleus: grouping of relatively nearby objects. Starting with nodes and all
            objects whose nearest are one of those, then finding average distance
            to center, then combining all clusters whose central nodes are closer than
            one of their averages.
        chain: different clustering where each cluster has a sole node and recursively
            finds all whose nearest's nearest's etc. nearest is a member of that node.
        cluster: Find nuclei, then add chainable objects whose nearest is closer
            than the average distance of those whose nearest are in one of the nodes,
            (those we started with only) and whose nearest is in the cluster.
        strong cluster: Same as cluster, but builds nuclei and clusters requiring
            objects to have both first and second nearest belonging to the same
            grouping in order for them to be added.  
        center: average location of all objects in a cluster.
        string factor: average cluster span divided by space dispersion
        regularity: find average cluster pop, then find average difference btw.
            cluster pop and average cluster pop. Regularity is 1/(1+this)
        density: average distance of nodes to their nearest neighbor.
        dispersion: average distance of nodes to the center of their distribution
        focus: averaged location of nodes in a cluster; concentration center.
        skew: distance from focus to center.
        anti-cluster: list of objects whose furthest are all the same outlier.


    Tools:
        NOTE: Those in parentheses cannot be used as measures of the properties
            of the embedding space since they depend directly on the number of
            embeddings, dimensionality, or some similar property.
            These could be seen as specific information.
        NOTE: Those properties with a * are of most import in measuring a space.

        Spatial:
            * centroid
            medoid

            * dispersion

            * density -- avg dist. to nearest --> graph of distr. of dist. to nearest
            min, max dist. to nearest  -------/
            range of distances to nearest  __/

            * (num extremities -- probably depends strongly on dimensionality,
                but shows the spherical-ness of the distribution in the space)
            * max, min extremity length  ---->  graph of distr. of extremity lengths
            avg extr. length  ------------/
            extr. length range  _________/
            NOTE: all extremity measurements may be estimates from sampling.

        Clustering:
            NOTE: For each property of a cluster type, the following stats are available:
                avg, min, max, range, distribution graph of.

            Nodes:
                (num nodes)
                * nodal factor
                node length stats
                * alignment factor

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
                nucleus density factor -- avg. nucleus density / space density,
                    (nucleus density stats)
                nucleus skew factor, (nucleus skew stats)

            Chains:
                chain population stats
                chain string factor -- avg. chain span / space dispersion,
                    (chain span stats)
                chain dispersion factor -- avg. chain disp. / space disp,
                    (chain dispersion stats)
                chain density factor -- avg. chain density / space density,
                    (chain density stats)
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
                * cluster density factor -- avg. cluster density / space density,
                    (cluster density stats)
                * cluster skew factor -- avg. cluster skew / space dispersion,
                    (cluster skew stats)

            Strong Clusters:
                Same info as for clusters.
            
            Anti-clusters:
                More or less the same information as for clusters,
                but it WILL not mean the same things. Note that these clusters
                DO NOT include the word that is their farthest neighbor.

        Specifics / Inspection:
            rank_outliers() -- by number of obj. for which this one is furthest neighbor.
                Resulting list contains exactly all objects which are members of an extremity.
            rank_clusters() -- by size; lists the indeces of the clusters.
            rank_hubs() -- by number of obj. for which this one is nearest neighbor.
            * clusters -- accessible variable; a list of the clusters. Further info in each.
            * strong clusters -- ''
            nodes -- ''
            supernodes -- ''
            nuclei -- ''
            chains -- ''
            extremities -- ''
            anti-clusters -- dictionary keyed to outlier objects, containing anti-clusters

        Simulation:
            simulate_space() -- Generates an entire fake embedding space with specified properties,
                and returns it wrapped in a new analyst object.
                NOTE: Includes cluster generation. No need to add to it.
            simulate_cluster() -- Generates generic test clusters to compare with,
                or to examine properties. Types listed in function comments.

        Analogical:
            run_analogies() !!!CANT--UNKNOWN OUTPUT??!!!
            member_of(object) -- displays cluster this object is a member of.
            cluster([list of objects]) -- a new cluster composed solely of the given objects.
            seeded_cluster([list of objects]) -- a new cluster composed of all nearby objects
                likely to be clustered with these, if these were treated as being together.
            inspect_clustering([list of objects]) -- analysis on given objects, returns:
                - number of unique clusters these words are found across
                - average ward dissimilarity of involved clusters
                - list of tuples containing: (object, cluster_index)
            circular_walk_graph(obj1, obj2) -- most useful in a normalized space, like word2vec.
    """

    def __init__(self, embeddings, metric="cosine_similarity",
        encoder=None, decoder=None, auto_print=True):
        """
        Parameters:
            embeddings -- list of vectors populating the space.
                Must have static indeces. (ie: not a dict or set)
            metric -- the distance metric used throughout,
                "l2" or "euclidean", "l1", "cosine_similarity",
                or a function object. Defaults to "cosine_similarity".
            encoder -- a callable to convert strings to vectors.
            decoder -- a callable to convert vectors to strings.
            auto -- whether to run analyses and print automatically.
        """
        self.space = embeddings
        if callable(metric): self.metric = metric
        elif metric == "l2" or metric == "euclidean":
            self.metric = sp.distance.euclidean
        elif metric == "cosine_similarity": self.metric = sp.distance.cosine
        elif metric == "l1": self.metric = sp.distance.cityblock
        else: raise ValueError("'metric' parameter unrecognized and uncallable")

        # Encoder/Decoder Initializations:
        #   While initializing these should, in theory, be unnecessary,
        #   failing to do so will limit all inputs to findable types.
        self.encode = encoder # string to vector
        self.decode = decoder # vector to string
        self.s_to_ix = {}
        self.ix_to_s = []
        try:
            self.ix_to_s = [self.decode(vec) for vec in self.space]
            for ix, s in enumerate(self.ix_to_s):
                self.s_to_ix[s] = ix
        except: pass

        # Run Analyses:
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


    # General Analyses:
    def _spatial_analysis(self):
        self.centroid = np.mean(self.space, axis=0)
        self.medoid = self.as_string(np.argmin([
            self.metric(self.centroid, v) for v in self.space])]
        self.dispersion = np.mean([self.metric(self.centroid, v)
            for v in self.space], axis=0)
        #NEAREST AND FURTHEST HERE!
        self.density = ........

    def _cluster_analysis(self):
        pass

    def print_report(self):
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


    # Simulation:
    @classmethod
    def simulate_space(cls, parameters):
        '''
        parameters:
            A list of lists, each of which follows the format:
                ["space_type", "cluster_type", num_clusters, space_radius, space_dims
                    (cluster_min_pop, cluster_max_pop),
                    (cluster_min_radius, cluster_max_radius),
                    cluster_occupied_dims, cluster_total_dims, randomize_dims,
                    noise, normalize]

                Types: (used for both cluster and space)
                    "shell" (circle if occupied_dims==2)
                    "ball"
                    "radial" (like ball only random dir, radius instead of x,y,z,...
                        concentrated in center)
                    "cube" (random x,y,z,... but in plane or hypercube instead of ball)
                    "even" (attempts amorphous semi-uniformity of distances btw. points)
                    "grid" (attempts a gridlike uniformity)
                    "pairs" (generates points in pairs of close proximity -- forces
                        excessive node generation)
                    "line" (generates points in lines)
                    "snake" (generate points in curvy lines)
                    "oval" (like radial, but randomly varies size of each axis within
                        allowed radius sizes)
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
            A list of the clusters used to create the space before clustering was
                recalculated, for comparison. This will be different if
                clusters overlapped.
        '''
        pass
        #note, need to make it create a generic identity function for encode/decode.
        #   or use indeces.

    @staticmethod
    def simulate_cluster(type, population, radius, occupied_dims,
        total_dims, randomize_dims=True, noise=0, normalize=False):
        # Same usage as the cluster parameters in simulate_space().
        # NOTE: when this function is called by simulate_space(), normalize
        #   is never True here. That would be done after the fact,
        #   on the whole simulated space, not on any one cluster.
        pass


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