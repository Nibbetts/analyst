import node

class Cluster:

    def __init__(self, encoder, metric, nearest=None, objects=[], nodes=[], auto=False):
        """
        Parameters:
            encoder: callable; gets one vector from one object at a time,
                in the form of a NUMPY ARRAY, or sum() won't work.
            metric:  callable; distance metric between multidimensional vectors.
            nearest: callable; takes object and returns its nearest neighbor object.
                If unavailable, the only difference is density will be unavailable
            objects: a list of unencoded objects in an embedding space,
                including those which are members of the given nodes.
            nodes: a list of Nodes, which each contain a pair of objects and have a centroid.
            auto: if true, calculates automatically after creation and after each addition.
        """
        self.objects = objects
        self.encoder = encoder
        self.metric = metric
        self.nearest = nearest
        self.nodes = nodes
        self.auto = auto

        # self.vectors = []
        # self.centroid = []
        # self.dispersion = 0
        # self.density = 0
        # self.focus = []
        # self.skew = 0
        if self.auto: self.calculate()

    def __iadd__(self, B):
        self.objects += B.objects # Both must be python lists
        self.nodes += B.nodes
        if self.auto: self.calculate()

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]

    def __str__(self):
        # Cardinality
        result = "Cluster("
        if len(self.objects) > 0:
            result += str(self.objects[0])
            for obj in self.objects[1:]:
                result += ", " + str(obj)
        return result + ")"

    def add_objects(self, obj_list):
        self.objects += obj_list

    def add_nodes(self, obj_list, node_list):
        # If adding nodes, must also add the objects belonging to them,
        #   since from here finding recursively could be complicated.
        self.objects += obj_list
        self.nodes += node_list

    def calculate(self):
        self.vectors = map(self.encoder, self.objects)
        self.centroid = sum(self.vectors) / len(self.vectors)
        self.dispersion = sum([self.metric(self.centroid, vec)
            for vec in self.vectors]) / len(self.objects)
        if self.nearest != None: self.density = sum([self.metric(
            v,
            self.encoder(self.nearest(self.objects[i])))
            for i, v in self.vectors]) / len(self.objects)
        else: self.density = None
            # NOTE: if objects are placed in clusters different from their
            #   nearest neighbor, this will include a few phony values.
        self.focus = sum([n.centroid for n in self.nodes]) / len(self.nodes)
        self.skew = self.metric(self.centroid, self.focus)

    def cluster_dist(self, B):
        return self.metric(self.centroid, B.centroid)

    def ward_dissimilarity(self, B):
        # In some sense a cost measurement of merging two clusters,
        #   from Ward's Method of agglomerative clustering.
        return ((self.metric(self.centroid, B.centroid)**2.0
                / (1.0/len(self) + 1.0/len(B))))

    #def get_centroid(self, cluster_name):
    #    return self.centroid

    #def get_dispersion(self, cluster_name):
    #    return self.dispersion

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