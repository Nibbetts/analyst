import node

class Cluster:

    def __init__(self, encoder, metric, objects=[], nodes=[], auto=False):
        """
        Parameters:
            encoder: callable; gets one vector from one object at a time,
                in the form of a NUMPY ARRAY, or sum() won't work.
            metric:  callable; distance metric between multidimensional vectors.
            objects: a list of unencoded objects in an embedding space.
            nodes: a list of Nodes, which each contain a pair of objects and have a center.
            auto: if true, calculates automatically after creation and after each addition.
        """
        self.objects = objects
        self.encoder = encoder
        self.metric = metric
        self.nodes = nodes
        self.auto = auto

        # self.vectors = []
        # self.center = []
        # self.dispersion = 0
        if self.auto: self.calculate()

    def __iadd__(self, objects=[], nodes=[]):
        self.objects += objects # Both must be python lists
        self.nodes += nodes
        if self.auto: self.calculate()

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]

    def calculate(self):
        self.vectors = map(self.encoder, self.objects)
        self.center = sum(self.vectors) / len(self.vectors)
        self.dispersion = sum([self.metric(self.center, vec)
            for vec in self.vectors]) / len(self.objects)
        self.focus = sum([n.center for n in self.nodes]) / len(self.nodes)
        self.skew = self.metric(self.center, self.focus)


    #def get_center(self, cluster_name):
    #    return self.center

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