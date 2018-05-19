import numpy as np
#import node

class Cluster:

    def __init__(self, encoder, metric, objects, nearest=None, vectors=None,
            nodes=[], auto=False, ID=None, name=None, **metric_args):
        """
        Parameters:
            encoder: callable; gets one vector from one object at a time.
            metric:  callable; distance metric between multidimensional vectors.
            nearest: callable; takes obj and returns its nearest neighbor obj.
                If unavailable, only difference is density will be unavailable.
            objects: a list of unencoded objects in an embedding space,
                including those which are members of the given nodes.
            vectors: a list of vectors from which to build a cluster.
                Both objects and vectors may be given, but must match,
                as no checks will be made to enforce this, creating problems.
                It is simply for saving on computation time in certain cases.
            nodes: a list of Nodes, which each contains a pair of objects and
                has a centroid.
            auto: if true, calculates automatically after creation and after
                each addition.
            id: only for convenience in indexing clusters when printed or
                grouped externally.
            name: only for convenience in identifying clusters externally.
            metric_args: additional arguments to the metric function.
                Use like **kwargs is used.
        """
        self.ID = ID
        self.name = name
        self.objects = objects
        self.encoder = encoder
        self.metric = metric
        self.nearest = nearest
        self.vectors = vectors
        self.nodes = nodes
        self.auto = auto
        self.metric_args = metric_args

        # self.vectors = []
        # self.centroid = []
        # self.centroid_length = 0
        # self.centroid_distances = []
        # self.dispersion = 0
        # self.std_dev = 0
        # self.repulsion = 0
        # self.focus = []
        # self.skew = 0
        # self.medoid = None

        if self.auto: self.calculate()

    def __iadd__(self, B):
        self.objects += B.objects # Both must be python lists,
        #   UNLESS USE np.hstack() or np.concatenate(),
        #   then change add_objects and add_nodes functions as well!
        self.nodes += B.nodes
        if self.auto: self.calculate()

    def __add__(self, B):
        return Cluster(
            self.encoder, self.metric, nearest=self.nearest,
            vectors=np.vstack((self.vectors, B.vectors)),
            objects=self.objects + B.objects, nodes=self.nodes + B.nodes,
            auto=self.auto & B.auto, ID=None, name=None, **self.metric_args)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]

    def __str__(self):
        # Cardinality
        return "Cluster(" \
            +  "\n\tID:            " + str(self.ID) \
            + ",\n\tname:          " + str(self.name) \
            + ",\n\tmedoid:        " + str(self.medoid) \
            + ",\n\tcentroid norm: " + str(self.centroid_length) \
            + ",\n\tcardinality:   " + str(len(self.objects)) \
            + ",\n\tdispersion:    " + str(self.dispersion) \
            + ",\n\tstdandard dev: " + str(self.std_dev) \
            + ",\n\trepulsion:     " + str(self.repulsion) \
            + ",\n\tskew:          " + str(self.skew) \
            + ",\n\tnodes:         " + str([str(node) for node in self.nodes]) \
            + ",\n\tobjects:       " + str([str(o) for o in self.objects]) +" )"
        #if len(self.objects) > 0:
        #    result += str(self.objects[0])
        #    for obj in self.objects[1:]:
        #        result += ", " + str(obj)
        #return result + " )"

    def add_objects(self, obj_list):
        self.objects += obj_list

    def add_nodes(self, obj_list, node_list):
        # If adding nodes, must also add the objects belonging to them,
        #   since from here finding recursively could be complicated.
        self.objects += obj_list
        self.nodes += node_list

    def calculate(self):
        if self.vectors is None or self.vectors is []:
            self.vectors = np.array([self.encoder(o) for o in self.objects])

        #self.centroid = sum(self.vectors) / len(self.vectors)
        self.centroid = np.mean(self.vectors, axis=0)
        self.centroid_length = np.linalg.norm(self.centroid)
        #self.focus = sum(n.centroid for n in self.nodes) / len(self.nodes)
        if len(self.nodes) != 0:
            self.focus = np.mean([n.centroid for n in self.nodes], axis=0)
            self.skew = self.metric(
                self.centroid, self.focus, **self.metric_args)
        else:
            self.focus = None
            self.skew = None

        # Calculate Dispersion:
        #self.dispersion = sum(self.metric(self.centroid, vec)
        #    for vec in self.vectors) / len(self.objects)
        self.centroid_distances = [self.metric(
                self.centroid, vec, **self.metric_args) for vec in self.vectors]
        self.dispersion = np.mean(self.centroid_distances, axis=0)

        # Calculate Standard Deviation:
        self.std_dev = np.std(self.vectors)

        # Calculate repulsion:
        #if self.nearest != None: self.repulsion = sum(self.metric(
        #    v, self.encoder(self.nearest(self.objects[i])))
        #    for i, v in self.vectors) / len(self.objects)
        if self.nearest != None:
            self.repulsion = np.mean([self.metric(
                    v, self.encoder(self.nearest(self.objects[i])),
                    **self.metric_args) \
                for i, v in enumerate(self.vectors)], axis=0)
        else: self.repulsion = None
            # NOTE: if objects are placed in clusters different from their
            #   nearest neighbor, this will include a few phony values.

        # Calculate Medoid:
        self.medoid = self.objects[np.argmin(
            self.metric(self.centroid, v, **self.metric_args) \
            for v in self.vectors)]
        self.medoid_dist = self.metric(
            self.medoid, self.centroid, **self.metric_args)


    def cluster_dist(self, B):
        return self.metric(self.centroid, B.centroid, **self.metric_args)

    def ward_dissimilarity(self, B):
        # In some sense a cost measurement of merging two clusters,
        #   from Ward's Method of agglomerative clustering.
        return ((self.metric(self.centroid, B.centroid, **self.metric_args)**2.0
                / (1.0/len(self) + 1.0/len(B))))

    #def _serialize(self): # For ability to pickle.
    #    # Note: Nodes do not keep the functions passed in, so only clusters
    #    #   need to be serialized.
    #    self.metric = None
    #    self.encoder = None
    #    self.decoder = None

    #def _deserialize(self, metric, encoder, decoder):
    #    self.metric = metric
    #    self.encoder = encoder
    #    self.decoder = decoder

    #def get_centroid(self, cluster_name):
    #    return self.centroid

    #def get_dispersion(self, cluster_name):
    #    return self.dispersion
