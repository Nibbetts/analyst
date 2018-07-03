import numpy as np
from collections import OrderedDict

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

        Usage:
            Inside an evaluator, if you want to make a cluster store more or
            different kinds of data, put these data inside of self.stats_dict.
            Clusterizers automatically find meta-stats on these values.
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
        # self.centroid = []
        # self.centroid_distances = []
        # self.focus = []
        # self.medoid = None
        # self.norms = []

        self.stats_dict = OrderedDict()
        # self.centroid_norm = 0
        # self.cardinality = len(self.objects)
        # self.dispersion = 0
        # self.std_dev = 0
        # self.repulsion = 0
        # self.skew = 0
        # self.norm_range = 0

        if self.auto: self.calculate()

    def __iadd__(self, B):
        self.objects = list(set(self.objects + B.objects))
        # Both must be python lists,
        #   UNLESS USE np.hstack() or np.concatenate(),
        #   then change add_objects and add_nodes functions as well!
        self.nodes = list(set(self.nodes + B.nodes))
        # Auto-calculation is dependent on the one being added to:
        #if self.auto | B.auto: self.calculate()
        self.vectors = None
        if self.auto:
            self.calculate()

        return self

    def __add__(self, B):
        return Cluster(
            self.encoder, self.metric, nearest=self.nearest,
            vectors=None,
            objects=list(set(self.objects + B.objects)),
            nodes=list(set(self.nodes + B.nodes)),
            # Both must be auto for it to calculate:
            auto=self.auto & B.auto, ID=None, name=None, **self.metric_args)

    def __len__(self):
        # Cardinality
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]

    def __str__(self):
        max_length = 16#max(max(), max()) # TODO
        format_str = "{:<" + str(max_length + 2) + "}:"
        def line(key, value, comma=True):
            c = "," if comma else ""
            return c + "\n\t" + format_str.format(key) + str(value)

        result = "Cluster(" \
            + line("ID",         self.ID, comma=False) \
            + line("Name",       self.name) \
            + line("Medoid",     self.medoid)
        for (key, value) in self.stats_dict.items():
            if key != "Norm Range": result += line(key, value)
        result = result \
            + line("Dispersion",       np.mean(self.centroid_distances)) \
            + line("Centr Dist Min",   np.min(self.centroid_distances)) \
            + line("Centr Dist Max",   np.max(self.centroid_distances)) \
            + line("Centr Dist Range", np.max(self.centroid_distances) - # ADD CENTROID NORM???
                np.min(self.centroid_distances)) \
            + line("Centr Dist Std",   np.std(self.centroid_distances)) \
            + line("Norm Avg",   np.mean(self.norms)) \
            + line("Norm Min",   np.min(self.norms)) \
            + line("Norm Max",   np.max(self.norms)) \
            + line("Norm Range", self.stats_dict["Norm Range"]) \
            + line("Norm Std",   np.std(self.norms)) \
            + line("Nodes",      [str(node) for node in sorted(self.nodes)]) \
            + line("Objects",    [str(o) for o in sorted(self.objects)]) + "\n)"

        return result

    def __repr__(self):
        return str(self)

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

        self.stats_dict["Population"] = len(self.objects)
        if len(self.objects) > 0:
            #self.centroid = sum(self.vectors) / len(self.vectors)
            self.centroid = np.mean(self.vectors, axis=0)
            self.stats_dict["Centroid Norm"] = np.linalg.norm(self.centroid)

            # Calculate Dispersion:
            #self.dispersion = sum(self.metric(self.centroid, vec)
            #    for vec in self.vectors) / len(self.objects)
            self.centroid_distances = [self.metric(
                self.centroid, vec, **self.metric_args) for vec in self.vectors]
            self.stats_dict["Dispersion"] = np.mean(
                self.centroid_distances, axis=0)

            # Calculate Medoid:
            medoid_i = np.argmin([
                self.metric(self.centroid, v, **self.metric_args) \
                for v in self.vectors])
            self.medoid = self.objects[medoid_i]
            self.stats_dict["Medoid Dist"] = self.metric(
                self.vectors[medoid_i], self.centroid, **self.metric_args)

            # Calculate Standard Deviation:
            self.stats_dict["Standard Dev"] = np.std(self.vectors)

            #self.focus = sum(n.centroid for n in self.nodes) / len(self.nodes)
            if len(self.nodes) != 0:
                self.stats_dict["Node Count"] = len(self.nodes)
                self.focus = np.mean([n.centroid for n in self.nodes], axis=0)
                self.stats_dict["Skew"] = self.metric(
                    self.centroid, self.focus, **self.metric_args)
            else:
                self.focus = None
                # NOTE: Macro stats are run on stats_dict items, so stats_dict
                #   should NEVER include None! Simply do not add it.

            # Calculate repulsion:
            # NOTE: if objects are placed in clusters different from their
            #   nearest neighbor, this will include a few phony values.
            #if self.nearest != None: self.repulsion = sum(self.metric(
            #    v, self.encoder(self.nearest(self.objects[i])))
            #    for i, v in self.vectors) / len(self.objects)
            if self.nearest != None:
                self.stats_dict["Repulsion"] = np.mean([self.metric(
                        v, self.encoder(self.nearest(self.objects[i])),
                        **self.metric_args) \
                    for i, v in enumerate(self.vectors)], axis=0)

            # Norm Stats:
            self.norms = [np.linalg.norm(v) for v in self.vectors]
            self.stats_dict["Norm Range"] = \
                np.max(self.norms) - np.min(self.norms)


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
