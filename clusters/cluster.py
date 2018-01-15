import numpy as np
import node

class Cluster:

    def __init__(self, encoder, metric, nearest=None,
        objects=[], nodes=[], auto=False, ID=None):
        """
        Parameters:
            encoder: callable; gets one vector from one object at a time.
            metric:  callable; distance metric between multidimensional vectors.
            nearest: callable; takes object and returns its nearest neighbor object.
                If unavailable, the only difference is density will be unavailable
            objects: a list of unencoded objects in an embedding space,
                including those which are members of the given nodes.
            nodes: a list of Nodes, which each contain a pair of objects and have a centroid.
            auto: if true, calculates automatically after creation and after each addition.
            id: only for convenience in identifying clusters when printed or grouped externally.
        """
        self.ID = ID
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
        # self.medoid = None
        if self.auto: self.calculate()

    def __iadd__(self, B):
        self.objects += B.objects # Both must be python lists,
        #   UNLESS USE np.hstack() or np.concatenate(),
        #   then change add_objects and add_nodes functions as well!
        self.nodes += B.nodes
        if self.auto: self.calculate()

    def __add__(self, B):
        return Cluster(self.encoder, self.metric, self.nearest,
            self.objects + B.objects, self.nodes + B.nodes, self.auto, None)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]

    def __str__(self):
        # Cardinality
        result = "Cluster( ID: " + str(self.ID) \
            + ", medoid: " + str(self.medoid) \
            + ", cardinality: " + len(self.objects) \
            + ", dispersion: " + str(self.dispersion) \
            + ", density: " + str(self.density) \
            + ", skew: " + str(self.skew) \
            + ", " + str(self.nodes) + " | "
        if len(self.objects) > 0:
            result += str(self.objects[0])
            for obj in self.objects[1:]:
                result += ", " + str(obj)
        return result + " )"

    def add_objects(self, obj_list):
        self.objects += obj_list

    def add_nodes(self, obj_list, node_list):
        # If adding nodes, must also add the objects belonging to them,
        #   since from here finding recursively could be complicated.
        self.objects += obj_list
        self.nodes += node_list

    def calculate(self):
        self.vectors = map(self.encoder, self.objects)
        #self.centroid = sum(self.vectors) / len(self.vectors)
        self.centroid = np.mean(self.vectors, axis=0)
        #self.focus = sum([n.centroid for n in self.nodes]) / len(self.nodes)
        self.focus = np.mean([n.centroid for n in self.nodes], axis=0)
        self.skew = self.metric(self.centroid, self.focus)

        # Calculate Dispersion:
        #self.dispersion = sum([self.metric(self.centroid, vec)
        #    for vec in self.vectors]) / len(self.objects)
        self.dispersion = np.mean([self.metric(self.centroid, vec)
            for vec in self.vectors], axis=0)

        # Calculate Density:
        #if self.nearest != None: self.density = sum([self.metric(
        #    v, self.encoder(self.nearest(self.objects[i])))
        #    for i, v in self.vectors]) / len(self.objects)
        if self.nearest != None: self.density = np.mean([self.metric(
            v, self.encoder(self.nearest(self.objects[i])))
            for i, v in enumerate(self.vectors)], axis=0)
        else: self.density = None
            # NOTE: if objects are placed in clusters different from their
            #   nearest neighbor, this will include a few phony values.

        # Calculate Medoid:
        self.medoid = self.objects[np.argmin([
            self.metric(self.centroid, v) for v in self.vectors])]


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
