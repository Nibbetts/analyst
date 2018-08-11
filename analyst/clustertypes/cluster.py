import numpy as np
from collections import OrderedDict
from copy import copy, deepcopy

class Cluster:

    QUIET_STATS = { # Members not used in Analyst reports.
        "CATEGORY",
        "ID",
        "Name",
        "Medoid",
        "Subcluster Category",

        "Nodes",
        "Objects",
        "Subcluster IDs",
        "Subcluster Count"

        "Ctr Dist Avg",
        "Ctr Dist Min",
        "Ctr Dist Max",
        "Ctr Dist Range",
        "Ctr Dist Std",

        "Norm Avg",
        "Norm Min",
        "Norm Max",
        "Norm Std",
    }

    def __init__(self, category, encoder, metric, objects, nearest=None,
            vectors=None, nodes=[], auto=False, ID=None, name=None,
            subcluster_category=None, subcluster_ids=[],
            quiet_stats_override=None, **metric_args):
        """
        Parameters:
            category: string; what type of cluster this is.
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
            subcluster_category: if composed of other clusters, or by nature
                contains others we should know about or are building off of,
                that sub-category name should go here, to be able to reference
                that evaluator.
            subcluster_ids: list of cluster ids so this cluster's specific
                subclusters can be found from the clusterizer they belong to.
            quiet_stats_override: iterable containing stats to withold from
                being list-documented in the analyst report. Has no effect on
                which stats are printed when we print the cluster itself.
                NOT commonly used, except to withold extraneous information -
                best to use QUIET_STATS with additions.
            metric_args: additional arguments to the metric function.
                Use like **kwargs is used.

        Usage:
            Inside an evaluator, if you want to make a cluster store more or
            different kinds of data, put these data inside of self.stats_dict.
            Clusterizers automatically find meta-stats on these values.
        """
        self.ID = ID
        self.CATEGORY = category
        self.SUBCLUSTER_CATEGORY = subcluster_category
        self.name = name
        self.objects = objects
        self.subcluster_ids = subcluster_ids
        self.encoder = encoder
        self.metric = metric
        self.nearest = nearest
        self.vectors = vectors
        self.nodes = nodes
        self.auto = auto
        self.quiet_stats_override = quiet_stats_override
        self.metric_args = metric_args
        # self.medoid = ""
        # self.centroid = []
        # self.centroid_distances = []
        # self.focus = []
        # self.norms = []

        self.stats_dict = OrderedDict()

        if self.auto: self.calculate()

    def __iadd__(self, B):
        self.objects = list(set(self.objects + B.objects))
        # Both must be python lists,
        #   UNLESS USE np.hstack() or np.concatenate(),
        #   then change add_objects and add_nodes functions as well!
        self.nodes = list(set(self.nodes + B.nodes))
        # Auto-calculation is dependent on the one being added to:
        #if self.auto | B.auto: self.calculate()
        self.subcluster_ids = list(set(self.subcluster_ids + B.subcluster_ids))
        self.vectors = None
        if self.auto:
            self.calculate()

        return self

    def __add__(self, B):
        return Cluster(
            self.CATEGORY, self.encoder, self.metric, nearest=self.nearest,
            vectors=None,
            objects=list(set(self.objects + B.objects)),
            nodes=list(set(self.nodes + B.nodes)),
            # Both must be auto for it to calculate:
            auto=self.auto & B.auto, ID=None, name=None,
            subcluster_category=self.SUBCLUSTER_CATEGORY if \
                self.SUBCLUSTER_CATEGORY == B.SUBCLUSTER_CATEGORY else None,
            subcluster_ids=list(set(self.subcluster_ids + B.subcluster_ids))
            **self.metric_args)

    def modifiable_copy(self, auto=False):
        # copy(list) is like [i for i in list]
        return Cluster(
            self.CATEGORY, # immutables are copied anyway
            self.encoder, # if references are modified, won't change original
            self.metric,
            copy(self.objects), # Make new lists referencing same info!
            nearest=self.nearest,
            vectors=copy(self.vectors), # ...
            nodes=copy(self.nodes), # ...
            auto=auto, # Expect it will be modified, so no auto.
            ID=self.ID, # Default keep unique data, expecting it will be changed
            name=self.name, # ...
            subcluster_category=self.SUBCLUSTER_CATEGORY,
            subcluster_ids=copy(self.subcluster_ids), # ...
            quiet_stats_override=self.quiet_stats_override,
            **self.metric_args)

    def __len__(self):
        # Cardinality
        return len(self.objects)

    def __getitem__(self, index):
        return self.objects[index]

    def __str__(self):
        max_length = max([len(s) for s in self.stats_dict])
        format_str = "{:<" + str(max_length + 1) + "}: "
        def line(key, value):#, comma=True):
            #c = "," if comma else ""
            #return c + "\n\t" + format_str.format(key) + str(value)
            return "\n\t" + format_str.format(key) + str(value)

        result = "Cluster("
        for (key, value) in self.stats_dict.items():
            if key not in ["Nodes", "Objects"]:
                result += line(key, value)
        result += line("Nodes", self.nodes) # Simply moving these to the bottom.
        result += line("Objects", self.objects)
        result += "\n)"

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
            self.vectors = np.array([self.encoder(o) for o in self.objects]) # TODO: vectorize?

        if len(self.objects) > 0:

            # Centroid:
            #self.centroid = sum(self.vectors) / len(self.vectors)
            self.centroid = np.mean(self.vectors, axis=0)

            # Calculate Medoid:
            medoid_i = np.argmin([
                self.metric(self.centroid, v, **self.metric_args) \
                for v in self.vectors])
            self.medoid = self.objects[medoid_i]
        
        else:
            self.medoid = ""

        self.stats_dict["CATEGORY"] = self.CATEGORY
        self.stats_dict["ID"] = self.ID
        self.stats_dict["Name"] = self.name
        self.stats_dict["Medoid"] = self.medoid
        self.stats_dict["Population"] = len(self.objects)
        self.stats_dict["Subcluster Category"] = self.SUBCLUSTER_CATEGORY
        if self.SUBCLUSTER_CATEGORY is not None:
            self.stats_dict["Subcluster Count"] = len(self.subcluster_ids)
            self.stats_dict["Subcluster IDs"] = self.subcluster_ids

        if len(self.objects) > 0:

            self.stats_dict["Centroid Norm"] = np.linalg.norm(self.centroid)

            # Calculate Dispersion:
            #self.dispersion = sum(self.metric(self.centroid, vec)
            #    for vec in self.vectors) / len(self.objects)
            self.centroid_distances = [self.metric(
                self.centroid, vec, **self.metric_args) for vec in self.vectors]
            self.stats_dict["Dispersion"] = np.mean(
                self.centroid_distances, axis=0)

            # Distance from Medoid to Centroid:
            self.stats_dict["Medoid Dist to Ctr"] = self.metric(
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

            # Mostly Quiet Stats, not used in Analyst, but printed in Cluster data:
            cd_min = np.min(self.centroid_distances)
            cd_max = np.max(self.centroid_distances)
            self.stats_dict["Ctr Dist Avg"] = self.stats_dict["Dispersion"]
            self.stats_dict["Ctr Dist Min"] = cd_min
            self.stats_dict["Ctr Dist Max"] = cd_max
            self.stats_dict["Ctr Dist Range"] = cd_max - cd_min
            self.stats_dict["Ctr Dist Std"] = np.std(self.centroid_distances)

            self.norms = [np.linalg.norm(v) for v in self.vectors]
            n_max = np.max(self.norms)
            n_min = np.min(self.norms)
            self.stats_dict["Norm Avg"]   = np.mean(self.norms)
            self.stats_dict["Norm Min"]   = n_min
            self.stats_dict["Norm Max"]   = n_max
            self.stats_dict["Norm Range"] = n_max - n_min # This one NOT quiet.
            self.stats_dict["Norm Std"]   = np.std(self.norms)

            self.stats_dict["Nodes"]      = \
                [str(node) for node in sorted(self.nodes)]
            self.stats_dict["Objects"]    = \
                [str(o) for o in sorted(self.objects)]


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
