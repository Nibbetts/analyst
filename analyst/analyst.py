# Compatibility for python 2 and 3:
from __future__ import print_function
from __future__ import absolute_import
from builtins import str, bytes
from io import open

# Normal Packages:
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
#import cPickle as pickle
import dill as pickle
import os
#from sys import version_info
#import multiprocessing
import traceback

# Own files:
from .evaluators import *
from .simulation import *
#import evaluators
#from test_set_2d import TestSet2D



#--------------------------------------------------------------------------#
# General Functions                                                        #
#--------------------------------------------------------------------------#

def isstring(obj):
    '''Basic string type checker, for python 2-3 compatibility.'''
    return isinstance(obj, str) or isinstance(obj, bytes)

def _file_extension(f_name):
    '''Appends missing file extension, or keeps original.'''
    return str(f_name) if u"." in str(f_name) else f_name + u".dill"

def save(analyst, file_name=None):
    '''Function for saving Analyst objects only.'''
    return analyst.save(file_name)

def load(f_name, print_report=False):
    '''Function for loading Analyst objects only.
        NOTE: Since we save with highest protocol, you will not be able
        to load an Analyst into python 2 if saved from python 3.
        This may also help prevent string encoding incompatibilities.'''
    name = _file_extension(f_name)
    try:
        with open(name, 'rb') as file:
            an = pickle.load(file)
            #an._deserialize(metric, encoder, decoder, cluster_algorithms, analogy_algorithms)
            an.analysis(print_report=print_report, recalculate=[])
            return an
    except Exception as e:
        print(u"ERROR: Unable to load or deserialize Analyst object "
            u"from file: '{}'".format(name))
        print(e)
        #raise e
        return None

def unsave(file_name):
    '''Remove an existing dill file.
        Will add file extension '.dill' if none given.
        Careful, this is irreversible and works on any file!'''
    f_name = _file_extension(file_name)
    try:
        os.remove(f_name)
        return True
    except:
        return False



#--------------------------------------------------------------------------#
# Metrics and Measurements                                                 #
#--------------------------------------------------------------------------#

def angle(vec1, vec2, degrees=True):
    '''Returns: scalar (a float) representing the angle between two vectors.
        Can be used as a metric.
        Default is degrees, since cosine dist is more common than radians.'''
    angle = np.arccos(np.clip(np.dot(
            vec1 / np.linalg.norm(vec1),
            vec2 / np.linalg.norm(vec2)),
        -1.0, 1.0))
    return angle * 180 / np.pi if degrees else angle

def weighted_diff(l):#(a,b):
    '''A comparison function which emphasizes numbers further from zero.
        l: array-like.
        returns: a float representing how different numbers in l are.'''
    a = np.max(l)
    b = np.min(l)
    average = (np.abs(a) + np.abs(b))/2.0
    if average != 0: return (a - b)/average
    else: return np.nan

def curiosity(l):
    '''A comparison function which emphasizes range properties of a list;
        in a way, how unevenly distributed they are:
            positive -> highly differing min and max
            near zero -> more evenly distributed (ex: when min is half of max)
            negative -> highly similar min and max
        All are scale invariant; differences emphasized relative to scale.
        l: array-like.
        returns: a float representing how different numbers in l are.'''
    # range/abs(min) = (max-min)/abs(min) = max/abs(min) - sign(min)
    # Then a natural log.
    m = np.min(l)
    rng = np.max(l) - m
    if rng == 0: return -np.inf
    absm = np.abs(m)
    if absm == 0: return np.inf
    return np.log(rng/float(absm))

def odd_one_out(l):
    '''A comparison function which spikes when there is a value which is very
        different and sticks out from the rest.
        l: array-like.
        returns: a float representing how oddly one value in l sticks out.'''
    vals = [curiosity(l[:skip] + l[skip+1:]) \
        for skip in range(len(l))]
    return np.max(vals) - np.min(vals)



#------------------------------------------------------------------------------#
# Analyst Class                                                                #
#------------------------------------------------------------------------------#

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
                * repulsion -- avg dist to nearest
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

    BUILT_IN_CATEGORIES = [
        u"Spatial",
        u"Nodes",
        u"Extremities",
        u"Supernodes",
        u"Hubs",
        u"Nuclei",
        u"Chains",
        u"NCC",
        u"LNCC",
        u"Anti-hubs"]

    def __init__(self, embeddings=None, strings=None,
        encoder=None, decoder=None, metric=u"cosine", evaluators=[u"All"],
        auto_print=True, desc=None, calculate=True, **metric_args):
        """
        Parameters:
            embeddings -- list of vectors populating the space.
                Must have static indeces. (ie: not dict or set) (semi-optional)
            strings -- a list of strings. (semi-optional)
            encoder -- a callable to convert strings to vectors. (semi-optional)
            decoder -- a callable to convert vectors to strings. (semi-optional)
            # NOTE: Put in as many of the above four as you can to reduce
            #   initial computation time. It WILL build the rest for rapid
            #   access, if it can, or will crash and tell you it can't.
            metric -- the distance metric used throughout,
                accepts any string accepted by scipy.spatial.distance.pdist,
                or any callable accepting vectors as the first two parameters
                and returning a scalar. This allows for custom distance metrics.
            WARNING: custom metrics are orders or magnitude slower than
                scipy's built-ins!
            evaluators -- FILL IN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # cluster_algorithms -- list of tuples (callable, "Description").
            #     Each callable must take an array-like list of vectors and return
            #     a list of array-like lists of vectors, each representing a
            #     cluster. They do not have to partition or even cover the space;
            #     ie, the can contain duplicates and do not have to contain all.
            #     If the left entry is None or is not callable, will expect a
            #     recognized tag in the right instead of a label, indicating to
            #     use a built-in function.
            #     Recognized tags:
            #         "Spatial" -- basic analysis of the space - must occur even
            #             if not included, but not including it will cause it not
            #             to display results.
            #         "Extremities" -- also generally include.
            #         "Nodes" -- also generally include. Many built-in clustering
            #             algorithms will cause these to be computed anyway.
            #         "Hubs"
            #         "Supernodes"
            #         "Nuclei"
            #         "Chains"
            #         "NNNCC" -- basic built-in experimental clustering algorithm
            #         "LNNNCC" -- advanced, or strong version of the above
            #         "Anti-clusters"
            #         "All" -- will automatically include all of the above.
            #     NOTE: As this variable contains functions, it will be altered
            #         to no longer contain functions when an Analyst is pickled.
            auto_print -- whether to print reports automatically after analyses.
            desc -- optional short description/title for this analyst instance.
            calculate -- whether or not to run the analysis.
                Typically always True.
            metric_args -- these are extra arguments to be given to metric.
        """

        self.auto_print = auto_print
        print(u"")
        self._print(u"Asking the Grand Question",
            u"What is the Purpose of this Space?")
        self.description = str(desc)
        self.file_name = None
        
        # Find and store a callable version of the given metric:
        self._print(u"Laying the Laws of Physics", u"Setting the Metric")
        if callable(metric):
            self.metric = metric
            self.metric_str = None
        else:
            try:
                self.metric_str = metric
                #self.metric_str = str(sp.distance._METRIC_ALIAS[metric])
                #   Only worked in python2.
                self.metric = sp.distance._TEST_METRICS[
                    u"test_" + self.metric_str]
            except Exception as e:
                print(e)
                raise ValueError(u"FATAL ERROR: %s PARAMETER UNRECOGNIZED "
                    u"AND UNCALLABLE!"
                    % str(metric))
        self.metric_args = metric_args

        # Encoder/Decoder Initializations:
        # OK if: encoder and strings, decoder and vectors, strings and vectors:
        #   encoder + strings -> vectors; vectors + strings -> decoder.
        #   decoder + vectors -> strings; strings + vectors -> encoder.
        #   strings + vectors -> encoder & decoder.
        self._print(u"Enumerating the Dimensions",
            u"Making Internal Converters")
        # Find embeddings:
        if embeddings is None:
            if encoder is None or strings is None:
                raise ValueError(u"FATAL ERROR: Without embeddings, you must "
                    u"give both an encoder and strings!")
            else:
                self._print(u"Stretching the Fabric of Space and Time",
                    u"Finding Embeddings")
                self.space = np.array([encoder(w) for w in
                    tqdm(strings, disable=(not self.auto_print))])
        else: self.space = embeddings
        #
        # Find strings:
        if strings is None:
            if embeddings is None or decoder is None:
                raise ValueError(u"FATAL ERROR: Without strings, you must give "
                    u"both a decoder and embeddings!")
            else:
                self._print(u"Naming Stars and Drawing a Star Map",
                    u"Collecting Strings")
                self.strings = [decoder(v) for v in
                    tqdm(embeddings, disable=(not self.auto_print))]
        else: self.strings = strings
        # Now we have both embeddings and strings.
        assert len(self.space) == len(self.strings)
        #
        # Make encoder:
        if encoder is None:
            self._print(u"Filling the Void", u"Mapping New Encoder")
            self.s_to_vec = {}
            for i in trange(len(self.strings), disable=(not self.auto_print)):
                self.s_to_vec[self.strings[i]] = self.space[i]
            self.encode = self.s_to_vec.__getitem__
        else:
            assert callable(encoder)
            self.encode = encoder
        #
        # Make decoder:
        if decoder is None:
            self._print(u"Mapping the Emptiness", u"Mapping New Decoder")
            self.vec_to_s = {}
            for i in trange(len(self.space), disable=(not self.auto_print)):
                self.vec_to_s[str(self.space[i].tolist())] = self.strings[i]
            #self.decode = self.vec_to_s.__getitem__
            self.decode = lambda vec: self.vec_to_s[str(vec.tolist())]
        else:
            assert callable(decoder)
            self.decode = decoder

        # A separate Map for quick string indexing:
        self.s_to_ix = {}
        self._print(u"Indexing Planets", u"Making String-Index Mappings")
        for ix, s in enumerate(
                tqdm(self.strings, disable=(not self.auto_print))):
            self.s_to_ix[s] = ix
        # NOTE: I'm not making vec_to_ix because I assume that the length of the
        #   vectors makes it more efficient to use s_to_ix[decode(vec)].

        # These are explained in the neighbors and distance matrix getters:
        self.distance_matrix = None
        self.neighbors = None
        self.neighbors_dist = None

        # Data
        self.graph_info = []
        self.categories = []
        self.evaluators = []
        self.category_lists = []
        self.add_evaluators(*evaluators)

        # Run Analyses:
        if calculate:
            self.analysis(
                print_report=self.auto_print, auto_save=False, recalculate=[])


    # Generic type converters & tools for inputs and outputs:
    # NOTE: These will be slower than type-specific functions.
    #
    def exists(self, obj):
        if obj in self.s_to_ix: return True
        try:
            self.decode(obj)
            return True
        except: pass
        if obj // len(self.space) == 0: return True
        return False
    #
    # These work on objects not in the model, but this requires in_model=False,
    #   and of course only returns the nearest object.
    def as_index(self, obj, in_model=True):
        if in_model:
            if isstring(obj):
                return self.s_to_ix[obj]
            try: return self.s_to_ix[self.decode(obj)]
            except: return int(obj)
        else:
            return self.s_to_ix[self.decode(self.nearest(obj, in_model=False))]
    #
    def as_vector(self, obj, in_model=True):
        if in_model:
            if isstring(obj):
                return self.encode(obj)
            try: return self.space[obj]
            except: return obj
        else:
            return self.nearest(obj, in_model=False)
    #
    def as_string(self, obj, in_model=True):
        if in_model:
            if isstring(obj):
                return obj
            try: return self.strings[obj]
            except: return self.decode(obj)
        else:
            return self.decode(self.nearest(obj, in_model=False))


    # Gets the kth neighbor of obj. Negatives for furthest, 0 for self, positive
    #   for nearest. If not in_model, 0 is nearest.
    # Attempts to return the same type given, ie: index, string, or vector.
    # Ensures neighbors will be calculated before, without being recalculated.
    def neighbor_k(self, obj, k, in_model=True):
        if in_model:
            assert -1 <= k <= 2
            if k == 0: return obj
            n = k - 1 if k > 0 else k
            i = self.neighbors_getter()[0][self.as_index(obj)][n]
            if isstring(obj):
                return self.strings[i]
            try:
                int(obj)
                return i
            except: return self.space[i]
        else:
            # Note that if not in_model, we require obj to be a vector.
            self.space[self.arbitrary_vector_neighbors(obj)[k]]
    #
    def nearest(self, obj, in_model=True):
        if in_model:
            return self.neighbor_k(obj, 1)
        else:
            # Note that if not in_model, we require obj to be a vector.
            if self.metric_str != u"cosine":
                return self.space[np.argmin(self.arbitrary_vector_dist(obj))]
            else: # Optimization for cosine similarity:
                return self.space[np.argmax(np.dot(
                    self.space, np.array([obj]).T.squeeze()))]


    # Distance and Neighbor Computation:
    def distance_matrix_getter(self):
        # Allows us to only have to compute the distance matrix once, if needed.
        # Makes use of scipy in the fastest way possible. Any metric string
        #   recognized by scipy will work, or any valid function.
        #   For speed, prefer to use string, if one exists for desired metric.
        if self.distance_matrix == None:
            # Distance Matrix Calculation
            self._print(u"Acquainting the Species",
                u"Computing Distance Matrix")
            self.distance_matrix = sp.distance.squareform(sp.distance.pdist(
                self.space,
                self.metric_str if self.metric_str != None else self.metric,
                **self.metric_args))
        return self.distance_matrix

    def neighbors_getter(self):
        # Allows us to only have to compute neighbors etc once, if needed.
        if self.neighbors is None: # Must use 'is' here, else elementwise!
            # Need distance matrix filled in:
            self.distance_matrix_getter()

            # Initialize Empty Arrays:
            self.neighbors = np.empty((len(self.space),3), dtype=np.uint64)
                # Indeces correspond to indeces of vectors in the space.
                #   For each:
                #   [index of nearest, index of 2nd-nearest, index of furthest]
            self.neighbors_dist = np.empty(
                (len(self.space),3))
                # Same format as above, except distances to those indexed above.

            # Finding Furthest Neighbors
            self._print(u"Misconstruing Relations",
                u"Finding Furthest Neighbors")
            self.neighbors[:,2] = np.argmax(self.distance_matrix, axis=1)
            self.neighbors_dist[:,2] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,2]]
            self.distance_matrix[
                range(len(self.space)), range(len(self.space))] = np.inf
            # Finding Nearest Neighbors
            self._print(u"Forming Alliances", u"Finding Nearest Neighbors")
            self.neighbors[:,0] = np.argmin(self.distance_matrix, axis=1)
            self.neighbors_dist[:,0] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,0]]
            self.distance_matrix[
                range(len(self.space)), self.neighbors[:,0]] = np.inf
            # Finding Second-Nearest Neighbors
            self._print(u"Obfuscating Dynastic Ties",
                u"Finding 2nd-Nearest Neighbors")
            self.neighbors[:,1] = np.argmin(self.distance_matrix, axis=1)
            self.neighbors_dist[:,1] = self.distance_matrix[
                range(len(self.distance_matrix)),
                self.neighbors[:,1]]
            # Put back the numbers we removed:
            self._print(u"Resetting the Ship's Computer",
                u"Repairing Distance Matrix")
            self.distance_matrix[
                range(len(self.space)), self.neighbors[:,0]
            ] = self.neighbors_dist[:,0]
            self.distance_matrix[
                range(len(self.space)), range(len(self.space))] = 0.0

        return self.neighbors, self.neighbors_dist

    def arbitrary_vector_dist(self, vector):
        # Takes a vector not in the model and finds its distance to every obj
        #   in the model, taking advantage of scipy's optimizations.
        # NOTE: results are not stored, so recomputes every time.
        return sp.distance.cdist(
                np.atleast_2d(vector), self.space,
                self.metric_str if self.metric_str != None else self.metric,
                **self.metric_args
            ).squeeze()
            
    def arbitrary_vector_neighbors(self, vector):
        # Takes a vector not in the model and finds its distance to every obj
        #   in the model, returning a 1D array of indeces (not vectors!)
        # Includes an extra optimization for the common case
        #   that metric is cosine similarity.
        # NOTE: results are not stored, so recomputes every time.
        if self.metric_str != u"cosine":
            return np.argsort(self.arbitrary_vector_dist(vector))
        else:
            distances = np.dot(self.space, np.array([vector]).T.squeeze())
            return distances.argsort()[::-1]

    # These return the kth neighbor of all objects in the space, index-to-
    #   index | distance, respectively.
    #   Use negative for furthest, 0 for self, positive for nearest.
    #   Default None will return the whole matrix.
    def kth_neighbors(self, k=None):
        if k == None: return self.neighbors_getter()[0]
        assert -1 <= k <= 2
        if k == 0: return range(len(self.space))
        n = k - 1 if k > 0 else k
        return self.neighbors_getter()[0][:,n]

    def kth_neighbors_dist(self, k=None):
        if k == None: return self.neighbors_getter()[1]
        assert -1 <= k <= 2
        if k == 0: return np.zeros((len(self.space)), dtype=np.uint64)
        n = k - 1 if k > 0 else k
        return self.neighbors_getter()[1][:,n]

    # Computes the downstream nearest neighbor, and lists the path if asked,
    #   starting from obj's kth-nearest neighbor,
    #   then going from one NEAREST neighbor to another until we start to repeat
    #   (having reached a node). Thus the last two in the list make a node. If
    #   you start from k=0, obj will be included. k=1 is same, but without obj.
    # Note: non type-specific, and returns same type as given.
    # Note: should be faster if path not kept.
    # Note: probably less efficient if you use straight vectors, because of
    #   equality checks. Maybe most efficient for indeces?
    def downstream(self, obj, start_neighbor_k=0, give_path=False):
        path = None
        if give_path: path=[]
        current = self.neighbor_k(obj, start_neighbor_k, in_model=True)
        while current not in path: # More efficient than overhead for a set?
            if give_path: path.append(current)
            current = self.nearest(current)
        if give_path: return path
        else: return current

    # Superfast metric function for objects within the model only, using dicts.
    # Note: generic types.
    def metric_in_model(self, obj1, obj2):
        return self.distance_matrix_getter()[
            self.as_index(obj1, in_model=True),
            self.as_index(obj2, in_model=True)]


    # NOTE: This function does NOT force the evaluator to pre-calculate!
    # NOTE: Since categories SHOULD be unique among evaluators,
    #   this function will only return the first match it finds. Or None.
    def find_evaluator(self, category, force_creation=False):
        # force_creation: whether or not to create a default evaluator for
        #   built-ins.
        for e in self.evaluators:
            if str(e.CATEGORY) == category: return e
        if force_creation: 
            return Analyst.make_default_evaluator(str(category))

    # Makes Built-in Clusterizers with Default Values:
    # Note: Can take some parameterization, such as "Nodal 10-Hubs", or "2Hubs".
    #   "Hubs" with no number defaults to "Nodal 4-Hubs".
    @staticmethod
    def make_default_evaluator(category):
        cat = category.lower()
        if cat == u"spatial":
            return Spatializer()
        elif cat == u"nodes":
            return NodeClusterizer()
        elif cat == u"extremities":
            return ExtremityClusterizer()
        elif cat == u"supernodes":
            return SupernodeClusterizer()
        elif u"hubs" in cat and u"anti" not in cat:
            thresh = 4
            nodal = False
            if u"nodal " in cat:
                nodal = True
            try:
                thresh = int(cat[6 if nodal else 0:-5]) if u"-hubs" in cat \
                    else int(cat[6 if nodal else 0:-4])
            except: pass
            if cat == u"hubs":
                nodal = True
                cat = u"Nodal 4-Hubs"
            else: cat = category
            return HubClusterizer(threshold=thresh, nodal=nodal, category=cat)
        # ADD STUFF!!
        else:
            print(u"UNRECOGNIZED BUILT-IN EVALUTATOR '"
                + category + u"'! SKIPPING IT.")
            return None

    def add_evaluators(self, *args):
        # Helper function
        def rename_evaluator(evaluator):
            version = 2
            category = evaluator.CATEGORY
            while category in self.categories:
                category = evaluator.CATEGORY + u" (" + str(version) + u")"
            evaluator.CATEGORY = category

        warning = u"WARNING: Evaluator category '{}' already exists! {} it."

        # Add evaluators and categories
        for e in args:
            if isstring(e): # If keyword
                if str(e.lower()) == u"all": # If keyword 'All'
                    for cat in Analyst.BUILT_IN_CATEGORIES:
                        evaluator = Analyst.make_default_evaluator(cat)
                        if evaluator != None:
                            if evaluator.CATEGORY not in self.categories:
                                self.categories.append(evaluator.CATEGORY)
                                self.evaluators.append(evaluator)
                            else:
                                self._print(warning.format(
                                    evaluator.CATEGORY, u"SKIPPING"))
                else: # If keyword matching a built-in
                    evaluator = Analyst.make_default_evaluator(e)
                    if evaluator != None:
                        if evaluator.CATEGORY in self.categories:
                            self._print(warning.format(
                                evaluator.CATEGORY, u"RENAMING"))
                            rename_evaluator(evaluator)
                        self.categories.append(evaluator.CATEGORY)
                        self.evaluators.append(evaluator)
            else: # If actual evaluator object
                if e != None:
                    if e.CATEGORY in self.categories:
                        self._print(warning.format(e.CATEGORY, u"RENAMING"))
                        rename_evaluator(e)
                    self.categories.append(e.CATEGORY)
                    self.evaluators.append(e)

        # Modify category_lists to match categories
        diff = len(self.categories) - len(self.category_lists)
        self.category_lists += np.empty(shape=(diff,0)).tolist()


    #--------------------------------------------------------------------------#
    # General Analyses:                                                        #
    #--------------------------------------------------------------------------#

    def analysis(self, print_report, auto_save=True, recalculate=[]):
        # Won't recalculate any but those whose categories are listed.
        # Even those it doesn't recalculate, it will still get their data and
        #   update its own in case it has changed.

        # Run the Evaluations:
        for evaluator in self.evaluators:
            try:
                data_dict, starred, category = evaluator.calculate(
                    recalculate_all=evaluator.CATEGORY in recalculate,
                    #   Only does those not yet done.

                    # NOTE: The rest are the kwargs:
                    embeddings=self.space,        draw_progress=self.auto_print,
                    strings=self.strings,         metric_str=self.metric_str,
                    printer_fn=self._print,       metric_fn=self.metric,
                    as_string_fn=self.as_string,  metric_args=self.metric_args,
                    as_index_fn=self.as_index,    encoder_fn=self.encode,
                    as_vector_fn=self.as_vector,  decoder_fn=self.decode,
                    string_ix_map=self.s_to_ix,   exists_fn=self.exists,
                    is_string_fn=isstring,        angle_fn=angle,

                    generic_neighbor_k_fn=self.neighbor_k,
                    generic_nearest_fn=self.nearest,
                    kth_neighbors_ix_fn=self.kth_neighbors,
                    kth_neighbors_dist_fn=self.kth_neighbors_dist,
                    distance_matrix_getter_fn=self.distance_matrix_getter,
                    arbitrary_dist_fn=self.arbitrary_vector_dist,
                    arbitrary_neighbors_fn=self.arbitrary_vector_neighbors,
                    metric_in_model_fn=self.metric_in_model,
                    downstream_fn=self.downstream,
                    evaluator_list=self.evaluators,
                    find_evaluator_fn=self.find_evaluator,
                    simulate_cluster_fn=simulate_cluster)

                # The below compatibilities should be unnecessary because both
                #   keys and starred come from same source, thus same version.
                #   Also, they're just going to be printed later, so no others
                #   need it either.
                #starred = map(str, starred)
                #key = str(key)
                #category = str(category)

                for (key, value) in data_dict.items():
                    self._add_info(value, category, key, key in starred)

            except Exception: # as e:
                #print(e)
                traceback.print_exc()
                print(u"ERROR IN CALCULATION OF %s. DOES YOUR EVALUATOR "
                    u"INHERIT FROM AN Evaluator CLASS?"
                    % evaluator.CATEGORY)
        
        if auto_save:
            if self.file_name != None: self.save()
            else: print("CANNOT AUTO-SAVE WITHOUT HAVING BEEN SAVED AT LEAST "
                "ONCE; NO FILENAME.")
        if print_report: self.print_report()


    # SPECIFICS INSPECTION:

    def graph(self, hist_key, bins=64, **kwargs):
        """
            Description: creates a histogram according to key printed in report.
        """
        # self.graph_info[hist_key] == 
        #   ([analyst_descriptions], category, description, [datasets])
        x = self.graph_info[hist_key][3]
        plt.hist(x, bins=bins, **kwargs)
        plt.legend(self.graph_info[hist_key][0])
        plt.xlabel(self.graph_info[hist_key][2])
        plt.ylabel("Occurrences")
        plt.title(self.graph_info[hist_key][1])
        plt.show()


    # COMPARATIVE:

    @staticmethod
    def _formatit(data, width=10, parentheses=False, start_at=0):
        #if parentheses: w = max(9, width)
        #else: w = max(7, width)
        w = max(9, width)
        result = ""
        if data is None:
            result = " " * w
        elif isstring(data) or parentheses or not np.isfinite(data):
            # Strings or Bytestrings
            result = " " # For negatives on others
            if parentheses: result += "(" + str(data) + ")"
            else: result += str(data)
            format_str = "{:" + str(w) + "}"
            result = format_str.format(result)
            if len(result) > w:
                result += "\n" + " " * (start_at + w)
        else:
            if (abs(data) >= 1e4 and not parentheses) \
                    or (abs(data) < 1e-4 and data != 0):
                # Large or Small
                format_str = "{: " + str(w) + "." + str(w - 7) + "e}"
                result = format_str.format(data)
            elif isinstance(data, int):
                # Integer
                format_str = "{:< " + str(w) + "d}"
                result = format_str.format(int(data))
            else:
                # Float
                format_str = "{: " + str(w - 1) + "." \
                    + str(w - 2 - len(str(int(abs(data))))) + "f}"
                result = format_str.format(data)
        return result

    @staticmethod
    def compare(ana_list, w=10, comparators=[u"default"]):
        # Lists side by side the values for each analyst in the list,
        #   as well as a column for each comparator, run on that stat.
        # w: Numbers will have space for w-2 digits, (w-2 because of . and - ).
        #   Total width will be: (6 + (w + 1)*(an + comp) + len(description))
        # comparator:
        #   empty list: no comparison columns
        #   callable (Takes list of available values; does NOT need to handle
        #       strings or None), or a built-in:
        #   "all": all builtins
        #   "default:" includes range, curiosity, and odd_one_out
        #   "std": standard deviation across the stat
        #   "avg" or "average": average across the stat
        #   "max" or "maximum": maximum value
        #   "min" or "minimum": minimum value
        #   "rng" or "range": max value minus min value
        #   "weighted_diff": accentuates numbers far from zero
        #   "curiosity": accentuates differences and uniformity
        #   "odd_one_out": spikes when one value differs largely from others
        # ana_list: a list of analysts. Kinda thought that was clear... :)
        # Returns: a grapher object with multi-histogram information from the!!!!!!!!!!!!!!!!!!!!!!!!!!
        #   the comparison.
        assert len(ana_list) > 0
        ana_list[0]._print(u"Bridging Universes",
            u"Building Comparison & Report")
        print(u"")

        # Descriptions to use:
        title = u"Comparison:"
        descriptions = []
        for i, a in enumerate(ana_list):
            if a.description == None:
                title += u" ANALYST " + str(i + 1)
                descriptions.append(u"ANALYST " + str(i + 1))
            else:
                title += " " + a.description.upper()
                descriptions.append(a.description.upper())
            if i < len(ana_list) - 1: title += ","
        print(title)

        # Comparator:
        comparisons = []
        def rng(l): return np.max(l) - np.min(l)
        defaults = [rng, weighted_diff, curiosity, odd_one_out]
        all_builtins = [np.std, np.mean, np.max, np.min] + defaults
        for i, c in enumerate(comparators):
            if callable(c): comparisons.append(c)
            else: 
                word = c.lower()
                if word == u"all":
                    for f in all_builtins:
                        if f not in comparisons: comparisons.append(f)
                elif word == u"default" or word == u"defaults":
                    for f in defaults:
                        if f not in comparisons: comparisons.append(f)
                elif word == u"std" and np.std not in comparisons:
                    comparisons.append(np.std)
                elif (word == u"avg" or word == u"average") \
                    and np.mean not in comparisons: comparisons.append(np.mean)
                elif (word == u"max" or word == u"amax" or word == u"maximum") \
                    and np.max not in comparisons: comparisons.append(np.max)
                elif (word == u"min" or word == u"amin" or word == u"minimum") \
                    and np.min not in comparisons: comparisons.append(np.min)
                elif (word == u"rng" or word == u"range") \
                    and rng not in comparisons: comparisons.append(rng)
                elif word == u"weighted_diff" and weighted_diff not in \
                    comparisons: comparisons.append(weighted_diff)
                elif word == u"curiosity" and curiosity not in comparisons:
                    comparisons.append(curiosity)
                elif word == u"odd_one_out" and odd_one_out not in comparisons:
                    comparisons.append(odd_one_out)

        # Column Headers:
        title_string = u"   " + u"{} " * len(ana_list) + u"|" + \
            "{} " * len(comparisons) + u"| PROPERTY"
        titles = descriptions + [c.__name__.upper() for c in comparisons]
        s = 3
        for i, t in enumerate(titles):
            titles[i] = Analyst._formatit(t, w, False, s)
            s += w + 1
            if i == len(ana_list): s += 1
        print(title_string.format(*titles))

        # Line Template:
        line_string = u"  {}" + u"{} " * len(ana_list) + u"|" + \
            u"{} " * len(comparisons) + u"|{}{}"
        #graphs = []

        # Helper function; All the work to print one category:
        def print_category(c):
            print(c + u":")
            # Get the information from this category in each Analyst:
            category_lists = []
            for a in ana_list:
                try:
                    index = a.categories.index(c)
                    category_lists.append(a.category_lists[index])
                except: category_lists.append(None)
            
            # Collect data:
            descriptions = []
            desc_lower = set()
            stars = {}
            values = {}
            for a, l in enumerate(category_lists): # a is index of analyst
                if l is not None:
                    for t in l:
                        desc = t[0].lower()
                        values[(a, desc)] = t[1]
                        if desc not in desc_lower:
                            desc_lower.add(desc)
                            descriptions.append(t[0])
                            stars[desc] = t[2]
                        else: stars[desc] = stars[desc] or t[2]

            # Print lines:
            for desc in descriptions:
                d = desc.lower()
                is_hist = u"histogram key" in d

                # Gather Data:
                datalist = [values[(a, d)] if (a, d) in values else u"" \
                    for a in range(len(ana_list))]
                # Comparisons:
                numbers = [d for d in datalist if not isstring(d)]
                if numbers == [] or is_hist:
                    comps = [u""] * len(comparisons)
                else: comps = [comp(numbers) for comp in comparisons]
                # Histograms:
                #if is_hist:
                #    if comps == []!!: comps = [len(graphs)]
                #    else: comps[0] = len(graphs)
                #    #keys = [datalist[i] if data]!!!
                #    graphs.append(tuple("c") + tuple([a.graph_info[...]]))
                # Formatting:
                data = []
                s = 3
                for i, v in enumerate(datalist + comps):
                    data.append(Analyst._formatit(v, w, is_hist, s))
                    s += w + 1
                    if i == len(ana_list): s += 1
                star = u"*" if stars[d] else u" "
                data = [star] + data + [star, desc]
                print(line_string.format(*data))

        # Put it all together - Loop through and print each category in order:
        categories = []
        for a in ana_list:
            for c in a.categories:
                if c not in categories:
                    categories.append(c)
                    print_category(c)



    #--------------------------------------------------------------------------#
    # Information Gathering, Reporting, and Saving Functions                   #
    #--------------------------------------------------------------------------#

    def _add_info(self, var, category, description, star=False):
        # Description and category must be strings.
        #variable = None
        #i = None
        if u"Histogram Key" in description:
            data = ([self.description], category, description, [var])
            try:
                variable = self.graph_info.index(data)
            except:
                variable = len(self.graph_info)
                self.graph_info.append(data)
        else: variable = var
        try:
            i = self.categories.index(category)
        except:
            i = len(self.categories)
            self.categories.append(category)
            self.category_lists.append([])
        found = False
        for entry in self.category_lists[i]: # Replace if it exists:
            if entry[0] == description:
                entry = (description, variable, star)
                found = True
                break # NOTE: will only replace the first. So don't duplicate.
        if not found: # Add it if it doesn't exist:
            self.category_lists[i].append((description, variable, star))

    def _print(self, string=u"", report=None):
        if self.auto_print: print("{:<40}".format(u"\r" + str(string) + u"...")
            + ("" if report is None else u"(" + report + u")"))

    def print_report(self, w=10):
        self._print(u"Revealing the Grand Plan", u"Printing Report")
        print(u"")
        if self.description != None: print(self.description.upper())
        for i, category in enumerate(self.categories):
            print(category + u": ")
            for cat in self.category_lists[i]:
                print(u"  {}{} {}{}".format(
                    "*" if cat[2] else u" ", # Stars
                    Analyst._formatit(cat[1], w, u"Histogram Key" in cat[0], 3),
                    u"*" if cat[2] else u" ", # Stars
                    cat[0]))

    def save(self, file_name=None):
        try:
            f_name = self.file_name if file_name is None else file_name
            #obj._serialize()
            with open(_file_extension(f_name), 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.file_name = f_name
            return True
        except Exception as e:
            print(e)
            return False


# END OF ANALYST CLASS.
################################################################################



# Brief script-like behavior for development, debugging, and testing purposes:
if __name__ == "__main__":
    #import TestSet2D

    raise Exception("USAGE ERROR: analyst module script behabior not defined. "
        "Should be imported as a package.")
