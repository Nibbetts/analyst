# Compatibility for python 2 and 3:
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import bytes, str # Requires 'future' package to be installed.
from io import open

# Normal Packages:
import numpy as np
import scipy.spatial as sp
import scipy.special as ss
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
#import cPickle as pickle
import dill as pickle
import os
#from sys import version_info
#import multiprocessing
import traceback
from collections import OrderedDict
import ray
import psutil
from copy import copy, deepcopy

# Own files:
from .evaluators import *
from .simulation import *
#import evaluators
#from test_set_2d import TestSet2D



#--------------------------------------------------------------------------#
# General Functions & Utilities                                            #
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
            # an = pickle.load(file)
            #an._deserialize(metric, encoder, decoder, cluster_algorithms, analogy_algorithms)
            # an.analysis(
            #     print_report=print_report, auto_save=False, recalculate=[])
            # return an
            a = pickle.load(file)
            a.file_name = name # Update file name in case it has changed
            return a
    except:
        traceback.print_exc()
        print(u"ERROR: Unable to load Analyst object "
            u"from file: '{}'".format(name))
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

# TODO! Now that this is implemented, need to refactor code to use it!
class UncasedOrderedDict(OrderedDict):
    """ Case In-Sensitive Ordered Dictionary.

        Keys are lowered, values are left unchanged.
        Even expressions such as: "string in dict" or "if in dict"
            have appropriate behavior, checking in an uncased way, but returning
            remembered, cased strings when iterated across, while using
            "dict.keys()" will instead will yield only the lower-case versions
            of the keys, as will dict.items()! While this gives flexibility,
            this requires care to not confuse the two!
        NOTE: While this works with multi-keys, any of which may be strings,
            it will NOT see data inside structures any differently.
        NOTE: Takes more memory than a normal dictionary, and is slower.
    """
    def __init__(self, *args, **kwargs): # Necessary only for remembering cases.
        self.cases = {}
        super(UncasedOrderedDict, self).__init__(*args, **kwargs)
        # Since child setitem is called by parent, no need to adjust cases after
        #   initialization from explicit {}.

    @staticmethod
    def simple(k):
        if type(k) == tuple:
            return tuple(e.lower() if isstring(e) else e for e in k)
        else: return k.lower() if isstring(k) else k
    
    def __getitem__(self, k):
        return super(UncasedOrderedDict, self).__getitem__(self.simple(k))
    
    def __setitem__(self, k, v): # If not remember cases, need only last line.
        key = self.simple(k)
        if key not in self.cases: self.cases[key] = k
        return super(UncasedOrderedDict, self).__setitem__(key, v)
    
    def __contains__(self, k):
        return super(UncasedOrderedDict, self).__contains__(k.lower())

    def __iter__(self): # Only necessary for remembering cases.
        for k in self.keys():
            yield self.cases[k]



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
# Distances Class                                                              #
#------------------------------------------------------------------------------#

class Distances:
    """
    A class for distance and neighbor calculations and operations on an
        embedding space.
    
    We do this because in very large spaces we cannot simply find a distance
        matrix, unless you have huge amounts of ram on hand, and any given
        measurement may not be needed many times.
    """

    PARALLELIZATION_BOUND = 3000

    def __init__(self, embeddings, metric_str, metric_fn, print_fn,
            auto_print=True, make_distance_matrix=False,
            make_kth_neighbors=[], parallel_count=None, **metric_args):
        """NOTE: Distance matrix calculation is not parallelized."""
        
        self.space = embeddings
        self.metric_str = metric_str
        self.metric = metric_fn
        self._print = print_fn
        self.auto_print = auto_print
        self.metric_args = metric_args
        self.make_distance_matrix = make_distance_matrix
        self.make_kth_neighbors = [ # Convert all k to positive
            k if k >= 0 else k + len(self.space) for k in make_kth_neighbors \
            if k != 0] # Drop zero. Zeroth-nearest neighbor is self, except for
                       #    objects not in the space.
        # Note the default is to not assume the user needs neighbors at all,
        #   but the Analyst overrides this, giving [-1, 1, 2]. See Analyst.

        self.distance_matrix = None
        self.neighbors = None
        self.neighbors_dist = None
        self.parallel_count = parallel_count
        if self.parallel_count == None:
            self.parallel_count = psutil.cpu_count()
            if len(self.space) <= Distances.PARALLELIZATION_BOUND:
                self.parallel_count = 1

        # NOTE: The Distances class conveniently delays computation of distance
        #   matrix and all the requested neighbors until they are requested
        #   for the first time, in case there are no default evaluators and
        #   no others need them, since some clustering algorithms have it
        #   built-in to recompute all of this internally.

    def get_distance_matrix(self):
        """
        DISTANCE MATRIX CALCULATION, if required:

        Any metric string recognized by scipy will work, or any valid
            function. However, any function, including scipy's, will be FAR
            slower than putting in the string representation of a recognized
            scipy function, so that scipy knows exactly what it is.

        NOTE: This is a condensed distance matrix. Use condensed_index.
        """

        if self.make_distance_matrix and self.distance_matrix is None:
            try:
                self._print(u"Acquainting the Species",
                    u"Computing Distance Matrix")
                self.distance_matrix = sp.distance.pdist(
                    self.space,
                    self.metric_str if self.metric_str != None else self.metric,
                    **self.metric_args)
                return self.distance_matrix
                # Note that our distance matrix is actually a condensed one, or
                #   a distance vector, so we have to use some special indexing.
                #   Converting to squareform would take ~twice as much memory.
            except: # NOTE: catching a memory error probably won't work!
                traceback.print_exc()
                print("FAILED TO MAKE DISTANCE MATRIX; "
                    "PROBABLY NOT ENOUGH MEMORY!")
                self.make_distance_matrix = False
        return None # If failed or if not supposed to make distance matrix

    def condensed_index(self, i, j):
        """Convert indeces to condensed distance vector index"""
        if i == j: return -1 # Not here; we have to check for this one anyway.
        if i < j: i, j = j, i
        return len(self.space)*j - j*(j+1)//2 + i - 1 - j

    def metric_in_model(self, i, j):
        """Fast metric that takes advantage of distance matrix if we have one"""
        if i == j: return 0.0
        d = self.get_distance_matrix()
        if d is not None:
            return d[self.condensed_index(i, j)]
        else:
            return self.metric(self.space[i], self.space[j], **self.metric_args)

    def distances_from(self, index):
        """
        Returns a row of the distance matrix, or the equivalent
        
        Won't take more than one at a time, because a request for all of them
            could cause memory problems. Also then sp.squareform(sp.pdist(...))
            would be more efficient than this if you need all of them anyway.
        """
        d = self.get_distance_matrix()
        if d is not None:
            indeces = [self.condensed_index(index, j) \
                for j in range(len(self.space))]
            distances = d[indeces]
            #distances[np.nonzero(indeces < 0)[0]] = 0.
            distances[index] = 0.
            return distances
        else:
            return sp.distance.cdist(
                np.atleast_2d(self.space[index]),
                self.space,
                self.metric_str if self.metric_str != None else self.metric,
                **self.metric_args).squeeze()

    def neighbor_k(self, index, k):
        """
        Gets the kth neighbor of obj.
        
        Negatives for furthest, 0 for self, positive for nearest.
            If not in_model, 0 is nearest.
        Takes and returns objects in index form.

        NOTE: Be sparing in using this on k not included from the start,
            as these will be heavy calculations because they are not stored!
        """
        try:
            return self.kth_neighbors(k)[index]
        except:
            d = self.distances_from(index)
            return np.argpartition(d, k)[k]

    def neighbors_of(self, index):
        """
            Gets all computed neighbors of index, in order of closeness,
                (zeroth is self). Will get ALL neighbors of index;
            NOTE: This is very slow!
        """
        d = self.distances_from(index)
        return np.argsort(d)

    def distances_from_arbitrary(self, vector):
        """
        Takes a vector not in the model and finds its distance to every obj
            in the model, taking advantage of scipy's optimizations.
        NOTE: results are not stored, so recomputes every time.
        """
        return sp.distance.cdist(
            np.atleast_2d(vector),
            self.space,
            self.metric_str if self.metric_str != None else self.metric,
            **self.metric_args).squeeze()

    def neighbor_k_of_arbitrary(self, vector, k):
        """Here 0 is closest. This is slow!"""
        if k < 0: k += len(self.space) # Convert all k to positive
        d = self.distances_from_arbitrary(vector)
        return np.argpartition(d, k)[k]
            
    def neighbors_of_arbitrary(self, vector, indeces=None):
        """
        Takes a vector NOT IN THE MODEL and finds its distance to every obj
          in the model, returning a 1D array of indeces (not vectors!)
        NOTE: results are not stored, so recomputes every time.
        """
        #if self.metric_str != u"cosine":
        distances = self.distances_from_arbitrary(vector)
        if indeces is None:
            return np.argsort(distances)
        else:
            return np.argpartition(distances, indeces)[indeces]
        # Includes an extra optimization for the common case
        #   that metric is cosine similarity. # REMOVED FOR INDECES OPTIMIZATION
        # else:
        #     distances = np.dot(self.space, np.array([vector]).T.squeeze())
        #     return distances.argsort()[::-1]

    def nearest_to_arbitrary(self, vector):
        """
        Takes in a vector and returns the index of the nearest object in the
            space. If given something IN the space, will return the obj given,
            (again by index,) because that is the nearest.
        """
        if self.metric_str != u"cosine":
            return self.space[np.argmin(self.distances_from_arbitrary(vector))]
        else: # Optimization for cosine similarity:
            return self.space[np.argmax(np.dot(
                self.space, np.array([vector]).T.squeeze()))]

    def kth_neighbors_dist(self, k):
        """
        Returns the kth neighbor's dist of all objects in the space.
        Use negative for furthest, 0 for self, positive for nearest.
        MUCH SLOWER if k not in self.make_kth_neighbors.
        """
        if k == 0: return np.zeros(len(self.space)) # pointless
        if k < 0: k += len(self.space) # Convert all k to positive
        assert k in self.make_kth_neighbors

        # Compute neighbors if haven't before:
        if self.neighbors_dist is None:
            self.kth_neighbors(k)

        return self.neighbors_dist[k]

    def kth_neighbors(self, k):
        """
        Returns the indeces of all objects' kth neighbor.
        Use negative for furthest, 0 for self, positive for nearest.
        MUCH SLOWER if k not in self.make_kth_neighbors.
        """
        if k == 0: return range(len(self.space)) # pointless
        if k < 0: k += len(self.space) # Convert all k to positive
        assert k in self.make_kth_neighbors

        # NEIGHBOR CALCULATIONS if haven't before:
        if self.neighbors is None:

            self.neighbors = {} # dicts keyed to the k we're told to calculate
            self.neighbors_dist = {}

            self._print(u"Setting the Ship's Computer",
                u"Allocating Space for Neighbor Matrices")
            for kth in self.make_kth_neighbors: # Allocate empty arrays
                self.neighbors[kth] = np.empty(len(self.space), dtype=np.uint64)
                self.neighbors_dist[kth] = \
                    np.empty(len(self.space), dtype=np.float64)

            if -1 in self.make_kth_neighbors: # Print stuff
                self._print(u"Misconstruing Relations")
            if 2 in self.make_kth_neighbors:
                self._print(u"Obfuscating Dynastic Ties")
            if 1 in self.make_kth_neighbors:
                self._print(u"Forming Alliances", u"Finding Nearest Neighbors")
            
            if self.parallel_count > 1:
                # PARALLELIZATION TO FILL IN NEIGHBORS:

                print("")
                try: ray.init() #TODO: reduce odd printing in each ray.init() by checking in globals(), and test UI to make sure later inits are broken or not...
                except: pass

                @ray.remote
                def neighbor_row(i, space, distance_matrix, len_space,
                        metric, metric_str, make_kth, metric_args):

                    def condensed(i, j):
                        if i == j: return -1
                        if i < j: i, j = j, i
                        return len_space*j - j*(j+1)//2 + i - 1 - j

                    if distance_matrix is not None:
                        indeces = [condensed(i, j) for j in range(len_space)]
                        distances = distance_matrix[indeces]
                        #distances[np.nonzero(indeces < 0)[0]] = 0.
                        distances[i] = 0.
                    else:
                        distances = sp.distance.cdist(
                            np.atleast_2d(space[i]),
                            space,
                            metric_str if metric_str != None else metric,
                            **metric_args).squeeze()
                    ordering = np.argpartition(distances, make_kth)

                    neighbors = np.empty(len(make_kth))
                    neighbors_dist = np.empty(len(make_kth))

                    for j, kth in enumerate(make_kth):
                        neighbors[j] = ordering[kth]
                        neighbors_dist[j] = distances[ordering[kth]]
                    
                    return i, neighbors, neighbors_dist
                    
                # Shove data into the Object Store:
                dm = self.get_distance_matrix()
                distance_matrix_id = ray.put(dm)
                space_id = ray.put(self.space if dm is None else None)
                metric_str_id = ray.put(self.metric_str)
                metric_id = ray.put(self.metric)
                metric_args_id = ray.put(self.metric_args)
                make_kth_id = ray.put(np.array(self.make_kth_neighbors))

                # Start the first few remote processes/threads:
                remaining_ids = [neighbor_row.remote(
                        i,
                        space_id,
                        distance_matrix_id,
                        len(self.space),
                        metric_id,
                        metric_str_id,
                        make_kth_id,
                        metric_args_id)
                    for i in range(min(len(self.space), self.parallel_count))]

                # Compute:
                for i in tqdm(range(len(self.space)),
                        disable=not self.auto_print):
                    # Using ray.wait allows us to make a progress bar:
                    ready_ids, remaining_ids = ray.wait(remaining_ids)
                    tup = ray.get(ready_ids[0])
                    # Add a new job:
                    if i + self.parallel_count < len(self.space):
                        remaining_ids.append(neighbor_row.remote(
                            i + self.parallel_count, space_id,
                            distance_matrix_id, len(self.space), metric_id,
                            metric_str_id, make_kth_id, metric_args_id))
                    # Process this one's result and fill in data:
                    i, nbrs, nbrs_d = tup
                    for j, kth in enumerate(self.make_kth_neighbors):
                        self.neighbors[kth][i] = nbrs[j]
                        self.neighbors_dist[kth][i] = nbrs_d[j]

            else:
                # ORIGINAL; ALTERNATIVE TO PARALLELIZATION:

                # Filling in neighbors - this may take a long time...
                for i in tqdm(range(len(self.space)),
                        disable=(not self.auto_print)):
                    d = self.distances_from(i)
                    ordering = np.argpartition(d, self.make_kth_neighbors)
                    for kth in self.make_kth_neighbors:
                        self.neighbors[kth][i] = ordering[kth]
                        self.neighbors_dist[kth][i] = d[ordering[kth]]

        return self.neighbors[k]



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
        u"KMeans",
        u"Nodes",
        u"Extremities",
        u"Supernodes",
        u"Hubs",
        u"Nuclei",
        #u"Chains",
        #u"NCC",
        #u"LNCC",
        #u"Anti-hubs",
    ]

    def __init__(self, embeddings=None, strings=None,
        encoder=None, decoder=None, metric=u"cosine", evaluators=[u"All"],
        auto_print=True, desc=None, evaluate=True, make_distance_matrix=False,
        make_kth_neighbors=[-1, 1, 2], parallel_count=None, auto_save=False,
        file_name=None, over_write=False, **metric_args):
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
            evaluate -- whether or not to run the analysis right away.
                Typically always True, unless you want to iteratively add
                evaluators or something.
            make_distance_matrix -- whether or not the user thinks they have
                enough memory to compute a distance matrix, which may be faster
                than the alternative. Beware, this is an n^2 / 2 algorithm for
                memory, meaning that a vocabulary of 100,000 will try to store
                approx. 5,000,000,000 floats, plus some overhead. Do the math.
                NOTE: Distance matrix calculation is not parallelized.
            make_kth_neighbors -- a list of which neighbors to store. Memory
                taken will be relative to len*n, so many neighbors for each
                object in the space. Built-ins require [-1, 1, 2], which are
                furthest, nearest, and 2nd-nearest neighbors, respectively.
                Failure to include one or more of these will result in much
                slower calculations of many Evaluators!
            parallel_count -- How many parallel threads to use in computing
                heavier, parallelizeable parts.
                If None, will try to choose based on CPU count and size of given
                embedding space (if under 3000, or less than 2 cores/threads,
                chooses not to for efficiency's sake). Note the overhead may
                make 2 less efficient than 1 in general. You can use a larger
                number than cores/threads, and it will compute that many at a
                time as by use of multiple tasks on each. Likewise a smaller
                number will not constrain itself to n specific CPUS, but will
                shuffle that many jobs between all those available.
            auto_save -- whether or not to save automatically after computing.
            file_name -- name of file to use. If blank, will base it on desc.
            over_write -- whether or not to overwrite existing file of same name
                when saving. Note we will overwrite our own, but not others.
            metric_args -- these are extra arguments to be given to metric.
        """

        self.auto_print = auto_print
        print(u"")
        self._print(u"Asking the Grand Question",
            u"What is the Purpose of this Embedding?")
        self.description = str(desc)

        # Set the file_name for this analyst:
        self.auto_save = auto_save
        self.over_write = over_write
        if file_name is not None:
            self.file_name = file_name
        else:
            self.file_name = _file_extension(self.description)
            if not self.over_write:
                i = 0
                name_only = self.file_name
                number = u""
                ext = u""
                for i, c in enumerate(self.file_name):
                    if c == u'.':
                        name_only = self.file_name[:i]
                        ext = self.file_name[i+1:]
                        break
                j = 1
                while(os.path.isfile(name_only + number + u'.' + ext)):
                    number = u'(' + str(j) + u')'
                    j += 1
                self.file_name = name_only + number + u'.' + ext
        
        # Find and store a callable version of the given metric:
        self._print(u"Laying the Laws of Physics", u"Setting the Metric")
        if callable(metric):
            self.metric = metric   # Pay attention to how you use the metric!
            self.metric_str = None #    It is for vectors. For indeces or other,
        else:                      #    you need to use metric_in_model!
            try:
                self.metric_str = metric
                #self.metric_str = str(sp.distance._METRIC_ALIAS[metric])
                #   Only worked in python2.
                self.metric = sp.distance._TEST_METRICS[
                    u"test_" + self.metric_str]
            except:
                traceback.print_exc()
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
                self._print(u"Stretching Fabric of Space and Time",
                    u"Finding Embeddings")
                self.space = np.array([encoder(w) for w in
                    tqdm(strings, disable=not self.auto_print)])
                # TODO: use np.vectorize?
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
        self.vec_to_s = {} # Declared for _decode, but unused if decoder given.
        if decoder is None:
            self._print(u"Mapping the Emptiness", u"Mapping New Decoder")
            for i in trange(len(self.space), disable=(not self.auto_print)):
                self.vec_to_s[str(self.space[i].tolist())] = self.strings[i]
            #self.decode = self.vec_to_s.__getitem__
            self.decode = self._decode
        else:
            assert callable(decoder)
            self.decode = decoder

        # A separate Map for quick string indexing:
        self.s_to_ix = {}
        self._print(u"Indexing Planets", u"Making String-Index Mappings")
        for ix, s in enumerate(self.strings):
            self.s_to_ix[s] = ix
        # NOTE: I'm not making vec_to_ix because I assume that the length of the
        #   vectors makes it more efficient to use s_to_ix[decode(vec)].

        # Data
        self.graph_info = []
        self.categories = []
        self.evaluators = []
        self.category_lists = []
        self.add_evaluators(*evaluators)
        
        self.make_distance_matrix = make_distance_matrix
        self.make_kth_neighbors = make_kth_neighbors
        self.D = None
        self.parallel_count = parallel_count # We'll get the right value from D.

        # Run Analyses:
        if evaluate:
            self.analysis(
                print_report=self.auto_print, recalculate=[])


    def _decode(self, vec):
        return self.vec_to_s[str(vec.tolist())]

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
    def as_index(self, obj, in_model=True):
        """
        These work on objects not in the model, but this requires
            in_model=False, and of course only returns the nearest object.
        """
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


    def metric_in_model(self, obj1, obj2):
        """
        Superfast metric function IF DISTANCE MATRIX COMPUTED,
            for objects within the model only.
        Note: generic types.
        """
        return self.D.metric_in_model(
            self.as_index(obj1, in_model=True),
            self.as_index(obj2, in_model=True))

    def neighbor_k(self, obj, k, in_model=True):
        """
        Gets the kth neighbor of obj. Negatives for furthest, 0 for self,
            positive for nearest. If not in_model, 0 is nearest.
        Attempts to return same type given, ie: index, string, or vector.
        Ensures nbrs will be calculated before, without being recalculated.
        """
        if in_model:
            i = self.D.neighbor_k(self.as_index(obj), k)
            if isstring(obj):
                return self.strings[i]
            try:
                int(obj)
                return i
            except: return self.space[i]
        else:
            # Note that if not in_model, we require obj to be a vector.
            self.space[self.D.neighbor_k_of_arbitrary(obj, k)]

    def nearest(self, obj, in_model=True):
        if in_model:
            return self.neighbor_k(obj, 1, in_model=True)
        else:
            # Note that if not in_model, we require obj to be a vector.
            return self.D.nearest_to_arbitrary(obj)

    def downstream(self, obj, start_neighbor_k=0, give_path=False):
        """
        Computes the downstream nearest neighbor, and lists the path if
            asked, starting from obj's kth-nearest neighbor,
            then going from one NEAREST neighbor to another until we start
            to repeat (having reached a node). Thus the last two in the list
            make a node. If you start from k=0, obj will be included.
            k=1 is same, but without obj.
        Note: non type-specific, and returns same type as given.
        Note: should be faster if path not kept.
        Note: probably less efficient if you use straight vectors,
            because of equality checks. Maybe most efficient for indeces?
        """
        path = None
        path=[]
        current = self.neighbor_k(obj, start_neighbor_k, in_model=True)
        while current not in path: # More efficient than overhead for a set?
            path.append(current)
            current = self.nearest(current)
        if give_path: return path
        else: return tuple(path[-2:]) # Right one is furthest downstream.


    def find_evaluator(self, category, force_creation=False):
        """
        Finds an evaluator beloning to this Analyst by its categorical name.

        force_creation: whether or not to create a default evaluator for
            built-ins, AND TO ADD IT to the Analyst.

        NOTE: This function does NOT force the evaluator to pre-calculate!
        NOTE: Since categories SHOULD be unique among evaluators,
            this function will only return the first match it finds. Or None.
        """
        cat = category.lower()
        for e in self.evaluators:
            if str(e.CATEGORY.lower()) == cat: return e
        for e in self.evaluators: # To find modified categories, ex. '5-Hubs':
            # But note, it picks the first match it finds, so this will be a
            #   problem if you've told it to compute two versions of a category.
            if cat in str(e.CATEGORY.lower()):
                print("DECISION: Found no category '" + category + "'; " \
                    "using '" + e.CATEGORY + "' instead.")
                return e
        if force_creation: 
            e = Analyst.make_default_evaluator(str(category))
            self.add_evaluators(e)
            return e
        return None

    def get_clusters(self, category, force_creation=False):
        """
        Gets list of clusters from evaluator with given name (category)
        
        Returns None if not found or has not get_clusters method.
        """
        try:
            return self.find_evaluator(
                category, force_creation=force_creation).get_clusters()
        except:
            traceback.print_exc()
            return None

    @staticmethod
    def make_default_evaluator(category):
        """
        Makes built-in Evaluators with defaulted values

        Note: Can take some parameterization, such as "Nodal 10-Hubs",
            or "2Hubs". "Hubs" with no number defaults to "Nodal 4-Hubs".
        """
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
        elif cat == u"nuclei":
            return NucleusClusterizer()
        elif cat[-6:] == "kmeans":
            try:
                if cat[-7] == '-': k = int(cat[:-7])
                cat = str(k) + "-KMeans"
                return KMeansClusterizer(category=cat, k_or_guess=k)
            except:
                return KMeansClusterizer(category="KMeans", k_or_guess=None)
        # ADD STUFF!!
        else:
            print(u"UNRECOGNIZED BUILT-IN EVALUTATOR '"
                + category + u"'! SKIPPING IT.")
            return None

    def add_evaluators(self, *args):
        """Adds new evaluators to the Analyst"""

        def rename_evaluator(evaluator):
            """Helper function"""
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

    def analysis(self, print_report=True, recalculate=[]):
        """
        Main Analysis function of the Analyst

        Preps a Distances object, then runs evaluate method on each Evaluator
            the analyst has. Won't REevaluate any but those whose categories
            are listed.
        Even those it doesn't recalculate, it will still get their data and
            update its own listings in case something has changed.
        Will only recalculate distances and neighbors if "all" or "All"
            in recalculate.
        """

        if self.D is None or "All" in recalculate or "all" in recalculate:
            # Delayed creation of this object till now because if
            #   make_distance_matrix is True, then Distances will immediately
            #   begin computing its distance matrix, preventing the user from
            #   soon doing whatever it was they delayed analysis for in the
            #   first place.
            self.D = Distances(
                embeddings=self.space,
                metric_str=self.metric_str,
                metric_fn=self.metric,
                print_fn=self._print,
                auto_print=self.auto_print,
                make_distance_matrix=self.make_distance_matrix,
                make_kth_neighbors=self.make_kth_neighbors,
                parallel_count=self.parallel_count,
                **self.metric_args)

            self.parallel_count = self.D.parallel_count

        # If want to recalculate all:
        if "All" in recalculate or "all" in recalculate:
            recalculate = [e.CATEGORY for e in self.evaluators]

        # Run the Evaluations:
        for evaluator in self.evaluators:
            try:
                stats_dict, starred, category = evaluator.calculate(
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

                    metric_in_model_fn=self.metric_in_model,

                    generic_neighbor_k_fn=self.neighbor_k,
                    generic_nearest_fn=self.nearest,
                    kth_neighbors_ix_fn=self.D.kth_neighbors,
                    kth_neighbors_dist_fn=self.D.kth_neighbors_dist,
                    arbitrary_dist_fn=self.D.distances_from_arbitrary,
                    arbitrary_neighbors_fn=self.D.neighbors_of_arbitrary,
                    distances_from_ix_fn=self.D.distances_from,
                    neighbors_of_ix_fn=self.D.neighbors_of,
                    condensed_dist_matrix_fn=self.D.get_distance_matrix,
                    condensed_ix_fn=self.D.condensed_index,

                    downstream_fn=self.downstream,
                    evaluator_list=self.evaluators,
                    find_evaluator_fn=self.find_evaluator,
                    get_clusters_fn=self.get_clusters,
                    make_kth_neighbors=self.make_kth_neighbors,
                    simulate_cluster_fn=simulate_cluster,
                    parallel_count=self.parallel_count,
                    make_dist_matrix=self.make_distance_matrix,
                )

                # The below compatibilities should be unnecessary because both
                #   keys and starred come from same source, thus same version.
                #   Also, they're just going to be printed later, so no others
                #   need it either.
                #starred = map(str, starred)
                #key = str(key)
                #category = str(category)

                for (key, value) in stats_dict.items():
                    self._add_info(value, category, key, key in starred)

            except:
                traceback.print_exc()
                print(u"ERROR IN CALCULATION OF %s."# DOES YOUR EVALUATOR "
                    #u"INHERIT FROM AN Evaluator CLASS?"
                    % evaluator.CATEGORY)

        # For convenience, we print the report before taking time to save;
        #   this behavior is protected from losing data with a try block in the
        #   print_report function:
        if print_report: self.print_report()
        
        if self.auto_save:
            if self.file_name != None:
                self.save()
            else: print("CANNOT AUTO-SAVE WITHOUT HAVING BEEN SAVED AT LEAST "
                "ONCE; NO FILENAME.")


    # SPECIFICS INSPECTION:

    #TODO!


    # GRAPHING:

    def graph_hist(self, hist_key, bins=64, **kwargs):
        """
        Creates a histogram according to key printed in report.
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

    def graph_bar(self, description, categories=None, **kwargs):
        """
        Creates a bar graph of the given stat (description) across multiple
            Evaluators. If categories is None, will look across all for that
            stat and only use evaluators which contain that stat;
            otherwise will only look in ones in the given list.
        Can take evaluator references instead of their categories.
        **kwargs is for matplotlib.pyplot.bar
        """
        if categories is None:
            cats = [e.CATEGORY for e in self.evaluators \
                if description in e.stats_dict]
        else:
            cats = []
            for c in categories:
                cats.append(c) if isstring(c) else cats.append(c.CATEGORY)
        
        vals = [self.get_stat(c, description) for c in cats]
        vals = [v if v is not None else 0 for v in vals]
        x = range(len(vals))

        plt.bar(x, vals, align='center', **kwargs)
        plt.xticks(x, cats)
        plt.setp(plt.gca().get_xticklabels(), rotation=-45,
            horizontalalignment='left')
        plt.ylabel(description)
        plt.title(description.title() + " Comparison in " + self.description)
        plt.tight_layout()
        plt.show()

    def graph_bar_multi(self, descriptions, categories=None,
            group_by_stat=False, **kwargs):
        """
        Creates a multi- bar graph comparing the given stats (descriptions)
            across multiple evaluators.
            If categories is None, will use every evaluator containing the
            FIRST description in its stat_dict.
        Can take evaluator references instead of their categories.
        **kwargs is for matplotlib.pyplot.bar
        """
        if categories is None:
            cats = [e.CATEGORY for e in self.evaluators \
                if descriptions[0] in e.stats_dict]
        else:
            cats = []
            for c in categories:
                cats.append(c) if isstring(c) else cats.append(c.CATEGORY)

        # one row per stat, one column per evaluator.
        vals = [[self.get_stat(c, d) for d in descriptions] for c in cats]
        vals = np.array([
            [v if v is not None else 0 for v in row] for row in vals])
        ticks = cats
        legend = descriptions

        if group_by_stat:
            legend, ticks = ticks, legend
        else:
            vals = vals.T

        x = np.arange(len(ticks))
        width = 1.0 / (len(legend) + 1)
        for i in range(len(legend)):
            plt.bar(x + width * (i + 1), vals[i],
                width=width, label=legend[i], **kwargs)
            
        plt.ylabel("Value")
        plt.xlabel("Statistic" if group_by_stat else "Category")
        plt.title("Comparison on Various Stats in " + self.description)
        plt.xticks(x + 0.5, ticks)
        plt.setp(plt.gca().get_xticklabels(), rotation=-45,
            horizontalalignment='left')
        plt.legend(loc='best')
        try:
            plt.tight_layout()
        except:
            print("Could not use tight_layout.")
        plt.show()

    def graph_evaluators(self, description, eval_array,
            xticks=None, legend=None, xlabel=None, transpose=None, **kwargs):
        """
        Compares evaluators from multiple groups against each other,
            on a single stat (description), by creating a multi- bar graph.

        eval_array: a list of lists or 2d array of evaluators or
            their categories, or a mix.
            Each ROW in the input array corresponds to a group of bars!
        xticks: labels along the x axis.
        legend: labels for each bar in a given group, shown in map legend.
        transpose: should we transpose eval_array for you? None will try
            to match shape of xticks and legend.
        **kwargs: for matplotlib.pyplot.bar
        """
        cats = [[c if isstring(c) else c.CATEGORY for c in row] \
            for row in eval_array]

        # one row per group, one column per bar.
        vals = [[self.get_stat(c, description) for c in row] for row in cats]
        vals = np.array([
            [v if v is not None else 0 for v in row] for row in vals])

        if transpose is None:
            transpose = len(eval_array) == len(xticks)
        if transpose:
            vals = vals.T

        x = np.arange(len(xticks))
        width = 1.0 / (len(legend) + 1)
        for i in range(len(legend)):
            plt.bar(x + width * (i + 1), vals[i],
                width=width, label=legend[i], **kwargs)
            
        plt.ylabel(description)
        if xlabel is not None: plt.xlabel(xlabel)
        plt.title(
            description + " Comparison on Evaluators in " + self.description)
        plt.xticks(x + 0.5, xticks)
        plt.setp(plt.gca().get_xticklabels(), rotation=-45,
            horizontalalignment='left')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def graph_comparison(ana_list, category, description, **kwargs):
        """
        Creates a bar graph for a given stat across multiple Analysts.

        category: can be string or reference to an evaluator.
        **kwargs is for matplotlib.pyplot.bar
        """
        cat = category if isstring(category) else category.CATEGORY

        vals = [a.get_stat(cat, description) for a in ana_list]
        vals = [v if v is not None else 0 for v in vals]
        labels = [a.description for a in ana_list]
        x = range(len(vals))

        plt.bar(x, vals, align='center', **kwargs)
        plt.xticks(x, labels)
        plt.setp(plt.gca().get_xticklabels(), rotation=-45,
            horizontalalignment='left')
        plt.ylabel(cat + ": " + description)
        plt.title("Analyst Comparison")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def graph_comparison_multi(ana_list, cat_desc_pairs, group_by_stat=True,
            **kwargs):
        """
        Creates a multi-bar graph showing stats side-by-side for
            multiple Analysts.

        Inputs:
            ana_list: a list of analysts.
            group_by_stat: True means bars will be grouped by stat,
                False means bars will be grouped by Analyst.
            cat_desc_pairs: list of tuples, each of the form
                ("Category", "Description").
            **kwargs: extra arguments to pass into matplotlib.pyplot.bar
        """
        categories, descriptions = zip(*cat_desc_pairs)
        # dicts = [[a.get_category_stats(c) for c in categories] \
        #     for a in ana_list]
        # vals = np.array([ # one row per analyst, one column per stat.
        #     [d[descriptions[i]] if descriptions[i] in d else 0 \
        #         for i, d in enumerate(row_of_dicts)] \
        #     for row_of_dicts in dicts])
        xlabel = None
        ticks = [t[0] + ": " + t[1] for t in cat_desc_pairs]
        for i, c in enumerate(categories):
            if c != categories[0]: break
            if i == len(categories) - 1:
                xlabel = "Category: " + c
                ticks = descriptions
        for i, d in enumerate(descriptions):
            if d != descriptions[0]: break
            if i == len(categories) - 1:
                xlabel = "Statistic: " + d
                ticks = categories

        vals = [
            [a.get_stat(c, descriptions[i]) for a in ana_list] \
            for i, c in enumerate(categories)]
        vals = np.array([
            [v if v is not None else 0 for v in row] \
            for row in vals])
        legend = [a.description for a in ana_list]

        if not group_by_stat:
            legend, ticks = ticks, legend
        else:
            vals = vals.T

        x = np.arange(len(ticks))
        width = 1.0 / (len(legend) + 1)
        for i in range(len(legend)):
            plt.bar(x + width * (i + 1), vals[i],
                width=width, label=legend[i], **kwargs)
            
        plt.ylabel("Value")
        if xlabel is not None: plt.xlabel(xlabel)
        plt.title("Analyst Comparison on Various Stats")
        plt.xticks(x + 0.5, ticks)
        plt.setp(plt.gca().get_xticklabels(), rotation=-45,
            horizontalalignment='left')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


    # COMPARATIVE:

    @staticmethod
    def _formatit(data, width=10, parentheses=False, start_at=0):
        """Helper function for printing a piece of data in a report"""
        #if parentheses: w = max(9, width)
        #else: w = max(7, width)
        w = max(9, width)
        result = ""
        try:
            if (data is None or type(data) == list or
                    type(data) == type(np.ndarray)):
                data = str(data)

            if isstring(data) or parentheses or not np.isfinite(data):
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
        except:
            traceback.print_exc()
            print("ERROR, Non-reportable data:", data)
        return result

    @staticmethod
    def compare(ana_list, w=10, comparators=[u"default"], report_path=None,
            categories=None):
        """
        Lists side by side the values for each analyst in the list,
            as well as a column for each comparator, run on that stat.

        w: Numbers will have space for w-2 digits, (w-2 because of . and - ).
            Total width will be: (6 + (w + 1)*(an + comp) + len(description))
        comparator:
            empty list: no comparison columns
            callable (Takes list of available values; does NOT need to handle
                strings or None), or a built-in:
            "all": all builtins
            "default:" includes range, curiosity, and odd_one_out
            "std": standard deviation across the stat
            "avg" or "average": average across the stat
            "max" or "maximum": maximum value
            "min" or "minimum": minimum value
            "rng" or "range": max value minus min value
            "weighted_diff": accentuates numbers far from zero
            "curiosity": accentuates differences and uniformity
            "odd_one_out": spikes when one value differs largely from others
        ana_list: a list of analysts. Kinda thought that was clear... :)
        report_path: file path to save report to.
        categories: list of category names, if want to limit what prints.

        Returns: an ordered stat_dict keyed to tuples (Category, Description),
            and a grapher object with multi-histogram information from TODO !!!!!!!!!!!!!!!!!!!!!!!!!!
            the comparison.
        """
        assert len(ana_list) > 0
        ana_list[0]._print(u"Bridging Universes",
            u"Building Comparison & Report")
        print(u"")
        result = u""
        stat_dict = OrderedDict()

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
        result += title + u"\n"

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
        result += title_string.format(*titles) + u"\n"

        # Line Template:
        line_string = u"  {}" + u"{} " * len(ana_list) + u"|" + \
            u"{} " * len(comparisons) + u"|{}{}"
        #graphs = []

        # Helper function; All the work to print one category:
        def get_category(c):
            string = c + u":\n"
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
                numbers = [d for d in datalist if (
                    not isstring(d) and d is not None and np.isscalar(d))]
                if numbers is [] or is_hist:
                    comps = [u""] * len(comparisons)
                else:
                    try: comps = [comp(numbers) for comp in comparisons]
                    except: comps = [u""] * len(comparisons)
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
                stat_dict[(c, desc)] = data
                data = [star] + data + [star, desc]
                string += line_string.format(*data) + u"\n"
            
            return string

        # Put it all together - loop through cats in order and build result:
        if categories is None:
            used_categories = []
            for a in ana_list:
                for c in a.categories:
                    if c not in used_categories:
                        used_categories.append(c)
                        result += get_category(c)
        else: # If selective choice of categories:
            for c in categories:
                result += get_category(c)
                    
        # Print report, save file:
        print(result)
        if report_path != None:
            with open(report_path, 'w') as f:
                f.write(result)
            
        return stat_dict #,grapher # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    #--------------------------------------------------------------------------#
    # Information Gathering, Reporting, and Saving Functions                   #
    #--------------------------------------------------------------------------#

    def _add_info(self, var, category, description, star=False):
        """Helper function"""
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
        if self.auto_print:
            print("{:<40}".format(u"\r" + str(string) + u"...")
                + ("" if report is None else u"(" + report + u")"))

    def print_report(self, w=10, report_path=None, auto_print=True,
            categories=None):
        self._print(u"Revealing the Grand Plan", u"Printing Report")
        print(u"")
        result = ""
        stat_dict = OrderedDict()
        if categories is None: categories = self.categories
        #   These are the categories we want to keep.
        try:
            if self.description != None:
                result += self.description.upper() + u"\n"
            for i, category in enumerate(self.categories):
                if category in categories:
                    result += category + u":\n"
                    for cat in self.category_lists[i]:
                        stat_dict[(category, cat[0])] = cat[1]
                        result += u"  {}{} {}{}".format(
                            "*" if cat[2] else u" ", # Stars
                            Analyst._formatit(
                                cat[1], w, u"Histogram Key" in cat[0], 3),
                            u"*" if cat[2] else u" ", # Stars
                            cat[0]) + u"\n"

            if auto_print: print(result)
            if report_path != None:
                with open(report_path, 'w') as f:
                    f.write(result)

        except:
            traceback.print_exc()
            print("ERROR IN PRINTING REPORT!")
            print("CATEGORY INFORMATION:")
            print(self.category_lists)
            print("REPORT PATH:", report_path)

        
        # If not printing, will return an ordered dict instead, keyed to tuples:
        #   (category, description)
        if not auto_print: return stat_dict #, grapher TODO.

    def get_category_stats(self, category, stat_dict=None):
        """
        Retrieve a dict containing only stats from desired category, such that
            new_d["Description"] == value

        Works on multi-analyst stat_dicts as well.
        Case in-sensitive.
        """
        d = self.print_report(auto_print=False, categories=[category]) \
            if stat_dict == None else stat_dict
        #d's keys are tuples, (category, description), and vals are stats.
        new_d = OrderedDict()
        for k in d:#.keys():
            if k[0].lower() == category.lower(): new_d[k[1]] = d[k]
        return new_d

    def get_stat(self, category, description):
        """Get a single stat; not case sensitive"""
        try:
            desc = description.lower()
            cat = category.lower()
            for i, c in enumerate(self.categories):
                if c.lower() == cat: break
            for t in self.category_lists[i]:
                if t[0].lower() == desc: return t[1]
        except: pass
        return None

    def save(self, file_name=None):
        """
        Saves self to dill file.

        NOTE: An Analyst with a space of one million 100D vectors and no
            evaluators will pickle to around 7GB because of the quick encoders/
            decoders. Most Evaluators will not add much more, though higher-D
            vectors will take up more hard drive space.
        """
        try:
            f_name = self.file_name if file_name is None else file_name
            if f_name is None:
                f_name = self.description
            #obj._serialize()
            self._print(u"Snapshotting the Universe", u"Saving the Analyst")
            with open(_file_extension(f_name), 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.file_name = f_name
            return True
        except:
            traceback.print_exc()
            return False


# END OF ANALYST CLASS.
################################################################################



# Brief script-like behavior for development, debugging, and testing purposes:
if __name__ == "__main__":
    #import TestSet2D

    raise Exception("USAGE ERROR: analyst module script behabior not defined. "
        "Should be imported as a package.")
