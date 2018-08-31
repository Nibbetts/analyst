import scipy.cluster.vq as sc
import numpy as np

from ..clustertypes.cluster import Cluster
from .clusterizer import Clusterizer



class KMeansClusterizer(Clusterizer, object):
    """
    Simple KMeans Clusterizer.
    Unfortunately, while we can take advantage of numpy speedups, in doing so we
        cannot take advantage of the things the Analyst may have already done,
        such as nearest neighbors or distances. Scipy has, though, parallelized
        this particular algorithm already, though we can't see our progress.
    """

    def __init__(self, category="KMeans", starred=None, k_or_guess=None,
            iter_limit=20, thresh=1e-05, check_finite=False):
        # By default, we choose round(sqrt(len(space))) if no K is specified.
        super(KMeansClusterizer, self).__init__(
            category=category, starred=starred)
        #   To inherit, must call parent init.
        # Inherited:
        # self.clusters = []
        # self.vector_groups = []
        # self.stats_dict = OrderedDict()
        # self.starred = []
        # self.calculated = False
        self.k_or_guess = k_or_guess
        #   Alternatively to specifying k, you can pass in a k by n array, an
        #   array of centroid vectors of length k - this essentially would
        #   cluster the space according you your specifications. Additionally,
        #   you can pass in a list of strings or indeces, and we will convert
        #   them for you.
        self.iter_limit = iter_limit
        self.thresh = thresh
        self.check_finite = check_finite
        # self.kmeans # Not necessary to track, since the Cluster object does
        #   this for us anyway.
        self.distortion = None
        self.distortion_list = None
        self.distortion_groups = None


    def compute_clusters(self, space, show_progress=True, **kwargs):
        # POST: By the time this function finishes, self.vector_groups must be
        #   filled in, as a vector of vectors of vectors,
        #   but self.clusters doesn't need to be filled in yet.
        printer   = kwargs["printer_fn"]
        as_vector = kwargs["as_vector_fn"]

        if self.k_or_guess is None:
            self.k_or_guess = round(len(space)**0.5)
        else:
            try:
                self.k_or_guess = np.array([
                    as_vector(o) for o in self.k_or_guess])
            except:
                pass

        # This may take some time...
        printer("Glossing Over the Rules", "Whitening the Data for KMeans")
        whitened = sc.whiten(space)
        printer("\t'They're more like guidelines, anyway'")
        printer("Centralizing the Powers", "Finding K Centroids")
        # Scipy has already parallelized this part! We just can't see progress.
        codebook, self.distortion = sc.kmeans(
            # Scipy KMeans computation
            whitened,
            self.k_or_guess,
            self.iter_limit,
            self.thresh,
            self.check_finite)
        printer("Dividing and Conquering", "Sorting Vectors by Their Means")
        indeces, self.distortion_list = sc.vq(
            # Scipy KMeans sorting
            whitened,
            codebook,
            check_finite=self.check_finite)

        # Build our collection of vectors:
        self.vector_groups = [[] for _ in range(len(codebook))]
        self.distortion_groups = [[] for _ in range(len(codebook))]
        for i, v in enumerate(space):
            self.vector_groups[indeces[i]].append(v)
            self.distortion_groups[indeces[i]].append(self.distortion_list[i])


    def compute_stats(self, **kwargs):
        # PRE: self.clusters must have been filled in (by vectors_to_clusters).
        # POST: self.stats_dict, self.starred filled in.
        printer = kwargs["printer_fn"]

        self.stats_dict["K"] = len(self.vector_groups)
        #self.stats_dict["KMeans Distortion"] = self.distortion
        #   Was duplicate of Spatial Distortion Avg!
        # Add cluster distortion stats:

        printer("Distorting Reality", "Measuring Cluster Distortion")
        for i, c in enumerate(self.clusters):
            c.stats_dict["Cluster Distortion"] = \
                np.mean(self.distortion_groups[i])
        # Add overall distortion stats:
        self._compute_list_stats(
            self.distortion_list, "Spatial Distortion", self.stats_dict,
            si="dispersion", **kwargs)

        super(KMeansClusterizer, self).compute_stats(**kwargs)
        # Count has been redubbed K:
        self.stats_dict.pop("Count")

        self.add_star("Population Max")
        self.add_star("Population Min")
        self.add_star("Dispersion Avg")
        self.add_star("Spatial Distortion Avg")
        self.add_star("Cluster Distortion Avg")
        self.add_star("Cluster Distortion Range")
        self.add_star("SI Spatial Distortion Avg")
        self.add_star("SI Cluster Distortion Avg")
        self.add_star("SI Cluster Distortion Range")
        self.add_star("K")
