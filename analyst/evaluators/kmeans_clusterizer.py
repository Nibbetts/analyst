import scipy.cluster.vq as sc

from ..clustertypes.cluster import Cluster
from .clusterizer import Clusterizer



class KMeansClusterizer(Clusterizer, object):
    """
    Simple KMeans Clusterizer.
    Unfortunately, while we can take advantage of numpy speedups, in doing so we
        cannot take advantage of the things the Analyst may have already done,
        such as nearest neighbors or distances.
    """

    def __init__(self, category, k_or_guess, iter_limit=20, thresh=1e-05,
            check_finite=False, starred=None):
        super(KMeansClusterizer, self).__init__(
            category=category, starred=starred)
        #   To inherit, must call parent init.
        # Inherited:
        # self.clusters = []
        # self.vector_groups = []
        # self.data_dict = OrderedDict()
        # self.starred = []
        # self.calculated = False
        self.k_or_guess = k_or_guess
        self.iter_limit = iter_limit
        self.thresh = thresh
        self.check_finite = check_finite
        # self.kmeans # Not necessary to track, since the Cluster object does
        #   this for us anyway.
        self.distortion = None


    def compute_clusters(self, space, show_progress=True, **kwargs):
        # POST: By the time this function finishes, self.vector_groups must be
        #   filled in, as a vector of vectors of vectors,
        #   but self.clusters doesn't need to be.

        # This may take some time...
        codebook, self.distortion = sc.kmeans(
            sc.whiten(space),
            self.k_or_guess,
            self.iter_limit,
            self.thresh,
            self.check_finite)

        # TODO! FINISH!


    def compute_stats(self, **kwargs):
        self.data_dict["K"] = None
        self.data_dict["Distortion"] = self.distortion

        super(KMeansClusterizer, self).compute_stats(**kwargs)
        # Rename Count to K:
        self.data_dict["K"] = self.data_dict.pop("Count")
