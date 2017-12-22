import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt

class Analyst:
    """
    Interface for toolset for analyzing embedding spaces.

    Use: Generally initialize one analyst instance,
        and access tools in the toolset through it.

    Parameters:
        embeddings -- list of vectors populating the space.
            Must have static indeces.
        metric -- the distance metric used throughout,
            "l2" or "euclidean", "l1", "cosine_similarity",
            or a function object. Defaults to "cosine_similarity".
        encoder -- a callable to convert strings to vectors.
        decoder -- a callable to convert vectors to strings.

    Tools:
        ?
        ?
        ?
    """

    def __init__(self, embeddings, metric="cosine_similarity", encoder=None, decoder=None):
        self.space = embeddings
        if callable(metric): self.metric = metric
        elif metric == "l2" or metric == "euclidean": self.metric = sp.distance.euclidean
        elif metric == "cosine_similarity": self.metric = sp.distance.cosine
        elif metric == "l1": self.metric = sp.distance.cityblock
        else: raise ValueError("'metric' parameter unrecognized and uncallable")

        # Encoder/Decoder Initializations:
        #   While initializing these should theoretically be unnecessary,
        #   failing to do so will limit all inputs to findable types.
        self.encode = encoder # string to vector
        self.decode = decoder # vector to string
        self.s_to_ix = {}
        self.ix_to_s = []
        try:
            self.ix_to_s = [self.decode(vec) for vec in self.space]
            for ix, s in enumerate(self.ix_to_s):
                self.s_to_ix[s] = ix
        except: pass
        
        # Run Analyses:
        self.spatial_analysis()
        self.cluster_analysis()
        self.print_report()


    # Generic type converters for inputs and outputs:
    def index(self, obj):
        if isinstance(obj, basestring): return self.s_to_ix[obj]
        try: return self.s_to_ix[self.decode(obj)]
        except: return int(obj)

    def vector(self, obj):
        if isinstance(obj, basestring): return self.encode(obj)
        try: return self.space[obj]
        except: return obj

    def string(self, obj):
        if isinstance(obj, basestring): return obj
        try: return self.ix_to_s[obj]
        except: return self.decode(obj)


    # General Analyses:
    def spatial_analysis(self):
        pass

    def cluster_analysis(self):
        pass

    def print_report(self):
        pass


    # Specific Functions:
    def rescale(self, theta, alpha=15, power=0.5):
        ''' Rescales based on observed distribution of angles between words
            in a postagged Wikipedia word embedding from BYU PCCL.
            Accepts theta in radians.'''
        return (0.5 + (math.atan((theta*180/np.pi - 90)/alpha)
                         / (2*math.atan(90/alpha))))**power


    def test_angles(self, n, alpha=15, power=0.5):
        dist = [self.rescale(self.s.angle(
                    self.s.get_vector(self.s.model.vocab[int(x)]),
                    self.s.get_vector(self.s.model.vocab[int(2*x)])),
                    alpha, power)
                for x in (np.random.random(n)*len(self.s.model.vocab)/2.0)]
        plt.hist(dist, 90)
        plt.show()

    #def scale_bimodal(self, theta):
    #    deg = theta*180/np.pi
    #    return 0.5 + (self.cbrt((deg-90)) / (2*self.cbrt(90)))