from abc import abstractmethod
from tqdm import tqdm
import numpy as np

from .analogizer import Analogizer
from .analogizer import WORD_ANALOGY_SEP
from .analogizer import WORD_ITEM_SEP
import analyst



class FrequencyAnalogizer(Analogizer, object):
    """
    Frequency-Weighted Linear Offset Analogy,
        Of the form used by Mikolov et al, only scales the perceived distance
        to the top few potential results by a measure of how well-matched their
        commonality is to the expected result's commonality.
    
    NOTE: This will only work if the vectors in the space are ordered by their
        frequency or commonality (most frequent first)!
    """

    def __init__(self, category="Frequency Matching Analogies", n=3, alpha=0.3,
            starred=None, analogies_path=None, analogies=None,
            analogy_vectors=None, analogy_sep=WORD_ANALOGY_SEP):
        super(FrequencyAnalogizer, self).__init__(
            category=category, starred=starred, analogies_path=analogies_path,
            analogies=analogies, analogy_vectors=analogy_vectors,
            analogy_sep=analogy_sep, item_sep=item_sep)
        # Examine n closest as our top choices:
        self.n = n
        # How important is frequency compared to location:
        self.alpha = alpha

    # OVERRIDEABLE
    def analogy(self, string_a, string_b, string_c, *args, **kwargs):
        # string_a, string_b, and string_c are the given analogy items.
        #   string_d is intended to be witheld.
        # RETURNS: string approximation for d, and vector approximation for d.
        #   Both are returned since some analogy algorithms may not naively
        #   choose the nearest possible decode, and the scoring is done in
        #   compute_stats.

        encode      = kwargs["encoder_fn"]
        nbrs_of     = kwargs["arbitrary_neighbors_fn"]
        metric      = kwargs["metric_fn"]
        strings     = kwargs["strings"]
        vectors     = kwargs["embeddings"]
        ix          = kwargs["as_index_fn"]
        metric_args = kwargs["metric_args"]
        # NOTE: we use as_string because the decoder only works on known objs!

        a  = encode(string_a)
        b  = encode(string_b)
        c  = encode(string_c)
        fa = ix(string_a)
        fb = ix(string_b)
        fc = ix(string_c)
        d  = b - a + c # our nearest guess for what d is
        fd = np.log(fb) - np.log(fa) + np.log(fc)

        # Grab the n+3 closest, pick closest n that aren't source words,
        #   scale their distances by how similar their commonality is to the
        #   expected result's commonality, and choose the least (closest).
        nbrs_ix = nbrs_of(d, indeces=range(self.n+3))
        sources = [string_a, string_b, string_c]
        options = [i for i in nbrs_ix if strings[i] not in sources][:self.n]
        #   Leave only n.
        dist = np.array([metric(d, vectors[i], **metric_args) for i in options])
        dist = dist / np.mean(dist)
        #freq_score = np.array([abs(fd - np.log(o)) for o in options])
        freq_score = np.array([np.log(abs(fd - o)) for o in options])
        freq_score = freq_score / np.mean(freq_score)
        new_dist = [d + self.alpha*freq_score[i] for i, d in enumerate(dist)]
        choice = options[np.argmin(new_dist)]

        self.stats_dict["Num Options"] = self.n
        self.stats_dict["Freq Weight"] = self.alpha

        return strings[choice], d
