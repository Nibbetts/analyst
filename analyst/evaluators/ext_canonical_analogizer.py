from abc import abstractmethod
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

from .evaluator import Evaluator
from .avg_canonical_analogizer import AvgCanonicalAnalogizer
from .analogizer import Analogizer
from .analogizer import WORD_ANALOGY_SEP
from .analogizer import WORD_ITEM_SEP
import analyst



class ExtCanonicalAnalogizer(AvgCanonicalAnalogizer, object):
    """
    Extended Canonical Analogizer:
    Analogizer which takes the first n unique pairs and uses them to create a
        canonical example of the desired analogy, then uses this same canonical
        vector as the offset in every analogy performed.
    This particular canonical analogizer extends the length of the average of
        the canonized vectors to match their average length, effectively using
        their directions as a guide and going as far as they would.
    Since canonicals only require pairs of words, but are often compared to
        linear offset analogies, files can be read whether they have a pair per
        line or a quadruple, and will be paired off. The first n pairs will be
        used to build the canonical.
    """

    def __init__(self, category="Ext Canonical Analogies", n=10,
            exclude_source_word=True, canonical_analogies=None,
            canonical_vec = None, starred=None,
            analogies_path=None, analogies=None, analogy_vectors=None,
            analogy_sep=WORD_ANALOGY_SEP, item_sep=WORD_ITEM_SEP):
        # NOTE: even if you use the canonical_vec override, in this class, we
        #   will still normalize it and set it to the average length of its
        #   components. This way you can use any vector as direction to go.
        super(ExtCanonicalAnalogizer, self).__init__(
            category=category, n=n, exclude_source_word=exclude_source_word,
            canonical_vec=canonical_vec, starred=starred,
            analogies_path=analogies_path,
            analogies=analogies, analogy_vectors=analogy_vectors,
            analogy_sep=analogy_sep, item_sep=item_sep)


    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # PRE: self.analogies needs to have been filled in (with pairs)
        # POST: self.stats_dict, self.starred will be filled in, as well as
        #   self.score, self.distances, self.lengths, self.dropped, and
        #   self.correct.

        # This preps several variables:
        self._prep_canonical(kwargs["encoder_fn"])

        if self.canonical is not None:
            self.canonical /= np.linalg.norm(self.canonical) # Normalize,
            self.canonical *= np.mean([ # And re-lengthen to average length
                np.linalg.norm(v) for v in self.canonical_vectors])

        # self.stats_dict["Canonical Length"] = np.linalg.norm(self.canonical)
        # self.stats_dict["Canonical Components"] = len(self.canonical_vectors)
        # self.add_star("Canonical Length")

        # Skip the parent and go to the grandparent?
        super(ExtCanonicalAnalogizer, self).compute_stats(**kwargs)
        # Analogizer.compute_stats(self, **kwargs)