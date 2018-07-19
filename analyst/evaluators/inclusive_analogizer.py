from abc import abstractmethod
from tqdm import tqdm
import numpy as np

from .analogizer import Analogizer
from .analogizer import WORD_ANALOGY_SEP
from .analogizer import WORD_ITEM_SEP
import analyst



class InclusiveAnalogizer(Analogizer, object):
    """
    Mikolov-like analogy evaluator,
        only does not exclude the source words as possible answers.
    """

    def __init__(self, category="Inclusive Analogies", starred=None,
            analogies_path=None, analogies=None, analogy_vectors=None,
            analogy_sep=WORD_ANALOGY_SEP,
            item_sep=WORD_ITEM_SEP):
        super(InclusiveAnalogizer, self).__init__(
            category=category, starred=starred, analogies_path=analogies_path,
            analogies=analogies, analogy_vectors=analogy_vectors,
            analogy_sep=analogy_sep, item_sep=item_sep)

    # OVERRIDEABLE
    def analogy(self, string_a, string_b, string_c, **kwargs):
        # string_a, string_b, and string_c are the given analogy items.
        #   string_d is not given.
        # RETURNS: vector approximation for d, and string approximation for d.
        #   Both are returned since some analogy algorithms may not naively
        #   choose the nearest possible decode, and the scoring is done in
        #   compute_stats.

        encode   = kwargs["encoder_fn"]
        stringit = kwargs["as_string_fn"]
        # NOTE: we use as_string because the decoder only works on known objs!

        a = encode(string_a)
        b = encode(string_b)
        c = encode(string_c)
        d = b - a + c # our nearest guess for what d is

        # Naive Mikolov-like analogy, except does not exclude source words:
        return stringit(d, in_model=False), d
