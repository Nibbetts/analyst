from abc import abstractmethod
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

from .evaluator import Evaluator
from .analogizer import Analogizer
from .analogizer import WORD_ANALOGY_SEP
from .analogizer import WORD_ITEM_SEP
import analyst



class AvgCanonicalAnalogizer(Analogizer, object):
    """
    Analogizer which takes the first n unique pairs and uses them to create a
        canonical example of the desired analogy, then uses this same canonical
        vector as the offset in every analogy performed.
    This particular canonical analogizer simply averages the first n vectors.
    Since canonicals only require pairs of words, but are often compared to
        linear offset analogies, files can be read whether they have a pair per
        line or a quadruple, and will be paired off. The first n pairs will be
        used to build the canonical.
    """

    def __init__(self, category="Avg Canonical Analogies", n=10,
            exclude_source_word=True, canonical_analogies=None,
            canonical_vec = None, starred=None,
            analogies_path=None, analogies=None, analogy_vectors=None,
            analogy_sep=WORD_ANALOGY_SEP, item_sep=WORD_ITEM_SEP):
        # n: the number of examples to use from which to form a canonical.
        # exclude_source_word: whether or not to ignore the start word when
        #   searching for the nearest viable answer.
        # canonical_analogies: optional override; use list of tuples of strings.
        # canonical_vec: optional override; use a vector.
        # NOTE: if you use both of the above overrides, I do not check that they
        #   are a valid match!
        super(AvgCanonicalAnalogizer, self).__init__(
            category=category, starred=starred, analogies_path=analogies_path,
            analogies=analogies, analogy_vectors=analogy_vectors,
            analogy_sep=analogy_sep, item_sep=item_sep)

        self.n = n
        self.exclude = exclude_source_word

        self.canonical_analogies = canonical_analogies
        self.canonical_vectors = None
        #   The list of vectors from which canonical is made.
        self.canonical = canonical_vec


    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # PRE: self.analogies needs to have been filled in (with pairs)
        # POST: self.data_dict, self.starred will be filled in, as well as
        #   self.score, self.distances, self.lengths, self.dropped, and
        #   self.correct.
        self._prep_canonical(kwargs["encoder_fn"])

        if self.canonical is not None:
            self.data_dict["Canonical Length"] = np.linalg.norm(self.canonical)
            self.add_star("Canonical Length")
        self.data_dict["Canonical Components"] = len(self.canonical_vectors)

        super(AvgCanonicalAnalogizer, self).compute_stats(**kwargs)


    # Helper function which splits data for canonical if needed, and preps
    #   the collections of vectors as well.
    def _prep_canonical(self, encoder):
        # Fills things in only if None.

        if self.canonical_analogies is None:
            self.canonical_analogies = self.analogies[:self.n]
            self.analogies = self.analogies[self.n:]
            self.canonical_vectors = self.analogy_vectors[:self.n]
            #   list of pairs of vectors.
            self.analogy_vectors = self.analogy_vectors[self.n:]
        elif self.canonical_vectors is None:
            self.canonical_vectors = [
                tuple(encoder(s) for s in a) for a in self.canonical_analogies]

        if len(self.canonical_vectors) > 0:
            if type(self.canonical_vectors[0]) == tuple:
                self.canonical_vectors = [
                    t[1]-t[0] for t in self.canonical_vectors]
                #   now a list of analogy vectors.
            if self.canonical is None:
                self.canonical = np.mean(self.canonical_vectors, axis=0)


    # OVERRIDEABLE
    def analogy(self, string_c, *args, **kwargs):
        # Notice: missing arguments for string_a and string_b doesn't break
        #   the compute_stats function, since it uses *analogy to split it.
        # RETURNS: vector approximation for d, and string approximation for d.

        encode   = kwargs["encoder_fn"]
        stringit = kwargs["as_string_fn"]
        nbrs_of  = kwargs["arbitrary_neighbors_fn"]
        strings  = kwargs["strings"]
        # NOTE: we use as_string because the decoder only works on known objs!

        c = encode(string_c)
        d = c + self.canonical # our nearest guess for what d is

        if self.exclude:
            # Grab the two closest, in case the first one is our source word.
            # NOTE: Since canonical analogies are pairs only, it doesn't make
            #   sense to exclude words a or b; we have neither to go off of.
            nbrs_ix = nbrs_of(d, indeces=range(2))
            for index in nbrs_ix:
                if strings[index] != string_c:
                    return strings[index], d
        else:
            return stringit(d, in_model=False), d


    # MAY BE OVERRIDDEN IF NEEDED
    def read_analogies_file(self, **kwargs):
        # File reader function. Works on pairs or quadruples per line.
        # It works on sentence analogies too, with double-line sep for groups,
        #   for example: analogy_sep='\n\n' and item_sep='\n',
        #   values for which there are declared constants.
        # Automatically strips whitespace from items after breaking up the file.

        printer_fn = kwargs["printer_fn"]
        encode     = kwargs["encoder_fn"]

        # Force input if no corpus found:
        if not analyst.isstring(self.file_name): #kwargs["is_string_fn"](...)
            raise ValueError("NO FILENAME GIVEN FOR {}!".format(self.CATEGORY))

        # Process the file
        printer_fn("Reading the Writing on the Wall", "Reading Analogy Corpus")
        with open(self.file_name, 'r') as f:
            #lines = f.readlines()
            whole = f.read()
        groups = whole.strip().split(self.analogy_sep)
        groups = [g.strip().split(self.item_sep) for g in groups \
            if not g.isspace() and len(g) != 0]
        groups = [[item.strip() for item in a] for a in groups]

        # Certify each is length 4:
        # Remove empty strings that may have arisen from incorrect numbers
        #   of separators between things.
        for i, a in enumerate(groups):
            for j, item in enumerate(a):
                if item == "": del groups[i][j]
        # Remove those still not of length 4:
        analogies = OrderedDict() # To remove duplicates, but keep the order
        dropped = []
        for g in groups:
            if len(g) == 4:
                analogies[(g[0], g[1])] = None # Using OrderedDict as OrderedSet
                analogies[(g[2], g[3])] = None
            elif len(g) == 2:
                analogies[(g[0], g[1])] = None
            else:
                dropped.append(g)
        if len(dropped) > 0:
            # Potential problems are printed outright, so not shushed when
            #   auto_print is False.
            print("WARNING: %d GROUPS OF LENGTH != 2 or 4 WERE DROPPED!" %
                len(dropped))

        # Certify each contains only encodable strings:
        vectors = []
        valid_analogies = []
        for a in analogies.keys():
            try:
                vectors.append(tuple(encode(item) for item in a))
                valid_analogies.append(a)
            except:
                dropped.append(a)
        if len(valid_analogies) < len(analogies):
            print("WARNING: %d UNENCODEABLE PAIRS WERE DROPPED!" %
                (len(analogies) - len(valid_analogies)))

        return valid_analogies, vectors, dropped