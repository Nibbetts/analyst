from abc import abstractmethod
from tqdm import tqdm
import numpy as np
import ray

from .evaluator import Evaluator
import analyst



# DEFAULTS:
WORD_ANALOGY_SEP = '\n'
WORD_ITEM_SEP = ' '
SENTENCE_ANALOGY_SEP = '\n\n'
SENTENCE_ITEM_SEP = '\n'

class Analogizer(Evaluator, object):
    """
    What: A fully-functional parent class for customizeable objects which
        perform analogical analyses on embeddings and collect scores and
        statistics. Envisioned for usage mainly with word-level embeddings,
        though may be useful for others.
    Why: Analogizers exist to allow you to run tests with common or custom
        analogy algorithms, and analogy test sets, on your embedding spaces.
        Unless overridden, the built-in functions will do basic scoring.
    How: Create an instance with your parameters and test sets, or a custom
        class and make such an instance, which can then be fed into an Analyst
        instance along with whatever other Evaluators you want.
    NOTE: If you are testing multiple different analogy sets, these can be run
        as separate instances of an Analogizer class, with different category
        labels. However, if you are testing multiple analogy algorithms,
        these should ideally be implemented as separate Analogizer classes,
        one for each algorithm.
    """

    def __init__(self, category="Linear Offset Analogies", starred=None,
            analogies_path=None, analogies=None, analogy_vectors=None,
            analogy_sep=WORD_ANALOGY_SEP, item_sep=WORD_ITEM_SEP):
        super(Analogizer, self).__init__(category=category, starred=starred)
        #   To inherit, must call parent init.
        self.file_name = analogies_path
        #   Path to find test set file, formatted typically with each analogy
        #   being a line, with four words a:b::c:d separated by spaces.
        #   This will of course be used differently for sentence level
        #   embeddings. Either this or analogies parameter should be filled in.
        #   (If both are, only analogies will be used, not the path.)
        self.analogies = analogies
        #   List of lists of words, or equivalent functional structure.
        #   Inner lists, for basic usage, will be of length 4, a:b::c:d.
        #   This will be None if reading from a file.
        self.analogy_sep = analogy_sep
        #   Separator for analogies (groups of 4 items, a:b::c:d)
        #   These separators can be multiple symbols, ie: "\n\n".
        self.analogy_vectors = analogy_vectors
        #   list of lists of vectors, representing analogy items, to replace
        #   other options, but I assume this will almost never be used.
        self.item_sep = item_sep
        #   Separator for items in an analogy group.
        # Notice that by default this class assumes this is a word-embedding
        #   space, but this can be easily changed when initializing it by using
        #   other separators. (And there are built-ins for sentence-embeddings.)
        self.correct = []
        self.score = None
        self.distances = None
        self.lengths = None
        self.dropped = []
        #   dropped will keep track of unusable analogies from input corpus.
        # self.CATEGORY = category        # See parent.
        # self.stats_dict = OrderedDict() # See parent.
        # self.starred = []               # See parent.
        # self.calculated = False         # See parent.

        #assert analogies_path is not None or analogies is not None \
        #    or analogy_vectors is not None

    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # This is where you do your analogical run, scoring each analogy
        #   and generating data based on these.
        # kwargs: see parent.
        # PRE: self.analogies, self.analogy_vectors, and self.dropped
        #   need to have been filled in.
        # POST: self.stats_dict, self.starred will be filled in, as well as
        #   self.score, self.distances, self.lengths, self.dropped, and
        #   self.correct.

        # This particular implementation is a simple scoring, counting the
        #   number of correct results and dividing by number of analogies,
        #   Though we will also give some distance stats.
        # TODO! Modify to contain Scale-Invariant Stats!!

        assert len(kwargs["embeddings"]) >= 4

        show_progress  = kwargs["draw_progress"]
        printer        = kwargs["printer_fn"]
        metric         = kwargs["metric_fn"]
        parallels      = kwargs["parallel_count"]
        find_evaluator = kwargs["find_evaluator_fn"]

        self.stats_dict["Analogy Count"] = len(self.analogies)
        self.stats_dict["Dropped Count"] = len(self.dropped)
        self.stats_dict["Total Count"] = len(self.analogies) + len(self.dropped)

        printer("Philosophizing about Relations", "Scoring Analogies")
        if len(self.analogies) > 0:

            if parallels > 1:
                # PARALLELIZATION:
                limited_kwargs = { # Don't want all of them in parallelization!
                    "encoder_fn"             : kwargs["encoder_fn"],
                    "arbitrary_neighbors_fn" : kwargs["arbitrary_neighbors_fn"],
                    "strings"                : kwargs["strings"],
                }

                print("")
                #ray.init(ignore_reinit_error=True) # TODO - update ray version upon next release
                try: ray.init()
                except: pass

                @ray.remote
                def parallel_analogy(
                        i, string_a, string_b, string_c, *args, **kwargs):
                    return i, self.analogy(
                        string_a, string_b, string_c, *args, **kwargs)

                limited_kwargs_id = ray.put(limited_kwargs)

                # remaining_ids = [parallel_analogy.remote(
                #     i, strings_id, neighbors_id) \
                #     for i in range(min(len(space), parallels))]

                # temp_hubs = []
                # for i in tqdm(range(len(space)), disable=not show_progress):
                #     ready_ids, remaining_ids = ray.wait(remaining_ids)
                #     objects = ray.get(ready_ids[0])
                #     if i + parallels < len(space):
                #         remaining_ids.append(find_hub.remote(
                #             i + parallels, strings_id, neighbors_id))
                #     temp_hubs.append(Cluster(
                #         self.CATEGORY, encoder, metric, objects, nearest=nearest,
                #         nodes=[], auto=False, name=objects[0], **metric_args))

            else:
                #NON-PARALLELIZED:
                data = list(zip(*[
                    self.analogy(*a, **kwargs) for a in tqdm(self.analogies,
                        disable=(not show_progress))]))
                answers = data[0]
                answer_vectors = data[1]

            self.correct = np.array(answers) == [a[-1] for a in self.analogies]
            self.score = np.sum(self.correct) / float(len(self.analogies))
            self.overall = np.sum(self.correct) / \
                float(self.stats_dict["Total Count"])

            self.stats_dict["Accuracy"] = self.score
            self.stats_dict["Total Accuracy"] = self.overall

            # Distance from point found to answer point
            self.distances = np.array([metric(group[-1], answer_vectors[i]) \
                for i, group in enumerate(self.analogy_vectors)]) # TODO: Do these need to be arrays? Can we avoid conversion?
            
            self._compute_list_stats(self.distances,
                "Dist All from Answer", self.stats_dict, si="dispersion",
                **kwargs)
            self._compute_list_stats(self.distances[np.nonzero(self.correct)],
                "Dist for Correct", self.stats_dict, si="dispersion", **kwargs)
            self._compute_list_stats(
                self.distances[np.nonzero(1 - self.correct)],
                "Dist for Incorrect", self.stats_dict, si="dispersion",
                **kwargs)

            # Distance from c to d; the length of the analogy vector
            self.lengths = np.array([metric(group[-2], answer_vectors[i]) \
                for i, group in enumerate(self.analogy_vectors)])

            self._compute_list_stats(self.lengths,
                "Analogy Length", self.stats_dict, si="dispersion", **kwargs)
            self._compute_list_stats(self.lengths[np.nonzero(self.correct)],
                "Length Correct", self.stats_dict, si="dispersion", **kwargs)
            self._compute_list_stats(self.lengths[np.nonzero(1 - self.correct)],
                "Length Incorrect", self.stats_dict, si="dispersion", **kwargs)

            # Ratios of how far off we were compared to how far we went
            self.ratios = self.distances / self.lengths

            self._compute_list_stats(self.ratios,
                "Dist Ratio All from Answer", self.stats_dict)
            self._compute_list_stats(self.ratios[np.nonzero(self.correct)],
                "Dist Ratio for Correct", self.stats_dict)
            self._compute_list_stats(
                self.ratios[np.nonzero(1 - self.correct)],
                "Dist Ratio for Incorrect", self.stats_dict)

            # # Add Scale Invariant Stats:
            # spatializer = find_evaluator(
            #     self.spatializer_category, force_creation=False)
            # if spatializer is not None:
            #     dispersion = spatializer.get_stats_dict(**kwargs)[
            #         "Dispersion - Centroid Dist Avg"]
            #     for key, value in self.stats_dict.items():
            #         if ("Avg" in key or "Max" in key or "Min" in key
            #                 or "Range" in key or "Std" in key)\
            #                 and "Ratio" not in key:
            #             # The last 2 are already invariant.
            #             self.stats_dict["SI " + key] = value / dispersion

            self.add_star("Accuracy")
            self.add_star("Total Accuracy")
            self.add_star("SI Dist for Correct Avg")
            self.add_star("SI Dist for Incorrect Avg")
            self.add_star("Dist Ratio for Correct Avg")
            self.add_star("Dist Ratio for Incorrect Avg")

    # OVERRIDEABLE
    def analogy(self, string_a, string_b, string_c, *args, **kwargs):
        # string_a, string_b, and string_c are the given analogy items.
        #   string_d is not given.
        # RETURNS: vector approximation for d, and string approximation for d.
        #   Both are returned since some analogy algorithms may not naively
        #   choose the nearest possible decode, and the scoring is done in
        #   compute_stats.
        # NOTE: Overridden versions could return more data for stat gathering;
        #   to use them, you may want to override compute_stats also, optionally
        #   calling the parent function as you do.
        # NOTE: While string_d is included in args, using it in computing actual
        #   analogies would skew results, since d is supposed to be unknown.
        #   I supposed it could be used here for generating statistics, though
        #   such should be done in the compute_stats function if you can.

        # This particular implementation is a simple, Mikolov-type analogy.

        encode   = kwargs["encoder_fn"]
        #stringit = kwargs["as_string_fn"]
        nbrs_of  = kwargs["arbitrary_neighbors_fn"]
        strings  = kwargs["strings"]
        # NOTE: we use as_string because the decoder only works on known objs!

        a = encode(string_a)
        b = encode(string_b)
        c = encode(string_c)
        d = b - a + c # our nearest guess for what d is

        # Grab the four closest, just in case the first three are source words,
        #   which we exclude by default. See InclusiveAnalogizer otherwise.
        nbrs_ix = nbrs_of(d, indeces=range(4))
        sources = [string_a, string_b, string_c]
        for index in nbrs_ix:
            if strings[index] not in sources:
                return strings[index], d


    # The Analyst will call this function, which pulls it all together.
    #   You shouldn't have to override this function:
    def calculate(self, recalculate_all=False, **kwargs):
        if not self.calculated or recalculate_all:
            if kwargs == {}:
                print("NOT YET CALCULATED AND NO KWARGS GIVEN!")
            printer       = kwargs["printer_fn"]
            show_progress = kwargs["draw_progress"]
            decode        = kwargs["decoder_fn"]
            encode        = kwargs["encoder_fn"]

            if show_progress:
                printer("Evaluating " + self.CATEGORY)

            if self.analogies is None:
                if self.analogy_vectors is None:
                    # This is the expected usage - through a file.
                    self.analogies, self.analogy_vectors, self.dropped = \
                        self.read_analogies_file(**kwargs)
                else: self.analogies = [ # Assumes given strings are all valid!
                    [decode(item) for item in a] for a in self.analogy_vectors]
            elif self.analogy_vectors is None:
                self.analogy_vectors = [ # Assumes given vectors are all valid!
                    [encode(item) for item in a] for a in self.analogies]

            self.compute_stats(**kwargs)

            # Override default stars if user gave any:
            if self.given_stars != None:
                self.starred = self.given_stars

            self.calculated = True

        # Returning these means the Analyst need only access datamembers
        #   directly if we are doing specifics inspection, later,
        #   or when searching by category.
        return self.stats_dict, self.starred, self.CATEGORY

    # MAY BE OVERRIDDEN IF NEEDED
    def read_analogies_file(self, **kwargs):
        # File reader function.
        # It works on sentence analogies too, with double-line sep for groups,
        #   for example: analogy_sep='\n\n' and item_sep='\n',
        #   values for which there are declared constants.
        # Automatically strips whitespace from items after breaking up the file.

        printer_fn = kwargs["printer_fn"]
        encode     = kwargs["encoder_fn"]

        # Force input if no corpus found:
        if not analyst.isstring(self.file_name):
            raise ValueError("NO FILENAME GIVEN FOR {}!".format(self.CATEGORY))

        # Process the file
        printer_fn("Reading the Writing on the Wall", "Reading Analogy Corpus")
        with open(self.file_name, 'r') as f:
            #lines = f.readlines()
            whole = f.read()
        analogies = whole.strip().split(self.analogy_sep)
        analogies = [group.strip().split(self.item_sep) for group in analogies \
            if not group.isspace() and len(group) != 0]
        analogies = [[item.strip() for item in a] for a in analogies]

        # Certify each is length 4:
        pre_size = len(analogies)
        # Remove empty strings that may have arisen from incorrect numbers
        #   of separators between things.
        for i, a in enumerate(analogies):
            for j, item in enumerate(a):
                if item == "": del analogies[i][j]
        # Remove those still not of length 4:
        analogies = [a for a in analogies if len(a) == 4]
        dropped = [a for a in analogies if len(a) != 4]
        if len(analogies) < pre_size:
            # Potential problems are printed outright, so not shushed when
            #   auto_print is False.
            print("WARNING: %d ANALOGIES OF LENGTH != 4 WERE DROPPED!" %
                (pre_size - len(analogies)))

        # Certify each contains only encodable strings:
        vectors = []
        valid_analogies = []
        for a in analogies:
            try:
                vectors.append([encode(item) for item in a])
                valid_analogies.append(a)
            except:
                dropped.append(a)
        if len(valid_analogies) < len(analogies):
            print("WARNING: %d UNENCODEABLE ANALOGIES WERE DROPPED!" %
                (len(analogies) - len(valid_analogies)))

        return valid_analogies, vectors, dropped
