from abc import abstractmethod
from tqdm import tqdm
import numpy as np

from .evaluator import Evaluator


class AnalogizerCombiner(Evaluator, object):
    """
    Combines the results from multiple analogizers in a final report category.
    """

    def __init__(self, category="Combined Analogizers", starred=None,
            analogizers=None):
        super(AnalogizerCombiner, self).__init__(category=category, starred=starred)
        self.analogizer_categories = analogizers
        self.analogy_count = None
        self.score = None
        self.distances = None
        self.lengths = None
        self.dropped_count = None
        # self.CATEGORY = category       # See parent.
        # self.data_dict = OrderedDict() # See parent.
        # self.starred = []              # See parent.
        # self.calculated = False        # See parent.

    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # This is where you do your analogical run, scoring each analogy
        #   and generating data based on these.
        # kwargs: see parent.
        # POST: self.data_dict, self.starred will be filled in.

        # This particular implementation is a simple scoring, counting the
        #   number of correct results and dividing by number of analogies,
        #   Though we will also give some distance stats.

        show_progress = kwargs["draw_progress"]
        printer       = kwargs["printer_fn"]
        metric        = kwargs["metric_fn"]

        printer("Philosophizing about Relations", "Scoring Mikolov Analogies")
        data = list(zip(*[
            self.analogy(*a[:3], **kwargs) for a in tqdm(self.analogies,
                disable=(not show_progress))]))
        answers = data[0]
        vectors = data[1]

        correct = np.array(answers) == [a[3] for a in self.analogies]
        self.score = np.sum(correct) / float(len(self.analogies))
        self.distances = np.array([metric(group[3], vectors[i]) \
            for i, group in enumerate(self.analogy_vectors)]) # TODO: Do these need to be arrays? Can we avoid conversion?
        self.lengths = np.array([metric(group[2], vectors[i]) \
            for i, group in enumerate(self.analogy_vectors)])

        self.data_dict["Analogy Count"] = len(self.analogies)
        self.data_dict["Dropped Count"] = self.dropped
        self.data_dict["Accuracy"] = self.score

        # Distance from point found to answer point
        self._compute_list_stats(self.distances,
            "Dist All from Answer", self.data_dict)
        self._compute_list_stats(self.distances[np.nonzero(correct)],
            "Dist for Correct", self.data_dict)
        self._compute_list_stats(self.distances[np.nonzero(1 - correct)],
            "Dist for Incorrect", self.data_dict)

        # Distance from c to d; the length of the analogy vector
        self._compute_list_stats(self.lengths,
            "Analogy Length", self.data_dict)
        self._compute_list_stats(self.lengths[np.nonzero(correct)],
            "Length Correct", self.data_dict)
        self._compute_list_stats(self.lengths[np.nonzero(1 - correct)],
            "Length Incorrect", self.data_dict)


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
        return self.data_dict, self.starred, self.CATEGORY
