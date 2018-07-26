from abc import abstractmethod
from tqdm import tqdm
import numpy as np
import traceback

from .evaluator import Evaluator
from .analogizer import Analogizer
import analyst


class AnalogizerCombiner(Evaluator, object):
    """
    Combines the results from multiple analogizers in a final report category.
    """

    def __init__(self, category="Combined Analogizers", starred=None,
            analogizers=None):
        # analogizers: if None, will automatically search for Analogizers, or
        #   instances of classes derived therefrom. Otherwise fill this with
        #   the categorical names of the ones you want it to combine data for,
        #   or references to the analogizers themselves.
        super(AnalogizerCombiner, self).__init__(category=category, starred=starred)
        self.analogizer_categories = analogizers
        self.analogizers = None
        self.score = None
        self.distances = None
        self.lengths = None
        self.score_list = None
        # self.CATEGORY = category       # See parent.
        # self.data_dict = OrderedDict() # See parent.
        # self.starred = []              # See parent.
        # self.calculated = False        # See parent.

    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # kwargs: see parent.

        printer        = kwargs["printer_fn"]
        find_evaluator = kwargs["find_evaluator_fn"]
        evaluators     = kwargs["evaluator_list"]

        # Fill in Analogizer Lists:
        self.analogizers = []
        if self.analogizer_categories is None:
            self.analogizer_categories = []
            for e in evaluators:
                if isinstance(e, Analogizer):
                    self.analogizer_categories.append(e.CATEGORY)
                    self.analogizers.append(e)
                # Only adds ones which inherit from Analogizer class.
        else:
            for i, c in enumerate(self.analogizer_categories):
                if analyst.isstring(c):
                    a = find_evaluator(c)
                    if a is not None: self.analogizers.append(a)
                    else: printer("WARNING: {} dropped {}; no evaluator with "
                        "this category was found.")
                elif isinstance(c, Analogizer):
                    self.analogizers.append(c)
                    self.analogizer_categories[i] = c.CATEGORY
                else: printer("WARNING: {} dropped {}; was not string and "
                    "does not inherit from Analogizer.".format(self.CATEGORY,
                    str(c)))

        # Make sure their data is filled in first:
        for a in self.analogizers:
            a.calculate(**kwargs)

        printer("Compiling Wisdom and Knowledge",
            "Combining Analogical Results")
            
        self.data_dict["Category Count"] = len(self.analogizers)

        if len(self.analogizers) > 0:
            try:
                correct = np.concatenate([a.correct for a in \
                    self.analogizers if a.correct is not None and \
                    len(a.correct) != 0])
                self.score = np.sum(correct) / float(len(correct))
                self.distances = np.concatenate([a.distances for a in \
                    self.analogizers if a.distances is not None and \
                    len(a.distances) != 0])
                self.lengths = np.concatenate([a.lengths for a in \
                    self.analogizers if a.lengths is not None and \
                    len(a.lengths) != 0])
            except ValueError as e:
                traceback.print_exc()
                correct = []

            self.data_dict["Analogy Count"] = len(correct)
            self.data_dict["Dropped Count"] = sum(
                [len(a.dropped) if a.dropped is not None else 0 \
                for a in self.analogizers])

            if len(correct) > 0:
                self.data_dict["Accuracy Per Category"] = np.mean([
                    a.score for a in self.analogizers if a.score is not None])
                self.data_dict["Accuracy Per Analogy"] = self.score

                # Category Score Data
                self.score_list = np.array([
                    a.score for a in self.analogizers if a.score is not None])
                if len(self.score_list) > 0:
                    self.data_dict["Most Accurate Category"] = self.analogizers[
                        np.argmax(self.score_list)].CATEGORY
                    self.data_dict["Least Accurate Category"] = \
                        self.analogizers[np.argmin(self.score_list)].CATEGORY
                    self._compute_list_stats(
                        self.score_list, "Category Score", self.data_dict)

                # Distance from point found to answer point
                self._compute_list_stats(self.distances,
                    "Dist All from Answer", self.data_dict)
                self._compute_list_stats(self.distances[np.nonzero(correct)],
                    "Dist for Correct", self.data_dict)
                self._compute_list_stats(
                    self.distances[np.nonzero(1 - correct)],
                    "Dist for Incorrect", self.data_dict)

                # Distance from c to d; the length of the analogy vector
                self._compute_list_stats(self.lengths,
                    "Analogy Length", self.data_dict)
                self._compute_list_stats(self.lengths[np.nonzero(correct)],
                    "Length Correct", self.data_dict)
                self._compute_list_stats(self.lengths[np.nonzero(1 - correct)],
                    "Length Incorrect", self.data_dict)

                self.add_star("Accuracy Per Category")
                self.add_star("Analogy Length Avg")
                self.add_star("Most Accurate Category")
                self.add_star("Least Accurate Category")
                self.add_star("Category Score Histogram Key")