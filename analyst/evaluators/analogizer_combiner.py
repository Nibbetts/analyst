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
        super(AnalogizerCombiner, self).__init__(
            category=category, starred=starred)
        self.analogizer_categories = analogizers
        self.analogizers = None
        self.score = None
        self.distances = None
        self.lengths = None
        self.score_list = None
        # self.CATEGORY = category       # See parent.
        # self.stats_dict = OrderedDict() # See parent.
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
            
        self.stats_dict["Category Count"] = len(self.analogizers)

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
                self.ratios = np.concatenate([a.ratios for a in \
                    self.analogizers if a.ratios is not None and \
                    len(a.ratios) != 0])
            except ValueError as e:
                # traceback.print_exc()
                print("WARNING: Failed to combine; no data in Analogizers.")
                correct = []

            self.stats_dict["Analogy Count"] = len(correct)
            self.stats_dict["Dropped Count"] = sum(
                [len(a.dropped) if a.dropped is not None else 0 \
                for a in self.analogizers])
            self.stats_dict["Total Count"] = self.stats_dict["Analogy Count"] +\
                self.stats_dict["Dropped Count"]

            self.overall = np.sum(correct) / float(
                self.stats_dict["Total Count"]) if len(correct) > 0 else 0.0

            if len(correct) > 0:
                self.stats_dict["Accuracy Per Category"] = np.mean([
                    a.score for a in self.analogizers if a.score is not None])
                self.stats_dict["Total Accuracy Per Category"] = np.mean([
                    a.overall for a in self.analogizers \
                    if a.overall is not None])
                self.stats_dict["Accuracy Per Analogy"] = self.score
                self.stats_dict["Total Accuracy Per Analogy"] = self.overall

                # Category Score Data
                self.score_list = np.array([
                    a.score for a in self.analogizers if a.score is not None])
                if len(self.score_list) > 0:
                    self.stats_dict["Most Accurate Category"] = \
                        self.analogizers[np.argmax(self.score_list)].CATEGORY
                    self.stats_dict["Least Accurate Category"] = \
                        self.analogizers[np.argmin(self.score_list)].CATEGORY
                    self._compute_list_stats(
                        self.score_list, "Category Score", self.stats_dict)
                    self.stats_dict.pop("Category Score Avg")
                    #   Duplicate of Accuracy Per Category.

                # Distance from point found to answer point
                self._compute_list_stats(self.distances,
                    "Dist All from Answer", self.stats_dict,
                    si="dispersion", **kwargs)
                self._compute_list_stats(self.distances[np.nonzero(correct)],
                    "Dist for Correct", self.stats_dict,
                    si="dispersion", **kwargs)
                self._compute_list_stats(
                    self.distances[np.nonzero(1 - correct)],
                    "Dist for Incorrect", self.stats_dict,
                    si="dispersion", **kwargs)

                # Distance from c to d; the length of the analogy vector
                self._compute_list_stats(self.lengths,
                    "Analogy Length", self.stats_dict,
                    si="dispersion", **kwargs)
                self._compute_list_stats(self.lengths[np.nonzero(correct)],
                    "Length Correct", self.stats_dict,
                    si="dispersion", **kwargs)
                self._compute_list_stats(self.lengths[np.nonzero(1 - correct)],
                    "Length Incorrect", self.stats_dict,
                    si="dispersion", **kwargs)

                # Ratios of how far off we were compared to how far we went
                self._compute_list_stats(self.ratios,
                    "Dist Ratio All from Answer", self.stats_dict)
                self._compute_list_stats(self.ratios[np.nonzero(correct)],
                    "Dist Ratio for Correct", self.stats_dict)
                self._compute_list_stats(self.ratios[np.nonzero(1 - correct)],
                    "Dist Ratio for Incorrect", self.stats_dict)

                # # Add Scale Invariant Stats:
                # spatializer = find_evaluator(
                #     self.analogizers[0].spatializer_category,
                #     force_creation=False)
                # if spatializer is not None:
                #     dispersion = spatializer.get_stats_dict(**kwargs)[
                #         "Dispersion - Centroid Dist Avg"]
                #     for key, value in self.stats_dict.items():
                #         if ("Avg" in key or "Max" in key or "Min" in key
                #                 or "Range" in key or "Std" in key)\
                #                 and "Ratio" not in key:
                #             # The last 2 are already invariant.
                #             self.stats_dict["SI " + key] = value / dispersion

                self.add_star("Accuracy Per Category")
                self.add_star("Total Accuracy Per Category")
                self.add_star("SI Analogy Length Avg")
                self.add_star("Most Accurate Category")
                self.add_star("Least Accurate Category")
                self.add_star("Category Score Histogram Key")
