import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from .evaluator import Evaluator


class PopulationAnalogizer(Evaluator, object):
    """
    Takes the scores from a CorpusCombiner and plots what would have
        happened to accuracy if the vocabulary had been smaller, iteratively
        shrinking it and removing successes.

    NOTE: Results will be much less interesting if the original strings for
        the embedding space are not given in frequency order, most frequent
        first!
    """

    def __init__(self, corpus_combiner, category="Population Analogizer",
            starred=None, n=200):
        # corpus_combiner: a CorpusCombiner instance to pull our data from.
        # n: how many words to remove at each iteration. So, if the size of the
        #   space is 100000 and n=200, there will be 500 data points.
        super(PopulationAnalogizer, self).__init__(
            category=category, starred=starred)
        self.n = n
        self.corpus_combiner = corpus_combiner
        self.scores = []
        self.total_scores = []
        self.word_count = []
        # self.CATEGORY = category       # See parent.
        # self.stats_dict = OrderedDict() # See parent.
        # self.starred = []              # See parent.
        # self.calculated = False        # See parent.

    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # kwargs: see parent.
        strings = kwargs["strings"]

        self.corpus_combiner.calculate(recalculate_all=False, **kwargs)

        try:
            analogies = []
            for a in self.corpus_combiner.analogizers:
                if a.analogies is not None:
                    analogies += copy(a.analogies)
            # analogies = np.concatenate([
            #     a.analogies for a in self.corpus_combiner.analogizers
            #     if a.analogies is not None and len(a.analogies) != 0])
            correct = np.concatenate([
                a.correct for a in self.corpus_combiner.analogizers
                if a.correct is not None and len(a.correct) != 0])
            dropped = sum([
                len(a.dropped) for a in self.corpus_combiner.analogizers
                if a.dropped is not None])
            valid_left = len(analogies)
            assert len(analogies) == len(correct)
        except ValueError:
            print("WARNING: Failed to run population analysis; "
                "no data in Analogizers.")
            analogies = []
            correct = []
            dropped = 0
            valid_left = 0
        
        for i in range(len(strings),0,-self.n):
            remove = strings[i:i + self.n]
            for word in remove:
                for j in range(len(analogies)):
                    if word in analogies[j]:
                        analogies[j] = []
                        correct[j] = 0
                        dropped += 1
                        valid_left -= 1
            self.scores.append(sum(correct) / valid_left)
            self.total_scores.append(sum(correct) / (valid_left + dropped))
            self.word_count.append(i)

        self.scores = self.scores[::-1]
        self.total_scores = self.scores[::-1]
        self.word_count = self.word_count[::-1]

        self.stats_dict["Size of Groups Removed (n)"] = self.n
        self.stats_dict["Number of Data Points (Groups)"] = len(self.scores)
        self.stats_dict["Graphing function"] = "graph_accuracy"


    def graph_accuracy(self, log_scale=False):
        fn = plt.semilogx if log_scale else plt.plot
        fn(self.word_count, self.scores, color='g',
            label="Score on Remaining")
        fn(self.word_count, self.total_scores, color='b',
            label="Total Score")
        plt.xlabel("Vocab Size")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.title("Accuracy per Vocab Size")
        plt.grid(True)
        plt.show()