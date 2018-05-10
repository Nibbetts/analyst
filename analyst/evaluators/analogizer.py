from abc import abstractmethod

from .evaluator import Evaluator


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

    def __init__(self, category, starred=None,
            analogies_path=None, analogies=None):
        super(Analogizer, self).__init__(category=category, starred=starred)
        #   To inherit, must call parent init.
        self.filename = analogies_path
        #   Path to find test set file, formatted typically with each analogy
        #   being a line, with four words a:b::c:d separated by spaces.
        #   This will of course be used differently for sentence level
        #   embeddings. Either this or analogies should be filled in. (If both
        #   are, only analogies will be used.)
        self.analogies = analogies
        #   List of lists of words, or equivalent functional structure.
        #   Inner lists, for basic usage, will be of length 4, a:b::c:d.
        #   This will be None if reading from a file.
        # self.CATEGORY = category       # See parent.
        # self.data_dict = OrderedDict() # See parent.
        # self.starred = []              # See parent.
        # self.calculated = False        # See parent.

    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # This is where you do your analogical run.
        # kwargs: see parent.
        # POST: self.data_dict, self.starred will be filled in.
        pass

    # OVERRIDEABLE
    def analogy(self, **kwargs):
        pass

    # The Analyst will call this function, which pulls it all together.
    #   You shouldn't have to override this function:
    def calculate(self, recalculate_all=False, **kwargs):
        if not self.calculated or recalculate_all:
            if kwargs == {}:
                print("NOT YET CALCULATED AND NO KWARGS GIVEN!")
            printer = kwargs["printer_fn"]
            show_progress = kwargs["show_progress"]

            if show_progress:
                printer("Evaluating " + self.CATEGORY)

            self._read_analogies_file(printer)
            self.compute_stats(**kwargs)

            # Override default stars if user gave any:
            if self.given_stars != None:
                self.starred = self.given_stars

            self.calculated = True

        # Returning these means the Analyst need only access datamembers
        #   directly if we are doing specifics inspection, later,
        #   or when searching by category.
        return self.data_dict, self.starred, self.CATEGORY

    # MAY BE OVERRIDDEN IF NEEDED
    def _read_analogies_file(self, printer_fn):
        # File reader function.
        pass