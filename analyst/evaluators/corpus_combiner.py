from .evaluator import Evaluator
from .analogizer import Analogizer
from .analogizer_combiner import AnalogizerCombiner



class CorpusCombiner(Evaluator, object):
    """
    Takes in anlogy corpora paths and an analogizer class, and creates its own
        analogizers, hidden from the Analyst, as well as an AnalogizerCombiner
        for them. The analyst will see only the CorpusCombiner instance, which
        will show the AnalogizerCombiner's stats.
    """

    def __init__(self, paths, category="Combined Corpora",
            analogizer_class=Analogizer, analogizer_categories=None,
            analogizer_combiner_category=None,
            starred=None):
        # paths: paths to the corpora to make analogizers for.
        # category: What the Analyst calls this CorpusCombiner instance.
        # analogizer_class: the actual class (not string) to use to create new
        #   analogizer instances.
        # categories: names for the analogizers, either None (will try to make
        #   names from the substring the paths don't have in common) or a list
        #   the same length as the given number of paths (else won't use it).
        # starred: to override which data are labeled with asterisks. If None,
        #   will use defaults of the AnalogizerCombiner created.
        super(CorpusCombiner, self).__init__(
            category=category, starred=starred)
        
        self.paths = paths
        self.analogizer_class = analogizer_class
        self.analogizer_categories = analogizer_categories
        self.analogizer_combiner_category = analogizer_combiner_category
        self.analogizers = []
        self.analogizer_combiner = None
        # self.CATEGORY = category       # See parent.
        # self.stats_dict = OrderedDict() # See parent.
        # self.starred = []              # See parent.
        # self.calculated = False        # See parent.

    # OVERRIDEABLE
    def compute_stats(self, **kwargs):
        # kwargs: see parent.

        # Fill in Analogizer categories:
        if self.analogizer_categories is None:
            self.analogizer_categories = unique_string_parts(self.paths)
        elif len(self.analogizer_categories) != len(self.paths):
            print("WARNING: {} given invalid-length list of analogizer "
                "categories! Constructing categories from path names.")
            self.analogizer_categories = unique_string_parts(self.paths)
        if self.analogizer_combiner_category is None:
            self.analogizer_combiner_category = \
                self.CATEGORY + " AnalogizerCombiner"

        # Fill in Analogizer list:
        self.analogizers = [
            self.analogizer_class(category=cat, analogies_path=self.paths[i]) \
            for i, cat in enumerate(self.analogizer_categories)]

        # Now for the AnalogizerCombiner:
        self.analogizer_combiner = AnalogizerCombiner(
            category=self.analogizer_combiner_category,
            analogizers=self.analogizers, starred=self.starred)
            
        self.stats_dict = self.analogizer_combiner.get_stats_dict(
            recalculate_all=True, **kwargs)
            # We created this one, so not calculated yet.
            # This will automatically do the Analogizers, as well.
        self.starred = self.analogizer_combiner.get_starred(
            recalculate_all=False, **kwargs)

    def print_report(self): # TODO
        raise NotImplementedError("This doesn't work yet. As a work-around, "
            "you can add the individual evaluators to the Analyst with the "
            "add_evaluators function, then remove them later if you don't "
            "want the clutter. In the end the functionality to print an "
            "evaluator specifically, or a list of, might be added to Analyst's"
            "print_report function instead, and this may call that.")


def unique_string_parts(paths):
    if len(paths) == 0:
        raise ValueError("No paths given to unique_string_parts!")
    elif len(paths) == 1:
        # If only one, remove directories and file extensions:
        s = paths[0]
        try: s = s[s.rindex("/")+1:]
        except: pass
        try: s = s[:s.index(".")]
        except: pass
        return [s]
    else:
        if len(paths) != len(set(paths)): # Must be unique to start
            raise ValueError(
                "List of non-unique paths given to unique_string_parts!")
        front_len = 0
        back_len = 0
        results = []
        # Find common front part:
        for i, c in enumerate(paths[0]):
            try:
                if all([c == p[i] for p in paths[1:]]):
                    front_len += 1
            except: break
        # Find common back part:
        for i, c in enumerate(paths[0][::-1]):
            try:
                if all([c == p[len(p)-i-1] for p in paths[1:]]):
                    back_len -= 1
            except: break
        # Find unique parts:
        if back_len == 0:
            results = [p[front_len:] for p in paths]
        else:
            results = [p[front_len:back_len] for p in paths]
        # Remove extensions, unless it makes them non-unique:
        no_extensions = [r[:r.index(".")] if "." in r else r for r in results]
        if len(no_extensions) == len(set(no_extensions)):
            results = no_extensions
        # Remove directories, unless it makes them non-unique:
        no_directories = [r[r.rindex("/")+1:] \
            if "/" in r else r for r in results]
        if len(no_directories) == len(set(no_directories)):
            results = no_directories

        return results