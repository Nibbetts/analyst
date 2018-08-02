from __future__ import unicode_literals
# ONLY WORKS IN PYTHON2?

if __name__ == "__main__":

    import analyst as an
    #from ...scholar.scholar import scholar as sch
    import numpy as np
    import scipy.spatial as sp
    from tqdm import tqdm
    import pickle as pkl
    import os.path
    import word2vec



    metric = "cosine"
    model = word2vec.load("/home/nate/Projects/scholar/" \
        "postagged_wikipedia_for_word2vec.bin")

    path_start = "/mnt/pccfs/backed_up/nathan/Projects/" \
        "analogy_corpora_postagged/analogy_subcorp"
    path_ends = [
        "1_capitals_countries",
        "2_capitals_world",
        "3_countries_currency",
        "4_city_state",
        "5_family_relations",
        "6_adj_adverb",
        "7_opposites",
        "8_comparative",
        "9_superlative",
        "10_present_participle",
        "11_nationality_adj",
        "12_past_tense",
        "13_plural",
        "14_plural_verbs",
    ]

    corpora = [an.evaluators.analogizer.Analogizer(
        category="Analogies_" + p,
        analogies_path=path_start+p) for p in path_ends]
    anagc = an.evaluators.analogizer_combiner.AnalogizerCombiner()
    an_fnc = an.Analyst(
        embeddings=None,
        strings=model.vocab,
        encoder=model.__getitem__,
        auto_print=True,
        metric=metric,
        desc="Word2Vec Analogies",
        evaluators=[anagc] + corpora, # + ["all"],
        auto_save=True,
        over_write=True,
    )
