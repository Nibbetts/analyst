import analyst as an
#from ...scholar.scholar import scholar as sch
import numpy as np
import scipy.spatial as sp
from tqdm import tqdm
import pickle as pkl
import os.path
import word2vec


# MAX_LINES = 200000
metric = "cosine"
printing = True
model = word2vec.load("/home/nate/Projects/scholar/" \
    "postagged_wikipedia_for_word2vec.bin")

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

def get_e():
    analogies_path="/mnt/pccfs/backed_up/nathan/Projects/analogy_corpora_uncased/analogy_subcorp"
    e_avg = [an.evaluators.avg_canonical_analogizer.AvgCanonicalAnalogizer(
        category="Avg Canonical " + p,
        analogies_path=analogies_path + p) \
        for p in path_ends]
    e_ext = [an.evaluators.ext_canonical_analogizer.ExtCanonicalAnalogizer(                              
        category="Ext Canonical " + p,                                                          
        analogies_path=analogies_path + p) \
        for p in path_ends]
    e_comb = [an.evaluators.analogizer_combiner.AnalogizerCombiner(
            category="Combined Avg Canonical", analogizers=e_avg),
        an.evaluators.analogizer_combiner.AnalogizerCombiner(
            category="Combined Ext Canonical", analogizers=e_ext)]
    return e_comb + e_avg + e_ext

def word2vec_analysis():
    return an.Analyst(
        embeddings=None,
        strings=model.vocab,
        encoder=model.__getitem__,
        auto_print=True,
        metric=metric,
        desc="Word2Vec Canonical Test",
        evaluators=get_e(),
        auto_save=True,
        over_write=True,
    )

# def fasttext(str_f, data_ft):
#     # Fasttext:
#     #   ordered by frequency, I think.
#     #   non-normalized.
#     #with open("embeddings/fasttext.en.pkl", 'rb') as f:
#     embed_f = data_ft['vectors'][:MAX_LINES]
#     #embed_fn = np.array([normalize(v) for v in embed_f])
#     return an.Analyst(embeddings=embed_f, strings=str_f,
#         auto_print=printing, metric=metric, desc="Fasttext",
#         evaluators=get_e(), auto_save=True, over_write=True)

# def read_text_table(path, firstline=True, limit_lines=None):
#     lines = open(path, 'rt').readlines()
#     if firstline:
#         numvecs, dim = map(int, lines[0].split(" "))
#     else:
#         numvecs = len(lines) if limit_lines == None \
#             else min(len(lines), limit_lines)
#         dim = len(lines[0].split(" ")) - 1
#     strings = []
#     embeddings = np.empty(shape=(numvecs, dim))
#     for i in tqdm(range(numvecs), desc="Reading " + path):
#         row = lines[i + firstline].split(" ")
#         strings.append(row[0])#str(row[0]))
#         embeddings[i] = row[1:]
#     return strings, embeddings

# def get_strings():
#     with open("embeddings/fasttext.en.py2.pkl", 'rb') as f:
#         data_ft = pkl.load(f)
#         str_f = data_ft['tokens'][:MAX_LINES]
#         return data_ft, list(map(str, str_f))


if __name__ == "__main__":

    #data_ft, str_f = get_strings()
    #fasttext(str_f, data_ft)
    word2vec_analysis()
