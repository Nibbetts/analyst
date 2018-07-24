#from __future__ import print_function
#from __future__ import absolute_import
#from builtins import str, bytes

if __name__ == "__main__":

    # Install the latest Tensorflow version.
    #!pip3 install --quiet "tensorflow>=1.7"
    # Install TF-Hub.
    #!pip3 install --quiet tensorflow-hub
    #???? !pip3 install seaborn

    import analyst as an
    #from ...scholar.scholar import scholar as sch
    import numpy as np
    import scipy.spatial as sp
    from tqdm import tqdm
    import pickle as pkl
    import os.path
    import gensim
    import tensorflow as tf
    import tensorflow_hub as hub

    #import tkinter
    #from tkinter import messagebox


    MAX_LINES = 500000

    metric = "euclidean"

    def read_text_table(path, firstline=True, limit_lines=None):
        lines = open(path, 'rt').readlines()
        if firstline:
            numvecs, dim = map(int, lines[0].split(" "))
        else:
            numvecs = len(lines) if limit_lines == None \
                else min(len(lines), limit_lines)
            dim = len(lines[0].split(" ")) - 1
        strings = []
        embeddings = np.empty(shape=(numvecs, dim))
        for i in tqdm(range(numvecs), desc="Reading " + path):
            row = lines[i + firstline].split(" ")
            strings.append(row[0])#str(row[0]))
            embeddings[i] = row[1:]
        return strings, embeddings

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
        return [an.evaluators.analogizer.Analogizer(category=p,
                analogies_path="/mnt/pccfs/backed_up/zac/zac_desktop/zac_docs/Corpora/"
                    "subcorp_analogy_storage/analogy_subcorp" + p) for p in path_ends] + \
                [an.evaluators.analogizer_combiner.AnalogizerCombiner()]
 
    # hide main window
    #root = tkinter.Tk()
    #root.withdraw()

    # Fasttext:
    #   ordered by frequency, I think.
    #   non-normalized.
    #with open("embeddings/fasttext.en.pkl", 'rb') as f:
    with open("embeddings/fasttext.en.py2.pkl", 'rb') as f:
        data_ft = pkl.load(f)
    str_f = data_ft['tokens'][:MAX_LINES]
    str_f = list(map(str, str_f))
    embed_f = data_ft['vectors'][:MAX_LINES]
    #embed_fn = np.array([normalize(v) for v in embed_f])
    an_fnc = an.Analyst(embeddings=embed_f, strings=str_f,
        auto_print=True, metric=metric, desc="Fasttext",
        evaluators=get_e())
    print("Success at saving Fasttext: "
        + str(an.Analyst.save(an_fnc,
            "saved_analyses/an" + str(MAX_LINES) + "_fasttext_analogies5_euclidean")))

    #messagebox.showinfo("Information","Analysis 1 complete!")

    # ConceptNet Numberbatch:
    #   alphanumeric order.
    #   normalized.
    #if not os.path.isfile("embeddings/an_numberbatch"):
    str_nb, embed_nb = read_text_table(
        "embeddings/numberbatch-en-17.06.txt", firstline=True)
    common_nb = [w for w in str_f if w in str_nb]
    indeces_nb = [str_nb.index(w) for w in common_nb]
    #embed_nb = np.array([embed_nb[i] for i in indeces_nb])
    embed_nb = embed_nb[indeces_nb]
    an_nb = an.Analyst(embeddings=embed_nb, strings=common_nb, metric=metric,
        auto_print=True, desc="ConceptNet Numberbatch",
        evaluators=get_e())
    print("Success at saving Numberbatch: " + str(an.Analyst.save(an_nb,
        "saved_analyses/an" + str(MAX_LINES) + "_numberbatch_analogies5_euclidean")))

    #messagebox.showinfo("Information","Analysis 2 complete!")

    # # Fasttext Normalization Comparison:
    # an_fe = an.Analyst(embeddings=embed_f, strings=str_f, auto_print=True,
    #     metric="euclidean", desc="Fasttext Euclidean")
    # print("Success at saving Fasttext Euclidean: " + str(an.Analyst.save(an_fe,
    #     "embeddings/an_fasttext_euclidean")))
    # an_fne = an.Analyst(embeddings=embed_fn, strings=str_f, auto_print=True,
    #     metric="euclidean", desc="Fasttext Normalized Euclidean")
    # print("Success at saving Fasttext Normalized Euclidean: "
    #     + str(an.Analyst.save(an_fne,
    #     "embeddings/an_fasttext_normalized_euclidean")))
    # an_fc = an.Analyst(embeddings=embed_f, strings=str_f, auto_print=True,
    #     metric=metric, desc="Fasttext Cosine")
    # print("Success at saving Fasttext Cosine: " + str(an.Analyst.save(an_fc,
    #     "embeddings/an_fasttext_cosine")))

    # Word2vec (GoogleNews):
    #   non-normalized.
    #   unordered, from gensim's dict-like structure.
    model_w = gensim.models.KeyedVectors.load_word2vec_format(
        'embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    #common_w = list(filter(lambda w: w in model_w.vocab.keys() \
    #    or bytes(w) in model_w.vocab.keys(), str_f))
    common_w = [w for w in str_f if w in model_w.vocab.keys()]
    embed_w = [model_w.get_vector(w) for w in common_w]
    an_w = an.Analyst(embeddings=embed_w, strings=common_w, metric=metric,
        auto_print=True, desc="GoogleNews",
        evaluators=get_e())
    print("Success at saving GoogleNews: " +
        str(an.Analyst.save(an_w,
            "saved_analyses/an" + str(MAX_LINES) +
            "_googlenews_analogies5_euclidean")))

    #messagebox.showinfo("Information","Analysis 3 complete!")

    # GloVe:
    #   ordered by frequency, I think.
    #   non-normalized.
    str_g, embed_g = read_text_table(
        "embeddings/glove.6B.300d.txt", firstline=False, limit_lines=MAX_LINES)
    #embed_g = [normalize(v) for v in embed_g]
    an_g = an.Analyst(embeddings=embed_g, strings=str_g, metric=metric,
        auto_print=True, desc="GloVe Normalized",
        evaluators=get_e())
    print("Success at saving GloVe Normalized: " + str(an.Analyst.save(an_g,
        "saved_analyses/an" + str(MAX_LINES) + "_glove_analogies5_euclidean")))

    #messagebox.showinfo("Information","Analysis 4 complete!")

    # Universal Sentence Encoder:
    #   embeddings must be found by hand from things to encode.
    #   normalized.
    #   512 dimensions.
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"
    embed = hub.Module(module_url)
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embed_u = sess.run(embed(str_f))
    an_u = an.Analyst(embeddings=embed_u, strings=str_f, metric=metric,
        auto_print=True, desc="Universal Sentence Encoder",
        evaluators=get_e())
    print("Success at saving Universal Sentence Encoder: " +
        str(an.Analyst.save(
            an_u, "saved_analyses/an" + str(MAX_LINES) +
            "_USE_analogies5_euclidean")))

    #messagebox.showinfo("Information","Analysis 5 complete!")'''
    
    an_fnc = an.load("saved_analyses/an" + str(MAX_LINES) + "_fasttext_analogies5_euclidean")
    an_nb  = an.load("saved_analyses/an" + str(MAX_LINES) + "_numberbatch_analogies5_euclidean")
    an_w   = an.load("saved_analyses/an" + str(MAX_LINES) + "_googlenews_analogies5_euclidean")
    an_g   = an.load("saved_analyses/an" + str(MAX_LINES) + "_glove_analogies5_euclidean")
    an_u   = an.load("saved_analyses/an" + str(MAX_LINES) + "_USE_analogies5_euclidean")

    #an.Analyst.compare([an_fnc, an_fe, an_fne, an_fc])
    an.Analyst.compare([an_w, an_fnc, an_g, an_nb, an_u])

    #an.Analyst.graph_comparison([an_w, an_fnc, an_g, an_nb, an_u], "Nodes", "Count")
    #an.Analyst.graph_multi([an_w, an_fnc, an_g, an_nb, an_u], [("Nodes", "Count"), ("Nuclei", "Count"), ("Nodal 4-Hubs", "Count")], group_by_stat=False)
