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
    import ray
    import scipy.cluster.vq as vq
    import sklearn.decomposition as sd



    MAX_LINES = 200000
    metric = "cosine"

    printing = True
    #ray.init(); printing = False

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

    def pca_whiten(X):
        Xp = sd.PCA().fit_transform(X)
        return np.array([vq.whiten(column) for column in Xp.T]).T

    def get_e():
        analogies_path="/mnt/pccfs/backed_up/nathan/Projects/" \
            "analogy_corpora_uncased/analogy_subcorp"
        e_analogy = [an.evaluators.analogizer.Analogizer(
            category="Linear Offset " + p,
            analogies_path=analogies_path + p) \
            for p in path_ends]
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
                category="Combined Ext Canonical", analogizers=e_ext),
            an.evaluators.analogizer_combiner.AnalogizerCombiner(
                category="Combined Linear Offset", analogizers=e_analogy)]
        return e_analogy + e_avg + e_ext + e_comb

    def get_strings():
        with open("embeddings/fasttext.en.py2.pkl", 'rb') as f:
            data_ft = pkl.load(f)
            str_f = data_ft['tokens'][:MAX_LINES]
            return data_ft, list(map(str, str_f))

    @ray.remote
    def fasttext(str_f, data_ft):
        # Fasttext:
        #   ordered by frequency, I think.
        #   non-normalized.
        #with open("embeddings/fasttext.en.pkl", 'rb') as f:
        embed_f = data_ft['vectors'][:MAX_LINES]
        #embed_fn = np.array([normalize(v) for v in embed_f])
        an_fnc = an.Analyst(embeddings=embed_f, strings=str_f,
            auto_print=printing, metric=metric, desc="Fasttext",
            evaluators=get_e())
        print("Success at saving Fasttext: "
            + str(an.Analyst.save(an_fnc,
                "saved_analyses/an" + str(MAX_LINES) + \
                "_fasttext_analogies5_" + metric)))

    @ray.remote
    def numberbatch(str_f, data_ft):
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
            auto_print=printing, desc="ConceptNet Numberbatch",
            evaluators=get_e())
        print("Success at saving Numberbatch: " + str(an.Analyst.save(an_nb,
            "saved_analyses/an" + str(MAX_LINES) + \
            "_numberbatch_analogies5_" + metric)))

    @ray.remote
    def googlenews(str_f, data_ft):
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
            auto_print=printing, desc="GoogleNews",
            evaluators=get_e())
        print("Success at saving GoogleNews: " +
            str(an.Analyst.save(an_w,
                "saved_analyses/an" + str(MAX_LINES) +
                "_googlenews_analogies5_" + metric)))

    @ray.remote
    def glove(str_f, data_ft):
        # GloVe:
        #   ordered by frequency, I think.
        #   non-normalized.
        str_g, embed_g = read_text_table(
            "embeddings/glove.6B.300d.txt", firstline=False, limit_lines=MAX_LINES)
        #embed_g = [normalize(v) for v in embed_g]
        an_g = an.Analyst(embeddings=embed_g, strings=str_g, metric=metric,
            auto_print=printing, desc="GloVe",
            evaluators=get_e())
        print("Success at saving GloVe: " + str(an.Analyst.save(an_g,
            "saved_analyses/an" + str(MAX_LINES) + "_glove_analogies5_" + metric)))

    @ray.remote
    def use(str_f, data_ft):
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
            auto_print=printing, desc="Universal Sentence Encoder",
            evaluators=get_e())
        print("Success at saving Universal Sentence Encoder: " +
            str(an.Analyst.save(
                an_u, "saved_analyses/an" + str(MAX_LINES) +
                "_USE_analogies5_" + metric)))

    #@ray.remote
    def white_use(str_f, data_ft):
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
        embed_u = pca_whiten(embed_u)
        an_u = an.Analyst(embeddings=embed_u, strings=str_f, metric=metric,
            auto_print=printing, desc="PCA Whitened USE",
            evaluators=get_e())
        print("Success at saving PCA Whitened USE: " +
            str(an.Analyst.save(
                an_u, "saved_analyses/an" + str(MAX_LINES) +
                "_pcawhiteUSE_analogies5_" + metric)))
    
    functions = [
        #fasttext,
        #numberbatch,
        #googlenews,
        #glove,
        #use,
        white_use,
    ]

    data_ft, str_f = get_strings()
    for f in functions:
        f(str_f, data_ft)

    #remotes = [f.remote(str_f, data_ft) for f in functions]

    #complete = ray.get(remotes)

    an_fnc = an.load("saved_analyses/an" + str(MAX_LINES) + "_fasttext_analogies5_" + metric)
    an_nb  = an.load("saved_analyses/an" + str(MAX_LINES) + "_numberbatch_analogies5_" + metric)
    an_w   = an.load("saved_analyses/an" + str(MAX_LINES) + "_googlenews_analogies5_" + metric)
    an_g   = an.load("saved_analyses/an" + str(MAX_LINES) + "_glove_analogies5_" + metric)
    an_u   = an.load("saved_analyses/an" + str(MAX_LINES) + "_USE_analogies5_" + metric)
    an_wu  = an.load("saved_analyses/an" + str(MAX_LINES) + "_whiteUSE_analogies5_" + metric)

    ana_list = [an_fnc, an_nb, an_w, an_g, an_u, an_wu]

    # for a in ana_list:
    #     a.analysis(True)
    an.Analyst.compare(ana_list, report_path="saved_analyses/analogies6_comparison_report.txt")

    #an.Analyst.graph_comparison([an_w, an_fnc, an_g, an_nb, an_u], "Nodes", "Count")
    #an.Analyst.graph_multi([an_w, an_fnc, an_g, an_nb, an_u], [("Nodes", "Count"), ("Nuclei", "Count"), ("Nodal 4-Hubs", "Count")], group_by_stat=False)
