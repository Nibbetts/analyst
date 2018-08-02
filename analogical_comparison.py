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
    import sentencepiece as spm



    MAX_LINES = 10000
    metric = "cosine"

    printing = True
    # ray.init(); printing = False

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

    fname_parts = [
        "fasttext",
        "numberbatch",
        "googlenews",
        "glove",
        "USE",
        "liteUSE",
        "largeUSE",
    ]

    fnames = ["saved_analyses/an" + str(MAX_LINES) + "_" + p + \
        "_analogies" for p in fname_parts]

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

    names_2 = [
        "accessing_containers",
        "affordance",
        "belong",
        "causation",
        "containers",
        "locations_for_objects",
        "rooms_for_containers",
        "rooms_for_objects",
        "tools",
        "trash_or_treasure",
        "travel",
    ]

    # def pca_whiten(X):
    #     Xp = sd.PCA().fit_transform(X)
    #     return np.array([vq.whiten(column) for column in Xp.T]).T

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

        analogies_path="/home/nate/Projects/BYU-Analogical-Reasoning-Dataset/text_untagged/"
        er_analogy = [an.evaluators.analogizer.Analogizer(
            category="Linear Offset " + p,
            analogies_path=analogies_path + p + ".txt") \
            for p in names_2]
        er_avg = [an.evaluators.avg_canonical_analogizer.AvgCanonicalAnalogizer(
            category="Avg Canonical " + p,
            analogies_path=analogies_path + p + ".txt") \
            for p in names_2]
        er_ext = [an.evaluators.ext_canonical_analogizer.ExtCanonicalAnalogizer(                              
            category="Ext Canonical " + p,                                                          
            analogies_path=analogies_path + p + ".txt") \
            for p in names_2]
        er_comb = [an.evaluators.analogizer_combiner.AnalogizerCombiner(
                category="Combined Avg Can Reasoning", analogizers=er_avg),
            an.evaluators.analogizer_combiner.AnalogizerCombiner(
                category="Combined Ext Can Reasoning", analogizers=er_ext),
            an.evaluators.analogizer_combiner.AnalogizerCombiner(
                category="Combined Lin Ofst Reasoning", analogizers=er_analogy)]
        
        return e_analogy + er_analogy + e_avg + er_avg + \
            e_ext + er_ext + e_comb + er_comb

    # def get_e_freq():
        # analogies_path="/mnt/pccfs/backed_up/nathan/Projects/" \
        #     "analogy_corpora_uncased/analogy_subcorp"
        # e_freq = [an.evaluators.frequency_analogizer.FrequencyAnalogizer(
        #     category="Frequency " + p,
        #     analogies_path=analogies_path + p) \
        #     for p in path_ends]
        # e_comb = [an.evaluators.analogizer_combiner.AnalogizerCombiner(
        #         category="Combined Frequency", analogizers=e_freq)]
        # return e_freq + e_comb

    def get_strings():
        with open("embeddings/fasttext.en.py2.pkl", 'rb') as f:
            data_ft = pkl.load(f)
            str_f = data_ft['tokens'][:MAX_LINES]
            return data_ft, list(map(str, str_f))

    # @ray.remote
    def fasttext(str_f, data_ft):
        # Fasttext:
        #   ordered by frequency, I think.
        #   non-normalized.
        an_fnc = an.load(fnames[0])
        if an_fnc is not None:
            an_fnc.add_evaluators(get_e())# + get_e_freq())
            an_fnc.analysis(print_report=False)
            an_fnc.save()
        else:
            #with open("embeddings/fasttext.en.pkl", 'rb') as f:
            embed_f = data_ft['vectors'][:MAX_LINES]
            #embed_fn = np.array([normalize(v) for v in embed_f])
            an_fnc = an.Analyst(embeddings=embed_f, strings=str_f,
                auto_print=printing, metric=metric, desc="Fasttext",
                evaluators=get_e())# + get_e_freq())
            print("Success at saving Fasttext: "
                + str(an.Analyst.save(an_fnc, fnames[0])))

    # @ray.remote
    def numberbatch(str_f, data_ft):
        # ConceptNet Numberbatch:
        #   alphanumeric order.
        #   normalized.
        #if not os.path.isfile("embeddings/an_numberbatch"):
        an_nb = an.load(fnames[1])
        if an_nb is not None:
            an_nb.add_evaluators(get_e())
            an_nb.analysis(print_report=False)
            an_nb.save()
        else:
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
                fnames[1])))

    # @ray.remote
    def googlenews(str_f, data_ft):
        # Word2vec (GoogleNews):
        #   non-normalized.
        #   unordered, from gensim's dict-like structure.
        an_w = an.load(fnames[2])
        if an_w is not None:
            an_w.add_evaluators(get_e())
            an_w.analysis(print_report=False)
            an_w.save()
        else:
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
                str(an.Analyst.save(an_w, fnames[2])))

    # @ray.remote
    def glove(str_f, data_ft):
        # GloVe:
        #   ordered by frequency, I think.
        #   non-normalized.
        an_g = an.load(fnames[3])
        if an_g is not None:
            an_g.add_evaluators(get_e())
            an_g.analysis(print_report=False)
            an_g.save()
        else:
            str_g, embed_g = read_text_table(
                "embeddings/glove.6B.300d.txt", firstline=False, limit_lines=MAX_LINES)
            #embed_g = [normalize(v) for v in embed_g]
            an_g = an.Analyst(embeddings=embed_g, strings=str_g, metric=metric,
                auto_print=printing, desc="GloVe",
                evaluators=get_e())
            print("Success at saving GloVe: " + str(an.Analyst.save(an_g, fnames[3])))

    # @ray.remote
    def use(str_f, data_ft):
        # Universal Sentence Encoder:
        #   embeddings must be found by hand from things to encode.
        #   normalized.
        #   512 dimensions.
        an_u = an.load(fnames[4])
        if an_u is not None:
            an_u.add_evaluators(get_e())
            an_u.analysis(print_report=False)
            an_u.save()
        else:
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
            embed = hub.Module(module_url)
            tf.logging.set_verbosity(tf.logging.ERROR)
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                embed_u = sess.run(embed(str_f))
            an_u = an.Analyst(embeddings=embed_u, strings=str_f, metric=metric,
                auto_print=printing, desc="Universal Sentence Encoder",
                evaluators=get_e())
            print("Success at saving Universal Sentence Encoder: " +
                str(an.Analyst.save(an_u, fnames[4])))

    # @ray.remote
    def use_lite(str_f, data_ft):
        # Universal Sentence Encoder:
        #   embeddings must be found by hand from things to encode.
        #   normalized.
        #   512 dimensions.
        an_u = an.load(fnames[5])
        if an_u is not None:
            an_u.add_evaluators(get_e())
            an_u.analysis(print_report=False)
            an_u.save()
        else:
            def process_to_IDs_in_sparse_format(sp, sentences):
                # An utility method that processes sentences with the sentence piece processor
                # 'sp' and returns the results in tf.SparseTensor-similar format:
                # (values, indices, dense_shape)
                ids = [sp.EncodeAsIds(x) for x in sentences]
                max_len = max(len(x) for x in ids)
                dense_shape=(len(ids), max_len)
                values=[item for sublist in ids for item in sublist]
                indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
                return (values, indices, dense_shape)

            with tf.Session() as sess:
                module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
                spm_path = sess.run(module(signature="spm_path"))
                # spm_path now contains a path to the SentencePiece model stored inside the
                # TF-Hub module

                sp = spm.SentencePieceProcessor()
                sp.Load(spm_path)

                input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
                embeddings = module(
                    inputs=dict(
                        values=input_placeholder.values,
                        indices=input_placeholder.indices,
                        dense_shape=input_placeholder.dense_shape))

                values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, str_f)

                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                message_embeddings = sess.run(
                    embeddings,
                    feed_dict={input_placeholder.values: values,
                            input_placeholder.indices: indices,
                            input_placeholder.dense_shape: dense_shape})

                an_u = an.Analyst(
                    embeddings=message_embeddings, strings=str_f, metric=metric,
                    auto_print=printing, desc="USE Lite",
                    evaluators=get_e())

                print("Success at saving USE Lite: " +
                    str(an.Analyst.save(an_u, fnames[5])))

    # @ray.remote
    def use_large(str_f, data_ft):
        # Universal Sentence Encoder:
        #   embeddings must be found by hand from things to encode.
        #   normalized.
        #   512 dimensions.
        an_u = an.load(fnames[6])
        if an_u is not None:
            an_u.add_evaluators(get_e())
            an_u.analysis(print_report=False)
            an_u.save()
        else:
            module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
            embed = hub.Module(module_url)
            tf.logging.set_verbosity(tf.logging.ERROR)
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                embed_u = sess.run(embed(str_f))
            an_u = an.Analyst(embeddings=embed_u, strings=str_f, metric=metric,
                auto_print=printing, desc="USE Large",
                evaluators=get_e())
            print("Success at saving USE Large: " +
                str(an.Analyst.save(an_u, fnames[6])))
    
    functions = [
        fasttext,
        numberbatch,
        googlenews,
        glove,
        use,
        use_lite,
        use_large,
    ]

    data_ft, str_f = get_strings()
    for f in functions:
        f(str_f, data_ft)

    # remotes = [f.remote(str_f, data_ft) for f in functions]

    # complete = ray.get(remotes)

    ana_list = [an.load(n) for n in fnames]
    ana_list = [a for a in ana_list if a is not None]

    # for a in ana_list:
    #     a.analysis(True)

    an.Analyst.correlate(ana_list,
        "Combined Linear Offset", "Accuracy Per Category",
        search_categories=["Combined Avg Canonical"])

    # an.Analyst.compare(ana_list)

    # an.Analyst.graph_comparison(
    #     ana_list, "Combined Linear Offset", "Accuracy Per Category")
    # an.Analyst.graph_comparison_multi(ana_list, [
    #         ("Combined Linear Offset", "Accuracy Per Category"),
    #         ("Combined Avg Canonical", "Accuracy Per Category"),
    #         ("Combined Ext Canonical", "Accuracy Per Category"),
    #         ("Combined Lin Ofst Reasoning", "Accuracy Per Category"),
    #         ("Combined Avg Can Reasoning", "Accuracy Per Category"),
    #         ("Combined Ext Can Reasoning", "Accuracy Per Category"),
    #     ], group_by_stat=False)
    # an.Analyst.graph_comparison_multi(ana_list, [
    #         ("Combined Linear Offset", "Accuracy Per Category"),
    #         ("Combined Avg Canonical", "Accuracy Per Category"),
    #         ("Combined Ext Canonical", "Accuracy Per Category"),
    #         ("Combined Lin Ofst Reasoning", "Accuracy Per Category"),
    #         ("Combined Avg Can Reasoning", "Accuracy Per Category"),
    #         ("Combined Ext Can Reasoning", "Accuracy Per Category"),
    #     ], group_by_stat=True)
