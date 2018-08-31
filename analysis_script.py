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
    # import scipy.spatial as sp
    from tqdm import tqdm
    import sys
    import ray
    import pickle as pkl



    MAX_LINES = 5000#400000
    metric = "cosine"

    printing = True
    # ray.init(ignore_reinit_error=True); printing = False
    cpus = 10

    def read_text_table(path, firstline=True, limit_lines=None):
        lines = open(path, 'r', errors='ignore').readlines()
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
        "sense2vec",
        "USE",
        "liteUSE",
        "largeUSE",
        "DEPS",
    ]

    fnames = ["/mnt/pccfs/backed_up/nate/Projects/analyst_project/"
        "saved_analyses/" + p + '_' + metric + str(MAX_LINES) \
        for p in fname_parts]

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
    #     import sklearn.decomposition as sd
    #     import scipy.cluster.vq as vq
    #     
    #     Xp = sd.PCA().fit_transform(X)
    #     return np.array([vq.whiten(column) for column in Xp.T]).T

    def get_e():
        analogies_path="/mnt/pccfs/backed_up/nate/Projects/" \
            "corpora/analogy_corpora_uncased/analogy_subcorp"
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

        analogies_path="/mnt/pccfs/backed_up/nate/Projects/" \
            "corpora/byu_analogical_reasoning_untagged/"
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
        
        return ["All"] + e_analogy + er_analogy + e_avg + er_avg + \
            e_ext + er_ext + e_comb + er_comb

    # def get_e_longer():
        # analogies_path="/mnt/pccfs/backed_up/nate/Projects/" \
        #     "corpora/analogy_corpora_uncased/analogy_subcorp"
        # e_ext = [an.evaluators.ext_canonical_analogizer.ExtCanonicalAnalogizer(                              
        #     category="Long Ext Canonical " + p,                                                          
        #     analogies_path=analogies_path + p) \
        #     for p in path_ends]
        # return e_ext + [an.evaluators.analogizer_combiner.AnalogizerCombiner(
        #     category="Combined Long Ext Canonical", analogizers=e_ext)]

    # def get_e_freq():
        # analogies_path="/mnt/pccfs/backed_up/nate/Projects/" \
        #     "corpora/analogy_corpora_uncased/analogy_subcorp"
        # e_freq = [an.evaluators.frequency_analogizer.FrequencyAnalogizer(
        #     category="Frequency " + p,
        #     analogies_path=analogies_path + p) \
        #     for p in path_ends]
        # e_comb = [an.evaluators.analogizer_combiner.AnalogizerCombiner(
        #         category="Combined Frequency", analogizers=e_freq)]
        # return e_freq + e_comb

    def get_strings():
        return pkl.load(open("/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
            "word_frequencies/clean_words_ordered_ALPHA_2.pkl", 'rb'))[:MAX_LINES]


    # @ray.remote
    def fasttext(allowed_str):
        # Fasttext:
        #   ordered by frequency.
        #   non-normalized.
        an_fnc = an.load(fnames[0], verbosity=1)
        if an_fnc is not None:
            an_fnc.add_evaluators(get_e())# + get_e_freq())
            an_fnc.analysis(print_report=False)
            an_fnc.save()
        else:
            with open("/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
                "fasttext.en.py2.pkl", 'rb') as f:
                data_ft = pkl.load(f)
            str_f = data_ft['tokens'][:MAX_LINES]
            str_f = list(map(str, str_f))
            embed_f = data_ft['vectors'][:MAX_LINES]
            #embed_fn = np.array([normalize(v) for v in embed_f])
            common = [w for w in allowed_str if w in str_f]
            indeces = [str_f.index(w) for w in common]
            embed_f = embed_f[indeces]
            an_fnc = an.Analyst(embeddings=embed_f, strings=common,
                auto_print=printing, metric=metric, desc="Fasttext",
                evaluators=get_e(), auto_save=2, file_name=fnames[0],
                over_write=True, parallel_count=cpus)# + get_e_freq())

    # @ray.remote
    def numberbatch(allowed_str):
        # ConceptNet Numberbatch:
        #   alphanumeric order.
        #   normalized.
        #if not os.path.isfile("embeddings/an_numberbatch"):
        an_nb = an.load(fnames[1], verbosity=1)
        if an_nb is not None:
            an_nb.add_evaluators(get_e())
            an_nb.analysis(print_report=False)
            an_nb.save()
        else:
            str_nb, embed_nb = read_text_table(
                "/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
                "numberbatch-en-17.06.txt", firstline=True)
            common_nb = [w for w in allowed_str if w in str_nb]
            indeces_nb = [str_nb.index(w) for w in common_nb]
            #embed_nb = np.array([embed_nb[i] for i in indeces_nb])
            embed_nb = embed_nb[indeces_nb]
            an_nb = an.Analyst(embeddings=embed_nb, strings=common_nb,
                metric=metric, auto_print=printing, parallel_count=cpus,
                desc="ConceptNet Numberbatch", evaluators=get_e(), auto_save=2,
                file_name=fnames[1], over_write=True)

    # @ray.remote
    def googlenews(allowed_str):
        # Word2vec (GoogleNews):
        #   non-normalized.
        #   unordered, from gensim's dict-like structure.
        an_w = an.load(fnames[2], verbosity=1)
        if an_w is not None:
            an_w.add_evaluators(get_e())
            an_w.analysis(print_report=False)
            an_w.save()
        else:
            import gensim

            model_w = gensim.models.KeyedVectors.load_word2vec_format(
                "/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
                "GoogleNews-vectors-negative300.bin", binary=True)
            #common_w = list(filter(lambda w: w in model_w.vocab.keys() \
            #    or bytes(w) in model_w.vocab.keys(), allowed_str))
            common_w = [w for w in allowed_str if w in model_w.vocab.keys()]
            embed_w = [model_w.get_vector(w) for w in common_w]
            an_w = an.Analyst(embeddings=embed_w, strings=common_w, metric=metric,
                auto_print=printing, desc="GoogleNews", parallel_count=cpus,
                evaluators=get_e(), auto_save=2, file_name=fnames[2],
                over_write=True)

    # @ray.remote
    def glove(allowed_str):
        # GloVe:
        #   ordered by frequency.
        #   non-normalized.
        an_g = an.load(fnames[3], verbosity=1)
        if an_g is not None:
            an_g.add_evaluators(get_e())
            an_g.analysis(print_report=False)
            an_g.save()
        else:
            str_g, embed_g = read_text_table(
                "/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
                "glove.6B.300d.txt", firstline=False, limit_lines=MAX_LINES)
            #embed_g = [normalize(v) for v in embed_g]
            common = [w for w in allowed_str if w in str_g]
            indeces = [str_g.index(w) for w in common]
            embed_g = embed_g[indeces]
            an_g = an.Analyst(embeddings=embed_g, strings=common, metric=metric,
                auto_print=printing, desc="GloVe", parallel_count=cpus,
                evaluators=get_e(), auto_save=2, file_name=fnames[3],
                over_write=True)

    # @ray.remote
    def sense_2_vec(allowed_str):
        # Sense2Vec:
        #   originally from reddit, then through sense2vec, I modify sense2vec
        #   by doing a weighted average of all the parts of speech of each word
        #   I seek, since they are often close in the space.
        #   NOT normalized.
        #   128 dimensions.

        a = an.load(fnames[4], verbosity=1)
        if a is not None:
            a.add_evaluators(get_e())
            a.analysis(print_report=False)
            a.save()
        else:
            import sense2vec
            
            s2v = sense2vec.load('/mnt/pccfs/not_backed_up/nate/'
                'analyst_embeddings/reddit_vectors-1.1.0/')
            strings = []
            vectors = []
            endings = ['|ADJ', '|ADP', '|ADV', '|AUX', '|CONJ', '|DET', '|INTJ',
                '|NOUN', '|NUM', '|PART', '|PRON', '|PROPN', '|PUNCT', '|SCONJ',
                '|SYM', '|VERB', '|X']
            for s in allowed_str:
                senses = []
                freq_sum = 0
                for e in endings:
                    try:
                        t = s2v[s+e]
                        senses.append(t[1]*t[0])
                        freq_sum += t[0]
                    except:
                        pass
                if len(senses) > 0:
                    strings.append(s)
                    vectors.append(np.sum(senses, axis=0)/freq_sum)
            a = an.Analyst(embeddings=np.array(vectors), strings=strings,
                metric=metric, auto_print=printing, desc="Sense2Vec",
                parallel_count=cpus, evaluators=get_e(), auto_save=2,
                file_name=fnames[4], over_write=True)

    # @ray.remote
    def use(allowed_str):
        # Universal Sentence Encoder:
        #   embeddings must be found by hand from things to encode.
        #   normalized.
        #   512 dimensions.
        an_u = an.load(fnames[5], verbosity=1)
        if an_u is not None:
            an_u.add_evaluators(get_e())
            an_u.analysis(print_report=False)
            an_u.save()
        else:
            import tensorflow as tf
            import tensorflow_hub as hub

            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
            embed = hub.Module(module_url)
            tf.logging.set_verbosity(tf.logging.ERROR)
            batches = [allowed_str[b:b+10000] for b in range(0, len(allowed_str), 10000)]
            embeddings = []
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                for b in batches:
                    embeddings.append(sess.run(embed(b)))
            embeddings = np.vstack(embeddings)
            an_u = an.Analyst(embeddings=embeddings, strings=allowed_str, metric=metric,
                auto_print=printing, desc="Universal Sentence Encoder",
                evaluators=get_e(), auto_save=2, file_name=fnames[5],
                over_write=True, parallel_count=cpus)

    # @ray.remote
    def use_lite(allowed_str):
        # Universal Sentence Encoder:
        #   embeddings must be found by hand from things to encode.
        #   normalized.
        #   512 dimensions.
        an_u = an.load(fnames[6], verbosity=1)
        if an_u is not None:
            an_u.add_evaluators(get_e())
            an_u.analysis(print_report=False)
            an_u.save()
        else:
            import tensorflow as tf
            import tensorflow_hub as hub
            import sentencepiece as spm

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
                embedder = module(
                    inputs=dict(
                        values=input_placeholder.values,
                        indices=input_placeholder.indices,
                        dense_shape=input_placeholder.dense_shape))

                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

                batches = [allowed_str[b:b+10000] for b in range(0, len(allowed_str), 10000)]
                embeddings = []
                for b in batches:
                    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, b)
                    embeddings.append(sess.run(
                        embedder,
                        feed_dict={input_placeholder.values: values,
                                input_placeholder.indices: indices,
                                input_placeholder.dense_shape: dense_shape}))
                embeddings=np.vstack(embeddings)

                an_u = an.Analyst(
                    embeddings=embeddings, strings=allowed_str, metric=metric,
                    auto_print=printing, desc="USE Lite", parallel_count=cpus,
                    evaluators=get_e(), auto_save=2, file_name=fnames[6],
                    over_write=True)

    # @ray.remote
    def use_large(allowed_str):
        # Universal Sentence Encoder:
        #   embeddings must be found by hand from things to encode.
        #   normalized.
        #   512 dimensions.
        an_u = an.load(fnames[7], verbosity=1)
        if an_u is not None:
            an_u.add_evaluators(get_e())
            an_u.analysis(print_report=False)
            an_u.save()
        else:
            import tensorflow as tf
            import tensorflow_hub as hub

            module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
            embed = hub.Module(module_url)
            tf.logging.set_verbosity(tf.logging.ERROR)
            batches = [allowed_str[b:b+10000] for b in range(0, len(allowed_str), 10000)]
            embeddings = []
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                for b in batches:
                    embeddings.append(sess.run(embed(b)))
            embeddings = np.vstack(embeddings)
            an_u = an.Analyst(embeddings=embeddings, strings=allowed_str, metric=metric,
                auto_print=printing, desc="USE Large", parallel_count=cpus,
                evaluators=get_e(), auto_save=2, file_name=fnames[7],
                over_write=True)
    
    # @ray.remote
    def deps(allowed_str):
        # Dependency-Based Word Embeddings:
        #   appears to be ordered by frequency.
        #   Normalized.
        a = an.load(fnames[8], verbosity=1)
        if a is not None:
            a.add_evaluators(get_e())
            a.analysis(print_report=False)
            a.save()
        else:
            strings, embed_g = read_text_table(
                "/mnt/pccfs/not_backed_up/nate/analyst_embeddings/"
                "dependency_based_word_embeddings/deps.words",
                firstline=False, limit_lines=MAX_LINES)
            common = [w for w in allowed_str if w in strings]
            indeces = [strings.index(w) for w in common]
            embed_g = embed_g[indeces]
            a = an.Analyst(embeddings=embed_g, strings=common, metric=metric,
                auto_print=printing, desc="DEPS", parallel_count=cpus,
                evaluators=get_e(), auto_save=2, file_name=fnames[8],
                over_write=True)
    
    functions = [
        fasttext,
        numberbatch,
        googlenews,
        glove,
        sense_2_vec,
        use,
        use_lite,
        use_large,
        deps,
    ]

    to_run = [int(a) for a in sys.argv[1:]] \
        if len(sys.argv) > 1 else range(len(functions))
    allowed_str = get_strings()
    for i in to_run:
        functions[i](allowed_str) # 0-6 indicate which analyst

    # remotes = [f.remote(allowed_str) for f in functions]

    # complete = ray.get(remotes)

    # ana_list = [an.load(n) for n in fnames]
    # ana_list = [a for a in ana_list if a is not None]

    # for a in ana_list:
    #     a.analysis(True)

    # an.Analyst.correlate(ana_list,
    #     "Combined Linear Offset", "Accuracy Per Category",
    #     search_categories=[c for c in ana_list[0].categories if (
    #         "Canonical" not in c and "Offset" not in c and "Reasoning" not in c
    #     )])

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
