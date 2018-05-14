if __name__ == "__main__":

    import analyst as an
    #from ...scholar.scholar import scholar as sch
    import numpy as np
    import scipy.spatial as sp
    from tqdm import tqdm
    import pickle as pkl
    import os.path
    import gensim

    def normalize(vec):
        return vec/np.linalg.norm(vec)

    def metric(vec1, vec2):
        #return s.angle(vec1, vec2)*180/np.pi
        return sp.distance.cosine(vec1, vec2)*180/np.pi

    def read_text_table(path):
        lines = open(path, 'rt').readlines()
        numvecs, dim = map(int, lines[0].split(" "))
        strings = []
        embeddings = np.empty(shape=(numvecs, dim))
        for i in tqdm(range(numvecs), desc="Reading " + path):
            row = lines[i + 1].split(" ")
            strings.append(row[0])
            embeddings[i] = row[1:]
        return strings, embeddings

    # Fasttext:
    with open("embeddings/fasttext.en.pkl", 'rb') as f:
        data_ft = pkl.load(f)
    str_f = data_ft['tokens'][:2000]
    embed_f = data_ft['vectors'][:2000]
    embed_fn = map(normalize, embed_f)
    an_fnc = an.Analyst(embeddings=embed_fn, strings=str_f, auto_print=True,
        metric=metric, desc="Fasttext Normalized Cosine")
    print("Success at saving Fasttext Normalized Cosine: "
        + str(an.Analyst.save(an_fnc,
        "analyst_project/embeddings/an_fasttext_normalized_cosine")))

    # ConceptNet Numberbatch:
    #if not os.path.isfile("analyst_project/embeddings/an_numberbatch"):
    str_nb, embed_nb = read_text_table(
        "embeddings/numberbatch-en-17.06.txt")
    common_nb = filter(lambda w: w in str_nb, str_f)
    indeces_nb = map(str_nb.index, common_nb)
    embed_nb = embed_nb[indeces_nb]
    an_nb = an.Analyst(embeddings=embed_nb, strings=common_nb, metric=metric,
        auto_print=True, desc="ConceptNet Numberbatch")
    print("Success at saving Numberbatch: " + str(an.Analyst.save(an_nb,
        "analyst_project/embeddings/an_numberbatch")))

    # # Fasttext Normalization Comparison:
    # an_fe = an.Analyst(embeddings=embed_f, strings=str_f, auto_print=True,
    #     metric="euclidean", desc="Fasttext Euclidean")
    # print("Success at saving Fasttext Euclidean: " + str(an.Analyst.save(an_fe,
    #     "analyst_project/embeddings/an_fasttext_euclidean")))
    # an_fne = an.Analyst(embeddings=embed_fn, strings=str_f, auto_print=True,
    #     metric="euclidean", desc="Fasttext Normalized Euclidean")
    # print("Success at saving Fasttext Normalized Euclidean: "
    #     + str(an.Analyst.save(an_fne,
    #     "analyst_project/embeddings/an_fasttext_normalized_euclidean")))
    # an_fc = an.Analyst(embeddings=embed_f, strings=str_f, auto_print=True,
    #     metric=metric, desc="Fasttext Cosine")
    # print("Success at saving Fasttext Cosine: " + str(an.Analyst.save(an_fc,
    #     "analyst_project/embeddings/an_fasttext_cosine")))

    # Word2vec (GoogleNews):
    model_w = gensim.models.KeyedVectors.load_word2vec_format(
        'embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    common_w = filter(lambda w: w in model_w.vocab.keys(), str_f)
    embed_w = map(model_w.get_vector, common_w)
    embed_w = map(normalize, embed_w)
    an_w = an.Analyst(embeddings=embed_w, strings=common_w, metric=metric,
        auto_print=True, desc="Word2Vec GoogleNews Normalized")
    print("Success at saving Word2Vec GoogleNews Normalized: " +
        str(an.Analyst.save(an_w,
            "analyst_project/embeddings/an_word2vec_googlenews_normalized")))

    
    

    an.Analyst.compare([an_fnc, an_fe, an_fne, an_fc])
    an.Analyst.compare([an_nb, an_fnc, an_w, an_u, an_g])