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


    MAX_LINES = 20000

    def normalize(vec):
        return vec/np.linalg.norm(vec)

    #metric = "cosine"
    metric = an.Analyst.angle

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

    # Fasttext:
    #   ordered by frequency, I think.
    #   non-normalized.
    #with open("embeddings/fasttext.en.pkl", 'rb') as f:
    with open("embeddings/fasttext.en.py2.pkl", 'rb') as f:
        data_ft = pkl.load(f)
    str_f = data_ft['tokens'][:MAX_LINES]
    str_f = list(map(str, str_f))

    # Word2vec (GoogleNews):
    #   non-normalized.
    #   unordered, from gensim's dict-like structure.
    model_w = gensim.models.KeyedVectors.load_word2vec_format(
        'embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    #common_w = list(filter(lambda w: w in model_w.vocab.keys() \
    #    or bytes(w) in model_w.vocab.keys(), str_f))
    common_w = [w for w in str_f if w in model_w.vocab.keys()]
    embed_w = [normalize(model_w.get_vector(w)) for w in common_w]
    an_w = an.Analyst(embeddings=embed_w, strings=common_w, metric=metric,
        auto_print=True, desc="Word2Vec GoogleNews Normalized")
    print("Success at saving Word2Vec GoogleNews Normalized: " +
        str(an.Analyst.save(an_w,
            "saved_analyses/an" + str(MAX_LINES) + "_word2vec_googlenews_normalized")))

