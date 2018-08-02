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
    #import tensorflow_hub as hub


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

    # GloVe:
    #   ordered by frequency, I think.
    #   non-normalized.
    str_g, embed_g = read_text_table(
        "embeddings/glove.6B.300d.txt", firstline=False, limit_lines=MAX_LINES)
    embed_g = [normalize(v) for v in embed_g]
    an_g = an.Analyst(embeddings=embed_g, strings=str_g, metric=metric,
        auto_print=True, desc="GloVe Normalized")
    print("Success at saving GloVe Normalized: " + str(an.Analyst.save(an_g,
        "saved_analyses/an" + str(MAX_LINES) + "_glove_normalized")))
