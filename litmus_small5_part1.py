#from __future__ import print_function
#from __future__ import absolute_import
#from builtins import str, bytes

if __name__ == "__main__":

    import analyst as an
    #from ...scholar.scholar import scholar as sch
    import numpy as np
    import scipy.spatial as sp
    from tqdm import tqdm
    import pickle as pkl
    import os.path


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
    embed_f = data_ft['vectors'][:MAX_LINES]
    embed_fn = np.array([normalize(v) for v in embed_f])
    an_fnc = an.Analyst(embeddings=embed_fn, strings=str_f, auto_print=True,
        metric=metric, desc="Fasttext Normalized Cosine")
    print("Success at saving Fasttext Normalized Cosine: "
        + str(an.Analyst.save(an_fnc,
            "saved_analyses/an" + str(MAX_LINES) + "_fasttext_normalized_cosine")))

