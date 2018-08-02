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
    #import gensim
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
        auto_print=True, desc="Universal Sentence Encoder with words")
    print("Success at saving Universal Sentence Encoder with words: " +
        str(an.Analyst.save(
            an_u, "saved_analyses/an" + str(MAX_LINES) + "_universal_sentence_encoder_with_words")))
