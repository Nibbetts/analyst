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
    import numpy as np
    #from ...scholar.scholar import scholar as sch
    #import scipy.spatial as sp
    #from tqdm import tqdm
    #import pickle as pkl
    #import os.path
    #import gensim
    #import tensorflow as tf
    #import tensorflow_hub as hub


    MAX_LINES = 50000

    #def normalize(vec):
    #    return vec/np.linalg.norm(vec)

    metric = "cosine"
    filename = "saved_analyses/an" + str(MAX_LINES) + "_ccc_utterance_hubs"

    # # Universal Sentence Encoder:
    # #   Embeddings must be found by hand from things to encode.
    # #   U. S. Encodings are already normalized.
    # #   512 dimensions.
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"
    # embedder = hub.Module(module_url)
    # tf.logging.set_verbosity(tf.logging.ERROR)
    # with tf.Session() as sess:
    #     sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #     embed = sess.run(embedder(str_f))

    lines = np.load( "/mnt/pccfs/not_backed_up/data/chitchat/processed/ccc_lines.npy" )
    pts = np.load( "/mnt/pccfs/not_backed_up/data/chitchat/processed/ccc_pts.npy" )

    #print("loaded")
    #print(len(lines))
    unique_lines = []
    unique_pts = []
    l = set()
    for i, line in enumerate(lines):
        if line not in l:
            l.add(line)
            unique_lines.append(line)
            unique_pts.append(pts[i])

    #print("unique things gathered")
    #print(len(unique_lines))
    #assert len(set([str(unique_pts[i]) for i in range(len(unique_pts))])) == len(unique_pts)
    #print("asserted uniqueness of vectors")

    an_ccc = an.Analyst(
        embeddings=unique_pts[:MAX_LINES],
        strings=unique_lines[:MAX_LINES],
        metric=metric,
        auto_print=True,
        desc="ChitChatChallenge Utterance Hubs",
        #evaluators=["Nodal 4-Hubs"],
        calculate=True
    )

    print("Success at saving ChitChatChallenge Utterance Hubs: " +
        str(an.Analyst.save(an_ccc, filename)))

    a = an.load(filename)

    hubber = a.find_evaluator("Nodal 4-Hubs")
    hubs = hubber.get_clusters()
    sizes = [len(h) for h in hubs]
    order = np.argsort(sizes)[::-1]
    #order = np.argsort([h.stats_dict["Dispersion"] for h in hubs])
    hubs = np.array(hubs)[order]#.tolist()

    #print(np.array(sizes)[order])

    print("Number of Utterances:", len(a.strings))
    print("Number of Hubs:", len(hubs))
    #"""
    for i, h in enumerate(hubs):
        print("")
        print("Hub:       ", i)#, h.name)
        print("Percentage:", "%", 100.0 * len(h) / float(len(a.strings)))
        print("Dispersion:", h.stats_dict["Dispersion"])
        
        inds = np.argsort(h.centroid_distances)
        sorted_objs = np.array(h.objects)[inds[:10]]
        print("Sample Utterances:")
        for u in sorted_objs:
            print(u"\t", u)
    #"""