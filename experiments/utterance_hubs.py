# REQUIREMENTS:
# ------------------------------------------------------
# REQUIRES ANALYST:
#   ~$ git clone https://github.com/Nibbetts/analyst.git
#
# REQUIRES TENSORFLOW 1.7 AT LEAST, CHECK VERSION FIRST:
#   ~$ pip3 show tensorflow
#   ~$ pip3 install --quiet "tensorflow>=1.7"
#
# REQUIRES TENSORFLOW HUB:
#   ~$ pip3 install --quiet tensorflow-hub
#
# TENSORFLOW HUB OR ANALYST MAY HAVE OTHER REQUIREMENTS,
# Try and see what it complains about.
# ------------------------------------------------------

try:
    from analyst_project import analyst as an
except:
    import analyst as an

import numpy as np
#from ...scholar.scholar import scholar as sch
#import scipy.spatial as sp
#from tqdm import tqdm
#import pickle as pkl
#import os.path
#import gensim
import tensorflow as tf
import tensorflow_hub as hub


MAX_LINES = 20000
METRIC = "cosine"

# Process a line-separated utterance file and encode it with the U.S.E:
def process_file(file_name):
    print("Reading utterances...")
    with open(file_name, 'r') as f:
        utterances = f.readlines()
    utterances = [u.strip() for u in utterances]

    print("Culling duplicate utterances...")
    lines = []
    l = set()
    for line in utterances:
        if line not in l:
            l.add(line)
            lines.append(line)

    # Universal Sentence Encoder:
    print("Encoding utterances...")
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"
    embedder = hub.Module(module_url)
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        pts = sess.run(embedder(lines))

    return lines, pts

# Make an Analyst and let it compute:
def run_analyst(lines, pts, tag="utterance_hubs", save=True):
    print("Analyzing space...")
    a = an.Analyst(
        embeddings=pts[:MAX_LINES],
        strings=lines[:MAX_LINES],
        metric=METRIC,
        auto_print=True,
        desc=tag + "_" + str(len(lines)),
        evaluators=["Nodal 4-Hubs"],
        calculate=True
    )

    if save:
        # Save a copy of the analyst:
        analyst_file = "experiments/" + tag + "_" + str(len(a.strings)) \
            + "_analyst"
        print("Success at saving Utterance Hubs: " +
            str(an.Analyst.save(a, analyst_file)))

    return a

# Extracts and orders hubs from an analyst:
def get_ordered_hubs(analyst_inst, ordering="population"):
    hubber = analyst_inst.find_evaluator("Nodal 4-Hubs")
    hubs = hubber.get_clusters()

    if ordering == "population":
        sizes = [len(h) for h in hubs]
        order = np.argsort(sizes)[::-1]
    elif ordering == "dispersion":
        order = np.argsort([h.stats_dict["Dispersion"] for h in hubs])
    elif ordering == "default": 
        order = range(len(hubs))
    else:
        raise ValueError("Unrecognized parameter '" + str(ordering) + "'!")

    return np.array(hubs)[order] # NOTE: This will break if all are same size,
    #   because it will convert the cluster objects themselves into rows
    #   of a 2D array.

# Make and save report:
def report(analyst_inst, hubs, ordering, tag="utterance_hubs", save=True):
    denom = len(analyst_inst.strings)
    report = ""
    report += "Number of Utterances: " + str(denom)
    report += "\nNumber of Hubs: " + str(len(hubs))
    report += "\nMetric: " + METRIC
    for i, h in enumerate(hubs):
        report += "\n\nHub:        " + str(i)#, h.name)
        report += "\nPercentage: %" + str(100.0 * len(h) / float(denom))
        report += "\nDispersion: " + str(h.stats_dict["Dispersion"])
        
        inds = np.argsort(h.centroid_distances)
        sorted_objs = np.array(h.objects)[inds[:10]]
        report += "\nSample Utterances:"
        for u in sorted_objs:
            report += "\n\t" + u

    if save:
        file_name = "experiments/" + tag + "_" + str(denom) + \
            "_sortedreport_" + ordering + ".txt"
        with open(file_name, 'w') as f:
            f.write(report)

    return report

# Combine it all:
def utterance_hubs_from_file(input_file, tag="utterance_hubs"):
    # WARNING: WILL recalculate all!
    a = run_analyst( *process_file(input_file), tag=tag )

    # Make and display our reports:
    print("")
    print( report(a, get_ordered_hubs(a, "population"), "population", tag) )
    report(a, get_ordered_hubs(a, "dispersion"), "dispersion", tag)

    return a



# SCRIPT BEHAVIOR:
#   Example:
#       Run this command from the analyst_project directory:
#           "python3 experiments/utterance_hubs.py input_utterances_path
#           [output_tag] [max_lines]"
#       And it will save to the experiments directory.
#   NOTE: if output_tag is undefined, will default to "utterance_hubs".
#       We use a tag instead of a name because multiple files are generated.
#       max_lines defaults to 20,000. Will work on up to 50,000 but cannot save.
if __name__ == "__main__":

    from sys import argv

    assert 2 <= len(argv) <= 4
    input_file = argv[1]
    if len(argv) == 2:
        tag = "utterance_hubs"
    elif len(argv) == 3:
        try:
            MAX_LINES = int(argv[2])
            tag = "utterance_hubs"
        except:
            tag = argv[2]
    elif len(argv) == 4:
        tag = argv[2]
        MAX_LINES = int(argv[3])

    utterance_hubs_from_file(input_file, tag=tag)
