# NOTE: Requires scholar to be sister directory to analyst.
#   ...and I think you can only run this from the parent directory of both.

if __name__ == "__main__":

    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        import scholar.scholar as sch
    else:
        raise Exception("Can't run litmus_100d.py as part of a package.")

    import analyst as an
    #from ...scholar.scholar import scholar as sch
    import numpy as np

    np.random.seed(19680801)

    def normalize(vec):
        return vec/np.linalg.norm(vec)

    try:
        s = sch.Scholar(slim=True)
    except Exception as e:
        #print("Error: " + str(e))
        raise e
        #try:
        #    s = sc.scholar.Scholar(slim=True)
        #except Exception as ex:
        #    print("Error: " + str(ex))

    #words = ["car_NN", "bus_NN", "egg_NN", "farm_NN", "cookie_NN", "beef_NN",
    #        "sheep_NN", "anvil_NN", "cake_NN", "dessert_NN", "animal_NN",
    #        "vehicle_NN"]

    words = [word + "_NN" for word in s.get_most_common_words("NN", 2000) if s.exists_in_model(word + "_NN")] +\
            [word + "_VB" for word in s.get_most_common_words("VB", 1000) if s.exists_in_model(word + "_VB")] +\
            [word + "_JJ" for word in s.get_most_common_words("JJ", 1000) if s.exists_in_model(word + "_JJ")]
    print("Number of words:", len(words))

    vectors_real = map(s.model.get_vector, words)
    vectors_fake = [normalize(vec) for vec in np.random.random(
        (len(words),len(vectors_real[0])))*2-1] # Allow negatives

    #realv_to_word = {}
    #falev_to_word = {}
    #for i, vec in enumerate(vectors_real):
    #    realv_to_word[vec] = words[i]
    #for i, vec in enumerate(vectors_fake):
    #    fakev_to_word[vec] = words[i]

    # def decode_real(vec):
        #return words[vectors_real.index(vec)]
        #return realv_to_word[vec]
    decode_real = s.model.get_word

    def decode_fake(vec):
        #return words[vectors_fake.index(vec)]
        #return fakev_to_word[vec]
        for i, vec2 in enumerate(vectors_fake):
            if np.array_equal(vec, vec2):
                return words[i]
        print("No matching vector")
        raise ValueError("No matching vector")

    def encode_real(word):
        return vectors_real[words.index(word)]

    def encode_fake(word):
        return vectors_fake[words.index(word)]

    def metric(vec1, vec2):
        return s.angle(vec1, vec2)*180/np.pi

    an_real = an.Analyst(vectors_real, metric, encode_real, decode_real,
        auto_print=True, desc="real scholar words")
    an_fake = an.Analyst(vectors_fake, metric, encode_fake, decode_fake,
        auto_print=True, desc="fake scholar words")
    an_real.compare_difference(an_fake, simple_diff=True)

    worked_r = an_real.save(an_real, "an_scholar4000_real")
    worked_f = an_fake.save(an_fake, "an_scholar4000_fake")

    
    #assert an_real.serialized
    #assert an_fake.serialized
    """
    print("an_real serialized: " + str(an_real.serialized))
    print("an_fake serialized: " + str(an_fake.serialized))
    for entry in an_real.__dict__:
        print(entry, type(an_real.__dict__[entry]))
    for entry in an_fake.__dict__:
        print(entry, type(an_fake.__dict__[entry]))
    """

    assert worked_r
    assert worked_f

    an_r = an_real.load("an_scholar4000_real", metric, encode_real, decode_real)
    an_f = an_fake.load("an_scholar4000_fake", metric, encode_fake, decode_fake)

    assert an_r != None
    assert an_f != None

    print(an_r.nodes)
    print(an_f.nodes)

"""
if __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath("litmus_100d.py") ) ) )
    import scholar.scholar as sch
else:
    raise Exception("Can't run litmus_100d.py as part of a package.")

import analyst as an
#from ...scholar.scholar import scholar as sch
import numpy as np

def normalize(vec):
    return vec/np.linalg.norm(vec)

words = ["car_NN", "bus_NN", "egg_NN", "farm_NN", "cookie_NN", "beef_NN",
        "sheep_NN", "anvil_NN", "cake_NN", "dessert_NN", "animal_NN",
        "vehicle_NN"]
try:
    s = sch.Scholar(slim=True)
except Exception as e:
    raise e

vectors_real = map(s.model.get_vector, words)
vectors_fake = [normalize(vec) for vec in np.random.random(
    (len(words),len(vectors_real[0])))*2-1] # Allow negatives

decode_real = s.model.get_word

def decode_fake(vec):
    #return words[vectors_fake.index(vec)]
    #return fakev_to_word[vec]
    for i, vec2 in enumerate(vectors_fake):
        if np.array_equal(vec, vec2):
            return words[i]
    print("No matching vector")
    raise ValueError("No matching vector")

def encode_real(word):
    return vectors_real[words.index(word)]

def encode_fake(word):
    return vectors_fake[words.index(word)]

def metric(vec1, vec2):
    return s.angle(vec1, vec2)*180/np.pi

an_real = an.Analyst(vectors_real, metric, encode_real, decode_real,
    auto_print=False, desc="real scholar words")
an_fake = an.Analyst(vectors_fake, metric, encode_fake, decode_fake,
    auto_print=False, desc="fake scholar words")
an_real.compare_difference(an_fake, simple_diff=True)
"""