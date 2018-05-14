# NOTE: Requires scholar to be sister directory to analyst.
#   ...and I think you can only run this from the parent directory of both.

if __name__ == "__main__":

    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        import scholar.scholar as sch

    import analyst as an
    #from ...scholar.scholar import scholar as sch
    import numpy as np
    import scipy.spatial as sp

    np.random.seed(19680801)

    def normalize(vec):
        return vec/np.linalg.norm(vec)

    try:
        s = sch.Scholar(slim=True)
    except Exception as e:
        raise e

    #words = ["car_NN", "bus_NN", "egg_NN", "farm_NN", "cookie_NN", "beef_NN",
    #        "sheep_NN", "anvil_NN", "cake_NN", "dessert_NN", "animal_NN",
    #        "vehicle_NN"]

    words = [word + "_NN" for word in s.get_most_common_words("NN", 200) if s.exists_in_model(word + "_NN")] +\
            [word + "_VB" for word in s.get_most_common_words("VB", 100) if s.exists_in_model(word + "_VB")] +\
            [word + "_JJ" for word in s.get_most_common_words("JJ", 100) if s.exists_in_model(word + "_JJ")]
    print("Number of words:", len(words))

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
        #return s.angle(vec1, vec2)*180/np.pi
        return sp.distance.cosine(vec1, vec2)*180/np.pi

    an_real = an.Analyst(embeddings=vectors_real, strings=words, metric=metric,
        #encoder=encode_real, decoder=decode_real,
        auto_print=True, desc="real scholar words")
    an_fake = an.Analyst(embeddings=vectors_fake, strings=words, metric=metric,
        #encoder=encode_fake, decoder=decode_fake,
        auto_print=True, desc="fake scholar words")

    worked_r = an_real.save(an_real, "analyst_project/an_scholar400_real")
    worked_f = an_fake.save(an_fake, "analyst_project/an_scholar400_fake")


    assert worked_r
    assert worked_f

    an.Analyst.compare([an_real, an_fake])

    # an_r = an.Analyst.load("analyst_project/an_scholar400_real")
    # an_f = an.Analyst.load("analyst_project/an_scholar400_fake")

    # assert an_r != None
    # assert an_f != None

    # an.Analyst.compare([an_r, an_f])
