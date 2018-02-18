# NOTE: Requires scholar to be sister directory to analyst

if __name__ == "__main__":

    if __package__ is None:
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        import scholar.scholar as sch
    else:
        from ..scholar import scholar as sch

    import Analyst as an
    #from ...scholar.scholar import scholar as sch
    import numpy as np

    def normalize(vec):
        return vec/np.linalg.norm(vec)

    words = ["car_NN", "bus_NN", "egg_NN", "farm_NN", "cookie_NN", "beef_NN",
            "sheep_NN", "anvil_NN", "icecream_NN", "dessert_NN", "animal_NN",
            "vehicle_NN"]
    s = sch.Scholar(slim=True)
    vectors_real = map(s.model.get_vector, words)
    vectors_fake = [normalize(vec) for vec in np.random.random(
        (len(words),len(vectors_real[0])))]

    def decode_real(vec):
        return words[vectors_real.index(vec)]

    def decode_fake(vec):
        return words[vectors_fake.index(vec)]

    def encode_real(word):
        return vectors_real[words.index(word)]

    def encode_fake(word):
        return vectors_fake[words.index(word)]

    metric = s.angle

    an_real = an.Analyst(vectors_real, metric, encode_real, decode_real, desc="real scholar words")
    an_fake = an.Analyst(vectors_fake, metric, encode_fake, decode_fake, desc="fake scholar words")
    an_real.compare_difference(an_fake)