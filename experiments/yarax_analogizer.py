from abc import abstractmethod
from tqdm import tqdm
import numpy as np

from .analogizer import Analogizer
import analogizer
import analyst
#BELONGS IN EVALUATORS FOLDER

# TODO: test eliptical yarax, so that this is still a valid algorithm on
#   non-normalized spaces!!


class YaraxAnalogizer(Analogizer, object):
    """
    YARAX Analogizer
        y = A^(-1) * R * A * x
        (See yarax function below for explanation)
    """

    def __init__(self, category="Yarax Analogies", starred=None,
            analogies_path=None, analogies=None, analogy_vectors=None,
            analogy_sep=analogizer.WORD_ANALOGY_SEP,
            item_sep=analogizer.WORD_ITEM_SEP):
        super(YaraxAnalogizer, self).__init__(
            category=category, starred=starred, analogies_path=analogies_path,
            analogies=analogies, analogy_vectors=analogy_vectors,
            analogy_sep=analogy_sep, item_sep=item_sep)


    # OVERRIDEABLE
    def analogy(self, string_a, string_b, string_c, **kwargs):
        # string_a, string_b, and string_c are the given analogy items.
        #   string_d is not given.
        # RETURNS: vector approximation for d, and string approximation for d.
        #   Both are returned since some analogy algorithms may not naively
        #   choose the nearest possible decode, and the scoring is done in
        #   compute_stats.

        encode  = kwargs["encoder_fn"]
        nbrs_of = kwargs["arbitrary_neighbors_fn"]
        strings = kwargs["strings"]
        # NOTE: we use as_string because the decoder only works on known objs!

        a = encode(string_a)
        b = encode(string_b)
        c = encode(string_c)
        
        analogy_dir = b - a
        analogy_angle = analyst.angle(a, b, degrees=False)
        d = self.yarax(c, analogy_dir, analogy_angle)
        d /= np.linalg.norm(d)

        # Grab the four closest, just in case the first three are source words,
        #   which we exclude by default. See InclusiveAnalogizer otherwise.
        nbrs_ix = nbrs_of(d, indeces=range(4))
        sources = [string_a, string_b, string_c]
        for index in nbrs_ix:
            if strings[index] not in sources:
                return strings[index], d


    def yarax(self, vec_x, vec_dir, theta):
        '''
            Distance Respecting Hypersphere Traversal
            Arc-tracer, instead of vector addittion.
            -----------------------------------------
            vec_x: the word vector to apply an analogy to--a normalized vector.
                This is our starting point.
            vec_ref: the vector of the analogy - length does not matter.
            theta: the angle traversed around the hypersphere
                in the direction of the reference vector,
                starting from the tip of vec_x. In radians.
            returns: the vector resulting from the angular traversal.
            Methodology: y = A^(-1) * R * A * x, where:
                x is our starting vector,
                A is the basis of the plane we want to rotate on,
                    made from vec_x and vec_dir, with vec_dir orthonormalized,
                R is the simple 2D rotation matrix,
                A^(-1) is A inverse -
                    achieved by transposing A, since A is orthonormal, and
                y is the rotated vector, now in the original basis. Returns.
            NOTE: reversing direction is not as simple as a negative angle,
                simply because if we have rotated past a certain point,
                our direction vector will already be pointing backwards!
        '''

        # Gram-Schmidt on second row of basis of plane, vec_dir:
        #    Orthogonalize 2nd row to 1st,
        row2 = vec_dir - np.dot(vec_x, vec_dir)*vec_x
        row2 /= np.linalg.norm(row2)  # ...and normalize it.

        return np.dot(
            np.vstack((vec_x, row2)).T,  # The basis of the plane to rotate in,
                # This is A^(-1), or A^T because it is orthonormal (truncated).
            np.array([np.cos(theta), np.sin(theta)]).T) # Truncated R;
                # This is R*A*x, where A*x = [1,0] because it is the
                #   representation of a vector in a plane created from itself
                #   and an orthogonal vector.
                # Thus only the left half of the original R remains.
