import numpy as np
import scipy.spatial as sp

class Node:

    def __init__(self, a, b, encoder, metric, **metric_args):
        """
        Parameters:
            a, b -- anything encodable.
            encoder -- callable with return of a numpy array
                (else will break at centroid calculation)
            metric -- callable with return of a scalar
            metric_args -- additional arguments for the metric function.
                Use like **kwargs is used.
        """
        assert a != b # Objects in a node must be different.
        self.a = a
        self.b = b
        self.vec_a = encoder(a) # if encoder != None else a
        self.vec_b = encoder(b) # if encoder != None else b
        self.distance = metric(self.vec_a, self.vec_b, **metric_args)
        self.centroid = (self.vec_a + self.vec_b) / 2.0
        self.alignment_vec = self.vec_b - self.vec_a
        self.alignment = 0.0
        # The method below for finding alignment factor has fail cases,
        #   so alignment must be finished externally. The code below is then
        #   not needed, but harms nothing.
        bisector = np.zeros(shape=len(self.vec_a))
        bisector[0] = 1 # Creating a one-hot vector to bisect the space
        if np.dot(self.alignment_vec, bisector) < 0:
            self.alignment_vec = -self.alignment_vec
            #   Flip vec if in wrong direction.
        align_len = np.linalg.norm(self.alignment_vec)
        if align_len != 0: self.alignment_vec /= align_len

    def __eq__(self, r_node):
        # Works on lists and tuples, as well, if only 2 elements.
        if len(r_node) != 2: return False
        return ((self.a == r_node[0] and self.b == r_node[1]) or
                (self.b == r_node[1] and self.a == r_node[0]))

    def __getitem__(self, index):
        if index == 0: return self.a
        elif index == 1: return self.b
        else: raise ValueError("Index out of bounds of Node")

    def __len__(self):
        return 2 # NOT distance between! __len__ can't return a float.

    def __str__(self):
        return "Node(" + str(self.a) + ", " + str(self.b) + ")"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(tuple(sorted((self.a, self.b)))))

    # This function exists to allow the creation of nodes of nodes - supernodes;
    #   Where get_centroid becomes the encoder passed in.
    #   ie: supernode_1 = Node(node_1, node_2, Node.get_centroid, node_1.metric)
    @staticmethod
    def get_centroid(node):
        return node.centroid