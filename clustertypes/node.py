import numpy as np
import scipy.spatial as sp

class Node:

    def __init__(self, a, b, encoder, metric):
        """
        Parameters:
            a, b -- anything encodable.
            encoder -- callable with return of a numpy array
                (else will break at centroid calculation)
            metric -- callable with return of a scalar
        """
        assert a != b # Objects in a node must be different.
        self.a = a
        self.b = b
        vec_a = encoder(a) # if encoder != None else a
        vec_b = encoder(b) # if encoder != None else b
        self.distance = metric(vec_a, vec_b)
        self.centroid = (vec_a + vec_b) / 2.0
        self.alignment = vec_b - vec_a
        if metric == sp.distance.euclidean:
            self.alignment /= self.distance
        else:
            self.alignment /= np.linalg.norm(self.alignment)

    def __eq__(self, r_node):
        return ((self.a == r_node.a and self.b == r_node.b) or
                (self.b == r_node.a and self.a == r_node.b))

    def __getitem__(self, index):
        if index == 0: return self.a
        elif index == 1: return self.b
        else: raise ValueError("Index out of bounds of Node")

    # __len__ not defined so as not to confuse, because cannot return float.

    def __str__(self):
        return "Node(" + str(self.a) + ", " + str(self.b) + ")"

    # This function exists to allow the creation of nodes of nodes - supernodes;
    #   Where get_centroid becomes the encoder passed in.
    #   ie: supernode_1 = Node(node_1, node_2, Node.get_centroid, node_1.metric)
    @staticmethod
    def get_centroid(node):
        return node.centroid