class Node:

    def __init__(self, a, b, encoder, metric):
        """
        Parameters:
            a, b -- anything encodable.
            encoder -- callable with return of a numpy array (or will break at centroid calculation)
            metric -- callable with return of a scalar
        """
        self.a = a
        self.b = b
        vec_a = encoder(a)
        vec_b = encoder(b)
        self.distance = metric(vec_a, vec_b)
        self.centroid = (vec_a + vec_b) / 2.0

    def __getitem__(self, index):
        if index == 0: return self.a
        elif index == 1: return self.b
        else: raise ValueError("Index out of bounds of Node")

    def __str__(self):
        return "Node(" + str(self.a) + ", " + str(self.b) + ")"

    # This function exists to allow the creation of nodes of nodes - supernodes;
    #   Where get_centroid becomes the encoder passed in.
    #   ie: supernode_1 = Node(node_1, node_2, Node.get_centroid, node_1.metric)
    @staticmethod
    def get_centroid(node):
        return node.centroid
