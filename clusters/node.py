class Node:

    def __init__(self, a, b, encoder, metric):
        self.a = a
        self.b = b
        self.distance = metric(a, b)
        self.center = sum(encoder(a), encoder(b)) / 2.0
        # NOTE: sum requires the encoder to return numpy arrays!

    def __getitem__(self, index):
        if index == 0: return self.a
        elif index == 1: return self.b
        else: raise ValueError("Index out of bounds of Node")