import numpy as np

class TestSet2D:
    """
    A 2D test set containing 32 pre-set points with some apparent clustering.
    To use in Analyst, just pass an instance of this object, or its data,
    in as "embeddings". Little more than an identity function is provided to
    replace the encode and decode functions.
    """

    def __init__(self):
        self.data = np.array([
            [ 0.00, 4.00 ], [ 0.25, 4.00 ], [ 2.00, 4.00 ], [ 2.50, 4.00 ],
            [-1.50, 3.00 ], [ 2.00, 3.00 ], [ 4.00, 3.00 ], [-3.00, 2.50 ],
            [-2.00, 2.50 ], [-1.75, 2.50 ], [-1.00, 2.50 ], [-2.00, 2.00 ],
            [ 0.00, 2.00 ], [ 3.50, 2.00 ], [-1.00, 1.00 ], [ 1.75, 0.50 ],
            [-3.00, 0.00 ], [ 1.00, 0.00 ], [ 3.00, 0.00 ], [ 1.00,-0.50 ],
            [ 2.00,-0.50 ], [-3.00,-1.00 ], [-1.00,-1.00 ], [ 1.25,-1.00 ],
            [-1.00,-1.50 ], [-1.50,-2.00 ], [ 1.50,-2.00 ], [ 3.00,-2.00 ],
            [-0.50,-3.00 ], [ 0.00,-3.00 ], [-2.00,-4.00 ], [ 4.00,-4.00 ] ])
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def encode(self, string):
        # return np.array(self.data[int(string)]) # in case array doesn't work
        return self.data[int(string)]

    def decode(self, vector):
        # return str(np.nonzero(self.data==vector)[0][0]) # doesn't work right...
        #    but was an attempt for numpy arrays
        # return str(self.data.index(vector)) # for list instead of np arrays

        #v = None
        for i, v in enumerate(self.data):
            if np.array_equal(v, vector):
                return str(i)
        return None