import numpy as np

class TestSet2d:
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

    def encode(self, obj):
        return obj

    def decode(self, obj):
        return obj