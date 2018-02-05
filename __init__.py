"""
Analyst
-------
    Interface for toolset for analyzing word, sentence, or other
        high-dimensional, probably sparse embedding spaces.
    It is an attempt to compile and create some quantifiable metrics which we
        can use to both understand and compare different embedding spaces.
"""
#------------------------------------------------------------------------------#
# IMPORT STUFF FROM ANALYST LAYER PACKAGE:

# Import all classes/functions in Analyst.py:
from Analyst import *
from TestSet2D import *

# Import only one class from Analyst.py:
#from Analyst import Analyst
#from TestSet2d import TestSet2d

# This doesn't work; not the right way to do it:
#import Analyst
#import TestSet2d

#------------------------------------------------------------------------------#
# IMPORTING SUBSIDIARY PACKAGES:

# Non-Recursive way: (subsidiary packages do not need imports in __init__)
#   user may type analyst.Cluster()
from clusters.Cluster import *
from clusters.Node import *

# NOTE: Don't need both ways, but done for convenience!
# Recursive way: (Must have import code in subsidiary package inits, too)
#   forces user outside of analyst folder to type analyst.clusters.Cluster()
import clusters