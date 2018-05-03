"""
Analyst
-------
    Interface for toolset for analyzing word, sentence, or other
        high-dimensional, probably sparse embedding spaces.
    It is an attempt to compile and create some quantifiable metrics which we
        can use to both understand and compare different embedding spaces.
"""


#------------------------------------------------------------------------------#
# Define list of modules contained in the analyst package (modules are files):
#__all__ = ['analyst', 'test_set_2d']


#------------------------------------------------------------------------------#
# IMPORT STUFF FROM ANALYST LAYER PACKAGE:
#   Note, imports in package __init__ automatically look at subfolders and
#   modules (files) instead of having to say analyst.module

# Import all classes/functions in Analyst.py:
from .analyst import * # was from Analyst import *
from .test_set_2d import *
#from analyst.analyst import Analyst
#from analyst.test_set_2d import TestSet2D

# Import only one class at a time from Analyst.py:
#from analyst import Analyst
#from test_set_2d import TestSet2d

# This doesn't work; these are modules, not classes:
#import analyst
#import test_set_2d

#------------------------------------------------------------------------------#
# IMPORTING SUBSIDIARY PACKAGES:

# Non-Recursive way: (subsidiary packages do not need imports in __init__)
#   user may type analyst.Cluster()
from .clustertypes import *
from .clustertypes.cluster import *
from .clustertypes.node import *
from .clusterizers.clusterizer import *

# NOTE: Don't need both ways, but done for convenience!
# Recursive way: (Must have import code in subsidiary package inits, too)
#   forces user outside of analyst folder to type analyst.clusters.Cluster()
#import clusters
