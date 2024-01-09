# -*- coding: utf-8 -*-

"""Top-level package for the Comprehensive Dynamic Time Warp library.

Please see the help for the dtw.dtw() function which is the package's
main entry point.

"""

__author__ = """Toni Giorgino"""
__email__ = 'toni.giorgino@gmail.com'
__version__ = '1.3.0'

# There are no comments in this package because it mirrors closely the R sources.

# List of things to export on "from dtw import *"
from dtw import *
from stepPattern import *
from countPaths import *
from dtwPlot import *
from mvm import *
from warp import *
from warpArea import *
from window import *

import sys

# Only print in interactive mode
# https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode/2356427#2356427
import __main__ as main
if bool(getattr(sys, 'ps1', sys.flags.interactive)):
    print("""Importing the dtw module. When using in academic works please cite:
  T. Giorgino. Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package.
  J. Stat. Soft., doi:10.18637/jss.v031.i07.\n""")

