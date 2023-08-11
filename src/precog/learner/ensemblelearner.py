"""baselearner.py

"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from abc import abstractmethod

# Third-Party Packages #

# Local Packages #
from .baselearner import BaseLearner


# Definitions #
class EnsembleLearner(BaseLearner):


    def update(self):
        self.group.update_all()
