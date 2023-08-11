"""nnmflearner.py

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
from typing import Any

# Third-Party Packages #

# Local Packages #
from .baselearner import BaseLearner


# Definitions #
class NNMFLearner(BaseLearner):
    # Magic Methods  #
    # Construction/Destruction
    def __init__(self, *, init=True, **kwargs) -> None:
        # New Attributes #

        # Parent Attributes #
        super().__init__(init=False)

        # Construct #
        if init:
            self.construct()

    def construct(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self):
        pass
