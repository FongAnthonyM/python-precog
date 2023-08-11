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
from baseobjects import BaseObject

# Local Packages #


# Definitions #
class BaseLearner(BaseObject):
    # Magic Methods  #
    # Construction/Destruction
    def __init__(self, *, init=True, **kwargs) -> None:
        # New Attributes #

        if init:
            self.construct()

    def construct(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def update(self):
        pass
