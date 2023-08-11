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
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject

# Local Packages #
from .baselearner import BaseLearner


# Definitions #
class BaseLearnerGroup(BaseObject):
    # Magic Methods  #
    # Construction/Destruction
    def __init__(self, *, init=True, **kwargs) -> None:
        # New Attributes #
        self.learners: dict[str, BaseLearner] = {}

        if init:
            self.construct()

    def construct(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def update(self):
        pass
