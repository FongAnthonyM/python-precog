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
from ..models import BaseModel


# Definitions #
class BaseLearner(BaseObject):
    # Magic Methods  #
    # Construction/Destruction
    def __init__(self, model: BaseModel | None = None, *, init=True, **kwargs) -> None:
        # New Attributes #
        self.model: BaseModel | None = None

        if init:
            self.construct()

    def construct(self, model: BaseModel | None = None, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def update(self):
        pass
