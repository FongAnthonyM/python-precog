""" basetrainer.py.py

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
import numpy as np

# Local Packages #
from ..operations import BaseOperation


# Definitions #
# Classes #
class BaseTrainer(BaseObject):
    # Attributes #
    block: BaseOperation | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #


        # Parent Attributes #
        super().__init__(init=False, **kwargs)

        # Construct #
        if init:
            self.construct(**kwargs)

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        # Construct Parent #
        super().construct(*args, **kwargs)