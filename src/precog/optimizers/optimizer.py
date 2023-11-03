""" baseoptimizer.py

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
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject
from torch.optim import Optimizer


# Local Packages #



# Definitions #
# Classes #
class AdpdativeMuOptimizer(Optimizer, BaseObject):

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
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(*args,**kwargs)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Construct Parent #
        super().construct(*args, **kwargs)


    def step(self, closure=...):
        pass



