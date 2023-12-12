"""basebasismodifier.py

"""
# Package Header #
from precog.header import *

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
from ..modelbasis import ModelBasis


# Definitions #
class BaseBasisModifier(BaseObject):
    default_state_variables: dict[str, Any] = {}
    precision: float = np.finfo(np.float64).precision

    @classmethod
    def create_state_variables(cls, **kwargs):
        return cls.default_state_variables | kwargs

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.bases: dict[str, ModelBasis] = {}
        self.state_variables: dict = self.create_state_variables()

        # Parent Attributes #
        super().__init__(init=False, **kwargs)

        # Construct #
        if init:
            self.construct(bases=bases, state_variables=state_variables, **kwargs)

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if bases is not None:
            self.bases.update(bases)

        if state_variables is not None:
            self.state_variables = state_variables

        # Construct Parent #
        super().construct(*args, **kwargs)
