"""baselearner.py

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
from baseobjects.functions import CallableMultiplexObject

# Local Packages #
from precog.models import BaseModel


# Definitions #
# Classes #
class BaseLearner(CallableMultiplexObject):
    default_model: tuple[type, dict[str, Any]] = ()
    default_basis_names: dict[str, str] = {}

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        model: BaseModel | None = None,
        basis_names: dict[str, str,] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.model: BaseModel | None = None
        self.basis_names = self.default_basis_names.copy()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(model=model, basis_names=basis_names, **kwargs)

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        model: BaseModel | None = None,
        basis_names: dict[str, str,] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if model is not None:
            self.model = model

        if basis_names is not None:
            self.basis_names.update(basis_names)

        # Construct Parent #
        super().construct(*args, **kwargs)
