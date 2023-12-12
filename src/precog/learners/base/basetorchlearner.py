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
from baseobjects import BaseObject
from torch.nn import Module

# Local Packages #
from precog.models import BaseModel
from .baselearner import BaseLearner


# Definitions #
# Classes #
class BaseTorchLearner(Module, BaseLearner):
    default_model: tuple[type, dict[str, Any]] = ()
    default_basis_names: dict[str, str] = {}

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        model: BaseModel | None = None,
        basis_names: dict[str, str,] | None = None,
        *args: Any,
        register_bases: bool = True,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(model=model, basis_names=basis_names, register_bases=register_bases, **kwargs)

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        model: BaseModel | None = None,
        basis_names: dict[str, str,] | None = None,
        *args: Any,
        register_bases: bool = False,
        **kwargs: Any,
    ) -> None:
        # Construct Parent #
        super().construct(model=model, basis_names=basis_names, **kwargs)

        if register_bases:
            self.register_bases()

    def register_bases(self):
        for basis_name in self.basis_names.values():
            self.register_parameter(basis_name, self.model.bases[basis_name].tensor)
