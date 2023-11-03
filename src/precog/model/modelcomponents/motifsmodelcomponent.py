"""motifsmodelcomponent.py

"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

# Third-Party Packages #
from torch import Tensor

# Local Packages #
from ..statevariables import BaseStateVariables, MotifsStateVariables
from .basemodelcomponent import BaseModelComponent


# Definitions #
# Classes #
class MotifsModelComponent(BaseModelComponent):
    state_variables_type: type[BaseStateVariables] = MotifsStateVariables

    # Magic Methods  #
    # Construction/Destruction
    def __init__(
        self,
        tensor: Tensor | None = None,
        size: Iterable[int] | None = None,
        requires_grad: bool = True,
        state_variables: Mapping[str, Any] | None = None,
        *,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #


        # Parent Attributes #
        super().__init__(init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                tensor=tensor,
                size=size,
                requires_grad=requires_grad,
                state_variables=state_variables,
                **kwargs,
            )

    def construct(
        self,
        tensor: Tensor | None = None,
        size: Iterable[int] | None = None,
        requires_grad: bool = True,
        state_variables: Mapping[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        # Construct Parent #
        super().construct(
            tensor=tensor,
            size=size,
            requires_grad=requires_grad,
            state_variables=state_variables,
            *args,
            **kwargs,
        )
