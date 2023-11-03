"""basemodel.py

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
from baseobjects import BaseObject
from baseobjects.wrappers import StaticWrapper
import torch
from torch import Tensor
from torch.nn import Parameter

# Local Packages #
from ..statevariables import BaseStateVariables


# Definitions #
# Classes #
class BaseModelComponent(StaticWrapper):
    _wrapped_types: list[Any] = [Tensor]
    _wrap_attributes: list[str] = ["tensor"]
    state_variables_type: type[BaseStateVariables] = BaseStateVariables

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
        self._tensor: Tensor | None = None
        self.state_variables: BaseStateVariables | None = None

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
        if tensor is not None:
            self._tensor = tensor
        elif size is not None:
            self.create_tensor(size=size, requires_grad=requires_grad)

        if state_variables is not None:
            self.state_variables(dict_=state_variables)

        # Construct Parent #
        super().construct(*args, **kwargs)

    def create_tensor(self, size: Iterable[int], requires_grad: bool = True, **kwargs: Any) -> Tensor:
        """Create an empty tensor which contains values.

        Args:
            size: The dimensions of the tensor.
            requires_grad:
            **kwargs: The keyword arguments for creating an empty tensor.
        """
        self._tensor = Parameter(torch.empty(*size, **kwargs), requires_grad=requires_grad)
        return self._tensor

    def create_state_variables(self, dict_: Mapping[str, Any], *args, **kwargs) -> BaseStateVariables:
        self.state_variables = self.state_variables_type(dict_, *args, **kwargs)
        return self.state_variables
