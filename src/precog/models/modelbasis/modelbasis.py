"""modelbasis.py

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
from baseobjects import BaseComposite
import numpy as np

# Local Packages #
from ..statevariables import BaseStateVariables


# Definitions #
# Classes #
class ModelBasis(BaseComposite):
    state_variables_type: type[BaseStateVariables] = BaseStateVariables
    default_component_types: dict[str, tuple[type, dict[str, Any]]] = {}

    # Magic Methods  #
    # Construction/Destruction
    def __init__(
        self,
        tensor: np.ndarray | None = None,
        state_variables: Mapping[str, Any] | None = None,
        *,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.tensor: Any = None
        self.state_variables: BaseStateVariables | None = None

        # Parent Attributes #
        super().__init__(init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                tensor=tensor,
                state_variables=state_variables,
                component_kwargs=component_kwargs,
                component_types=component_types,
                components=components,
                **kwargs,
            )

    def construct(
        self,
        tensor: np.ndarray | None = None,
        state_variables: Mapping[str, Any] | None = None,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if state_variables is not None:
            self.state_variables(dict_=state_variables)

        # Construct Parent #
        super().construct(
            component_kwargs=component_kwargs,
            component_types=component_types,
            components=components,
            **kwargs,
        )

    def create_state_variables(self, dict_: Mapping[str, Any], *args, **kwargs) -> BaseStateVariables:
        self.state_variables = self.state_variables_type(dict_, *args, **kwargs)
        return self.state_variables
