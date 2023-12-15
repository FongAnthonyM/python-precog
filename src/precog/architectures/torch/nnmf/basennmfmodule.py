"""basennmfmodule.py

"""
# Package Header #
from ....header import *

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
import torch
from torch.nn import Module, Parameter
from torch import Tensor

# Local Packages #


# Definitions #
class BaseNNMFModule(Module):
    # Attributes #
    W: Parameter | None
    H: Parameter | None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        W: Parameter | dict[str, Any] | None = None,
        H: Parameter | dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, **kwargs)

        # Attributes #
        self.W = Parameter(torch.empty(**W)) if isinstance(W, dict) else W
        self.H = Parameter(torch.empty(**H)) if isinstance(H, dict) else H

    # Instance Methods  #
    # Module
    @abstractmethod
    def reconstruct(self, *args, **kwargs) -> Tensor:
        """An abstract method which creates a reconstruction using W and H."""
        raise NotImplementedError

    def forward(self, W: Parameter | None = None, H: Parameter | None = None, *args, **kwargs) -> Tensor:
        """The evaluation of the model which is reconstruction using W and H

        Args:
            H: input activation tensor H.
            W: input template tensor W.

        Returns:
            The reconstructed tensor.
        """
        if H is not None:
            self.H = H
        if W is not None:
            self.W = W

        return self.reconstruct(*args, **kwargs)
