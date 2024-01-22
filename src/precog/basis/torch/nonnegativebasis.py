"""nonnegativebasis.py

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
from collections.abc import Iterable, Mapping
from typing import Any

# Third-Party Packages #
import torch
from torch import Tensor
from torch.nn import Parameter

# Local Packages #
from .torchmodelbasis import TorchModelBasis
from precog.basis.bases.modelbasis import ModelBasis


# Definitions #
# Classes #
class NonNegativeBasis(TorchModelBasis):
    def create_tensor(self, size: Iterable[int], requires_grad: bool = True, **kwargs: Any) -> Tensor:
        """Create an empty tensor which contains values.

        Args:
            size: The dimensions of the tensor.
            requires_grad:
            **kwargs: The keyword arguments for creating an empty tensor.
        """
        self.tensor = Parameter(torch.rand(*size, **kwargs).abs(), requires_grad=requires_grad)
        return self.tensor
