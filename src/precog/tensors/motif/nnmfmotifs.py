"""nnmfmotifs.py

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
from collections.abc import Iterable
from abc import abstractmethod
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject
import torch
from torch import Tensor
from torch.nn import Parameter

# Local Packages #


# Definitions #
class NNMFMotifs(BaseObject):
    # Magic Methods  #
    # Construction/Destruction
    def __init__(
        self,
        tensor: Tensor | None = None,
        size: Iterable[int] | None = None,
        requires_grad: bool = True,
        *,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.tensor: Tensor | None = None
        self.motifs: list = []

        # Parent Attributes #
        super().__init__(init=False, **kwargs)

        # Construct #
        if init:
            self.construct(tensor=tensor, size=size, requires_grad=requires_grad)

    def construct(
        self,
        tensor: Tensor | None = None,
        size: Iterable[int] | None = None,
        requires_grad: bool = True,
        **kwargs: Any,
    ) -> None:
        if tensor is not None:
            self.tensor = tensor
        elif size is not None:
            self.create_tensor(size=size, requires_grad=requires_grad)

        if self.tensor is not None:
            self.construct_motifs()

    def create_tensor(self, size: Iterable[int], requires_grad: bool = True, **kwargs: Any) -> None:
        """Create an empty tensor which contains the motif values.

        Args:
            size: The dimensions of the tensor.
            requires_grad:
            **kwargs: The keyword arguments for creating an empty tensor.
        """
        self.tensor = Parameter(torch.empty(*size, **kwargs), requires_grad=requires_grad)

    def construct_motifs(self):
        pass

