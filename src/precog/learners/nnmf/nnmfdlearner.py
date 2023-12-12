"""nnmfdlearner.py

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
from typing import Any

# Third-Party Packages #
from baseobjects.functions import CallableMultiplexer, FunctionRegister
import torch
from torch import Tensor
from torch.nn.functional import conv1d, conv2d, conv3d
from torch.nn import Parameter

# Local Packages #
from ...basis import ModelBasis
from ...models import BaseModel, NNMFModel
from .basennmflearner import BaseNNMFLearner


# Definitions #
class NNMFDLearner(BaseNNMFLearner):
    conv_register: FunctionRegister = FunctionRegister(
        conv1d=conv1d,
        conv2d=conv2d,
        conv3d=conv3d,
    )

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
        self.conv: CallableMultiplexer = CallableMultiplexer(register=self.conv_register)
        self._padding_size: tuple[int, ...] = ()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(model=model, basis_names=basis_names, register_bases=register_bases, **kwargs)

    @property
    def padding_size(self) -> tuple[int, ...]:
        if not self._padding_size:
            self.convolution_setup()
        return self._padding_size

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
        super().construct(model=model, basis_names=basis_names, register_bases=register_bases, **kwargs)

        if self.model is not None:
            self.convolution_setup()

    def convolution_setup(self):
        shape = self.W.tensor.shape
        ndim = len(shape)
        self._padding_size = (shape[d] - 1 for d in range(2, len(shape)))
        if ndim == 3:
            self.conv.select("conv1d")
        elif ndim == 4:
            self.conv.select("conv2d")
        elif ndim == 5:
            self.conv.select("conv3d")

    # Instance Methods  #
    def reconstruct(self, *args, **kwargs) -> Tensor:
        """Creates a reconstruction by taking the product of W and H."""
        return self.conv(self.H.tensor, self.W.tensor, padding=self.padding_size)
