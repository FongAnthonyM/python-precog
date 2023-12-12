"""basennmflearner.py

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
import torch
from torch import Tensor
from torch.nn import Parameter

# Local Packages #
from ...basis import ModelBasis
from ...models import BaseModel, NNMFModel
from ..base import BaseTorchLearner


# Definitions #
class BaseNNMFLearner(BaseTorchLearner):
    default_model = NNMFModel
    default_basis_names = {"W": "W", "H": "H"}

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

    @property
    def W(self) -> ModelBasis:
        return self.model.bases[self.basis_names["W"]]

    @W.setter
    def W(self, value: ModelBasis | Tensor) -> None:
        if isinstance(value, ModelBasis):
            self.model.bases[self.basis_name["W"]] = value
        else:
            self.model.bases[self.basis_name["W"]].tensor = value

    @property
    def H(self) -> ModelBasis:
        return self.model.bases[self.basis_names["H"]]

    @H.setter
    def H(self, value: ModelBasis | Tensor) -> None:
        if isinstance(value, ModelBasis):
            self.model.bases[self.basis_name["H"]] = value
        else:
            self.model.bases[self.basis_name["H"]].tensor = value

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

    @abstractmethod
    def reconstruct(self, *args, **kwargs) -> Tensor:
        """An abstract method which creates a reconstruction using W and H."""
        raise NotImplementedError

    def forward(self, H: Tensor | None = None, W: Tensor | None = None, *args, **kwargs) -> Tensor:
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
