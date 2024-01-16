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
from typing import ClassVar, Any

# Third-Party Packages #
from torch.nn import Parameter
from torch import Tensor

# Local Packages #
from ....basis import ModelBasis
from precog.architectures.torch.base.basemodulearchitecture import BaseModuleArchitecture


# Definitions #
class BaseNNMFModule(BaseModuleArchitecture):
    # Class Attributes #
    default_bases: ClassVar[dict[str, tuple[type, dict[str, Any]]]] = {"H": (), "W": ()}
    
    # Properties #
    @property
    def W(self) -> ModelBasis | None:
        return self.bases.get("W", None)
    
    @W.setter
    def W(self, value: ModelBasis) -> None:
        self.bases["W"] = value

    @property
    def H(self) -> ModelBasis | None:
        return self.bases.get("H", None)

    @H.setter
    def H(self, value: ModelBasis) -> None:
        self.bases["H"] = value

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        W: ModelBasis | Parameter | dict[str, Any] | None = None,
        H: ModelBasis | Parameter | dict[str, Any] | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                W=W,
                H=H,
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        W: ModelBasis | Parameter | dict[str, Any] | None = None,
        H: ModelBasis | Parameter | dict[str, Any] | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # New Setup #
        if bases_kwargs is None:
            bases_kwargs = {}
            if isinstance(W, dict):
                bases_kwargs["W"] = W
            if isinstance(H, dict):
                bases_kwargs["H"] = H
            if not bases_kwargs:
                bases_kwargs = None
        else:
            if "W" not in bases_kwargs and isinstance(W, dict):
                bases_kwargs["W"] = W
            
            if "H" not in bases_kwargs and isinstance(H, dict):
                bases_kwargs["H"] = H
        
        # Construct Parent #
        super().construct(
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                **kwargs,
            )

        # Construct New #
        if isinstance(W, ModelBasis):
            self.bases["W"] = W
        elif isinstance(W, Parameter):
            self.bases["W"].tensor = W
    
        if isinstance(H, ModelBasis):
            self.bases["H"] = H
        elif isinstance(H, Parameter):
            self.bases["H"].tensor = H
    
    # Module
    @abstractmethod
    def reconstruct(self, *args, **kwargs) -> Tensor:
        """An abstract method which creates a reconstruction using W and H."""
        raise NotImplementedError

    def forward(self, W: ModelBasis | None = None, H: ModelBasis | None = None, *args, **kwargs) -> Tensor:
        """The evaluation of the model which is the reconstruction using W and H

        Args:
            H: input activation tensor H.
            W: input template tensor W.

        Returns:
            The reconstructed tensor.
        """
        if H is not None:
            self.bases["H"] = H
        if W is not None:
            self.bases["W"] = W

        return self.reconstruct(*args, **kwargs)
