""" basemodulearchitecture.py.py

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
from typing import Any

# Third-Party Packages #
from torch.nn import Module

# Local Packages #
from ....basis import ModelBasis, TorchModelBasis
from .basemodulearchitecture import BaseModuleArchitecture


# Definitions #
# Classes #
class ModuleWrapperArchitecture(BaseModuleArchitecture):
    """A mixin abstract bases class for Module and BaseArchitecture."""
    # Attributes #
    module: Module | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        module: Module | None = None,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        *args: Any,
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
                module=module,
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
        module: Module | None = None,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # Assign New #
        if module is not None:
            self.module = module

        # Construct Parent #
        super().construct(
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                **kwargs,
            )

    # Bases
    def get_bases(self) -> dict[str, TorchModelBasis]:
        for n, p in self.module.named_parameters(recurse=False):
            if n not in self._bases:
                self._bases[n] = self.basis_type(tensor=p)

        return self._bases

    # Subarchitecture
    def get_subarchitectures(self) -> dict[str, BaseModuleArchitecture]:
        subarchitectures = {}
        for name, module in self.module.named_modules():
            if isinstance(module, BaseModuleArchitecture):
                subarchitectures[name] = module
            else:
                subarchitectures[name] = self.wrapper_architecture_type(module=module)
        return subarchitectures


# Cyclic
ModuleWrapperArchitecture.wrapper_architecture_type = ModuleWrapperArchitecture
