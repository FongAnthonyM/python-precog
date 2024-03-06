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
from typing import ClassVar, Any, Generator

# Third-Party Packages #
from torch.nn import Module

# Local Packages #
from ....basis.torch import TorchModelBasis
from .basemodulearchitecture import BaseModuleArchitecture
from .modulewrapperarchitecture import ModuleWrapperArchitecture


# Definitions #
# Classes #
class ModuleArchitecture(Module, BaseModuleArchitecture):
    """A mixin abstract bases class for Module and BaseArchitecture."""
    # Class Attributes #
    call_super_init: bool = True

    # Attributes #
    wrapper_architecture_type: type[ModuleWrapperArchitecture] = ModuleWrapperArchitecture

    # Instance Methods #
    # Bases
    def get_bases(self) -> dict[str, TorchModelBasis]:
        for n, p in self.named_parameters(recurse=False):
            if n not in self._bases:
                self._bases[n] = self.basis_type(tensor=p)

        return self._bases

    # Subarchitecture
    def iter_subarchitectures(
        self,
        memo: dict[Any, "BaseModuleArchitecture"] | None = None,
        recursive: bool = False,
        rebuild: bool = False,
    ) -> Generator[tuple[str, "BaseModuleArchitecture"], None, None]:
        if memo is None:
            memo = {}

        if self not in memo:
            memo[self] = self

        for name, module in self._modules.items():
            if module is not None:
                if module not in memo:
                    if not isinstance(module, BaseModuleArchitecture):
                        module_a = self.wrapper_architecture_type(module=module)
                    else:
                        module_a = module

                    if recursive:
                        module_a.get_subarchitectures(memo=memo, recursive=recursive, rebuild=rebuild)

                    memo[module] = module_a
                    yield name, module_a
                else:
                    yield name, memo[module]
