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
from typing import ClassVar

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
    def get_subarchitectures(self) -> dict[str, BaseModuleArchitecture]:
        subarchitectures = {}
        for name, module in self.named_modules():
            if isinstance(module, BaseModuleArchitecture):
                subarchitectures[name] = module
            else:
                subarchitectures[name] = self.wrapper_architecture_type(module=module)
        return subarchitectures
