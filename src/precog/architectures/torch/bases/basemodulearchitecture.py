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
from ....basis import TorchModelBasis
from ...bases import BaseArchitecture


# Definitions #
# Classes #
class BaseModuleArchitecture(BaseArchitecture):
    """An abstract bases class for BaseArchitecture using Module."""
    # Attributes #
    basis_type: type[TorchModelBasis] = TorchModelBasis
    wrapper_architecture_type: type["BaseModuleArchitecture"]
    _bases: dict[str, TorchModelBasis]

    # Properties #
    @property
    def bases(self) -> dict[str, TorchModelBasis]:
        return self.get_bases()

    @bases.setter
    def bases(self, value: dict[str, TorchModelBasis]) -> None:
        self._bases = value

    @property
    def subarchitectures(self) -> dict[str, "BaseModuleArchitecture"]:
        return self.get_subarchitectures()

    # Instance Methods #
    # Bases
    def get_bases(self) -> dict[str, TorchModelBasis]:
        """"""

    # Subarchitecture
    def get_subarchitectures(self) -> dict[str, "BaseModuleArchitecture"]:
        """"""
