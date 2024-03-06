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
from ...bases import BaseArchitecture


# Definitions #
# Classes #
class BaseModuleArchitecture(BaseArchitecture):
    """An abstract bases class for BaseArchitecture using Module."""
    # Attributes #
    basis_type: type[TorchModelBasis] = TorchModelBasis
    wrapper_architecture_type: type["BaseModuleArchitecture"]
    _bases: dict[str, TorchModelBasis]

    # Attributes #
    _subarchitectures: dict[str, "BaseModuleArchitecture"]

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

    @subarchitectures.setter
    def subarchitectures(self, value: dict[str, "BaseModuleArchitecture"]) -> None:
        self._subarchitectures = value

    # Instance Methods #
    # Bases
    def get_bases(self) -> dict[str, TorchModelBasis]:
        """"""

    # Subarchitecture
    def iter_subarchitectures(
        self,
        memo: dict[Any, "BaseModuleArchitecture"] | None = None,
        recursive: bool = False,
        rebuild: bool = False,
    ) -> Generator[tuple[str, "BaseModuleArchitecture"], None, None]:
        """ """

    def get_subarchitectures(
        self,
        memo: dict[Any, "BaseModuleArchitecture"] | None = None,
        recursive: bool = False,
        rebuild: bool = False,
    ) -> dict[str, "BaseModuleArchitecture"]:
        if not rebuild:
            if memo is not None:
                memo = memo | {getattr(a, "module", a): a for a in self._subarchitectures.values()}
            else:
                memo = {getattr(a, "module", a): a for a in self._subarchitectures.values()}

        self._subarchitectures.update({k: v for k, v in self.iter_subarchitectures(memo, recursive, rebuild)})
        return self._subarchitectures
