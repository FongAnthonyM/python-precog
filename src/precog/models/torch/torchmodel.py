"""torchmodel.py

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
from typing import ClassVar, Any

# Third-Party Packages #
from torch.nn.modules.module import Module

# Local Packages #
from ...basis import ModelBasis, TorchModelBasis
from ..base import BaseModel


# Definitions #
# Classes #
class TorchModel(BaseModel):
    # Attributes #
    _architecture: Module | None = None

    # Properties
    @property
    def architecture(self) -> Module | None:
        return self._architecture

    @architecture.setter
    def architecture(self, value: Module | None) -> None:
        self.set_architecture(value)

    # Instance Methods  #
    # Architecture
    def set_architecture(self, architecture: Module | None, retain_bases: bool = True) -> None:
        self._architecture = architecture
        if architecture is not None:
            self.submodels.clear()
            self.submodels.update({n: TorchModel(architecture=a) for n, a in architecture.named_modules()})

            if retain_bases:
                for name, parameter in architecture.named_parameters(recurse=False):
                    if (basis := self._bases.get(name, None)) is not None:
                        setattr(architecture, name, basis.tensor)
                    else:
                        self._bases[name] = TorchModelBasis(tensor=parameter)
            else:
                self._bases.clear()
                bases = {n: TorchModelBasis(tensor=p) for n, p in architecture.named_parameters(recurse=False)}
                self._bases.update(bases)

    def set_architecture_bases(self) -> None:
        for name in self.architecture_bases:
            setattr(self.architecture, name, self._bases[name].tensor)
