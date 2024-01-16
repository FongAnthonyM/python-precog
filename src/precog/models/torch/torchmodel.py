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
    def set_architecture(self, architecture: Module | None) -> None:
        self._architecture = architecture
        if architecture is not None:
            self.submodels.clear()
            self.submodels.update({n: TorchModel(architecture=a) for n, a in architecture.named_modules()})
