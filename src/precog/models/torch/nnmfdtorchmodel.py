"""nnmfdtorchmodel.py

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

# Local Packages #
from ...architectures.torch import NNMFDModule
from ...trainers.torch import NNMFSpikeTrainer
from ..bases import BaseModel


# Definitions #
class NNMFDTorchModel(BaseModel):
    # Class Attributes #
    default_architecture: ClassVar[tuple[type, dict[str, Any]]] = (NNMFDModule, {"create_defaults": True})
    default_trainer: ClassVar[tuple[type, dict[str]]] = (NNMFSpikeTrainer, {"create_defaults": True})

    # Instance Methods #
    # Architecture
    def build_architecture(self, *args: Any, **kwargs: Any) -> None:
        pass

    # Trainer
    def build_trainer(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.W_architecture = self.architecture
        self.trainer.H_architecture = self.architecture
