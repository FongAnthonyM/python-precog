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
from ...basis import TorchModelBasis
from ...architectures.torch.nnmf import NNMFDModule
from ...trainers import NNMFSpikeTrainer
from .torchmodel import TorchModel


# Definitions #
class NNMFDTorchModel(TorchModel):
    # Class Attributes #
    default_bases: ClassVar[dict[str, tuple[type, dict[str, Any]]]] = {
        "W": (TorchModelBasis, {}),
        "H": (TorchModelBasis, {}),
    }
    default_architecture: ClassVar[tuple[type, dict[str, Any]]] = (NNMFDModule, {})
    default_trainer: ClassVar[tuple[type, dict[str]]] = (NNMFSpikeTrainer, {})
