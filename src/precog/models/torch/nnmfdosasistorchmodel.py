"""nnmfdoasistorchmodel.py

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
from .nnmfdtorchmodel import NNMFDTorchModel


# Definitions #
class NNMFDOASISTorchModel(NNMFDTorchModel):
    # Class Attributes #
    default_bases: ClassVar[dict[str, tuple[type, dict[str, Any]]]] = {
        "W": (TorchModelBasis, {}),
        "H": (TorchModelBasis, {}),
        "S": (TorchModelBasis, {}),
    }
