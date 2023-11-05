"""nnmfoasisemodel.py

"""
# Package Header #
from precog.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from abc import abstractmethod
from copy import deepcopy
from typing import Any

# Third-Party Packages #

# Local Packages #
from .modelbasis import ModelBasis
from .base import CompositeModel
from .nnmfmodel import NNMFModel
from .oasismodel import OASISModel


# Definitions #
# Classes #
class NNMFOASISModel(CompositeModel):
    default_bases: tuple[tuple[str, type, dict[str, Any]], ...] = (
        ("W", ModelBasis, {}),
        ("H", ModelBasis, {}),
        ("S", ModelBasis, {}),
    )
    default_models: tuple[tuple[str, type, dict[str, Any]], ...] = (
        ("NNMF", NNMFModel, {}),
        ("OASIS", OASISModel, {},),
    )
    default_basis_map: dict[str, dict[str, tuple[str, ...] | str]] = {
        "W": {"NNMF": "W"},
        "H": {"NNMF": "H", "OASIS": "H"},
        "S": {"OASIS": "S"},
    }
