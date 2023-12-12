"""basemodel.py

"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from typing import Any

# Third-Party Packages #

# Local Packages #
from ..basis import ModelBasis
from .base import BaseModel


# Definitions #
class OASISModel(BaseModel):
    default_bases: tuple[tuple[str, type, dict[str, Any]], ...] = (
        ("H", ModelBasis, {}),
        ("S", ModelBasis, {}),
    )
