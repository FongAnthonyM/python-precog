""" basetrainer.py.py

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
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject
# Local Packages #
from ...basis import ModelBasis, BasisContainer
from ...operations import BaseOperation


# Definitions #
# Classes #
class BaseTrainer(BasisContainer):
    """An abstract base class for model trainers."""
    # Attributes #
    subtrainers: dict[str, "BaseTrainer"]
