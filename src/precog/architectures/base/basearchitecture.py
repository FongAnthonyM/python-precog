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
from abc import abstractmethod
from typing import ClassVar, Any

# Third-Party Packages #

# Local Packages #
from ...basis import ModelBasis, BasisContainer


# Definitions #
# Classes #
class BaseArchitecture(BasisContainer):
    """An abstract base class for model architectures."""
