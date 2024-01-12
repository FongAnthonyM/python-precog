""" basemodulearchitecture.py.py

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

# Third-Party Packages #
from torch.nn import Module

# Local Packages #
from ..base import BaseArchitecture


# Definitions #
# Classes #
class BaseModuleArchitecture(Module, BaseArchitecture):
    """A mixin abstract base class for Module and BaseArchitecture."""
    # Class Attributes #
    call_super_init: bool = True
