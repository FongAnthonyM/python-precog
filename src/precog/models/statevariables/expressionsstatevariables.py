"""expressionsstatevariables.py

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

# Local Packages #
from .basestatevariables import BaseStateVariables


# Definitions #
class ExpressionsStateVariables(BaseStateVariables):
    default_state_variables: dict[str, Any] = {}
