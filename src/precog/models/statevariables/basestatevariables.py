"""basestatevaraibles.py

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
from typing import Any

# Third-Party Packages #
from baseobjects import BaseDict

# Local Packages #


# Definitions #
class BaseStateVariables(BaseDict):
    """Abstract base class for state variables."""
    default_state_variables: dict[str, Any] = {}

    # Magic Methods  #
    # Construction/Destruction
    def __init__(self, dict: Any = None, /, *args: Any, **kwargs: Any) -> None:
        # New Attributes #
        dict = self.default_state_variables | (kwargs if dict is None else dict)

        # Parent Attributes #
        super().__init__(dict, *args, init=False, **kwargs)

