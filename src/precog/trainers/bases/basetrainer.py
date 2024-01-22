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

# Local Packages #
from ...basis import BasisContainer


# Definitions #
# Classes #
class BaseTrainer(BasisContainer):
    """An abstract bases class for model trainers."""
    # Attributes #
    local_state_variables: dict[str, Any]
    subtrainers: dict[str, "BaseTrainer"]

    # Properties #
    @property
    def state_variables(self) -> dict[str, Any]:
        return self.get_state_variables()

    @state_variables.setter
    def state_variables(self, value: dict[str, Any]) -> None:
        self.local_state_variables = value

    # Instance Methods  #
    # State Variables
    def get_state_variables(self) -> dict[str, Any]:
        return {"local": self.local_state_variables}
