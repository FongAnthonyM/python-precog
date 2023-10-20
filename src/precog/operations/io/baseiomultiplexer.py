""" iomanager.py

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
from baseobjects.functions import CallableMultiplexObject, MethodMultiplexer

# Local Packages #
from .baseio import BaseIO


# Definitions #
# Classes #
class BaseIOMultiplexer(BaseIO, CallableMultiplexObject):

    default_get: str | None = None
    default_put: str | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # New Attributes #
        self.get: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_get)
        self.put: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_put)

        # Parent Attributes #
        super().__init__(*args, **kwargs)

