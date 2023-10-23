""" baseio.py
An abstract class for IO objects.
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


# Definitions #
# Classes #
class BaseIO(BaseObject):
    """An abstract class for IO objects."""

    # Instance Methods #
    def get(self, *args, **kwargs) -> Any:
        """Gets the requested item.

        Args:
            *args: The arguments for getting the item.
            **kwargs: The keyword arguments for getting the item.

        Returns:
            The requested item.
        """
        raise NotImplementedError

    def put(self, value: Any, *args, **kwargs) -> Any:
        """Puts the requested item.

        Args:
            value: The value to put into this object.
            *args: The arguments for putting the item.
            **kwargs: The keyword arguments for putting the item.
        """
        raise NotImplementedError

