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
from collections.abc import Iterable
from typing import Any

# Third-Party Packages #
from baseobjects import BaseDict

# Local Packages #
from .baseio import BaseIO
from .baseiomultiplexer import BaseIOMultiplexer
from .ioconatiner import IOContainer


# Definitions #
# Classes #
class IOManager(BaseDict, BaseIOMultiplexer):

    default_get: str | None = "get_all"
    default_put: str | None = "put_all"
    default_io: type[BaseIO] = IOContainer

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        io_: dict[str, BaseIO | None] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, **kwargs)

        # Construction #
        if init:
            self.construct(io_=io_, *args, **kwargs)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        io_: dict[str, BaseIO | None] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            io_: The input/outputs to be managed
            *args: Arguments for inheritance.
            **kwargs: Keyword arguments for inheritance.
        """
        if io_ is not None:
            self.update_io(io_)

        super().construct(*args, **kwargs)

    # Mapping
    def update_io(self, __m: Any = {}, /, **kwargs) -> None:
        """Updates this object's items. Nones are replaced with the default io type.

        Args:
            __m: A mapping with io objects which will replace items in this manager.
            **kwargs: Io objects which will replace items in this manager.
        """
        items = (kwargs if __m is None else (__m | kwargs))
        self.update({k: (self.default_io() if v is None else v) for k, v in items.items()})

    # IO Objects
    def create_io(
        self,
        name: str | Iterable[str],
        type_: type[BaseIO] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Creates a new named io object or new io objects from a list of names.

        Args:
            name: The key name of the io to create.
            type_: The type of io to create.
            *args: The arguments for constructing the new io object.
            **kwargs: The keyword arguments for constructing the new io object.
        """
        if type_ is None:
            type_ = self.default_io

        if isinstance(name, str):
            name = (name,)

        for n in name:
            self.data[n] = type_(*args, **kwargs)

    # Get
    def get_item(self, name: str, **kwargs: Any) -> Any:
        """Gets a value from an io object.

        Returns:
            The all items.
        """
        return self.data[name].get(name=name, **kwargs)

    def get_all(self, *args, **kwargs) -> dict[str, Any]:
        """Gets all items from the io object.

        Returns:
            The all items in the io objects.
        """
        return {k: v.get(*args, **kwargs) for k, v in self.data.items()}

    # Put
    def put_item(self, name: str, value: dict[str, Any]) -> None:
        """Put an item into an io object.

        Args:
            name: The key name to the io object to put the item into.
            value: The keyword arguments of the put of the io object.
        """
        self.data[name].put(**value)

    def put_all(self, __m: Any = None, /, **kwargs: Any) -> None:
        """Puts all given keyword io values into their io objects.

        Args:
            __m: A mapping with the io values to put into io objects.
            **kwargs: The io values to put into io objects.
        """
        for k, v in (kwargs if __m is None else (__m | kwargs)).items():
            self.data[k].put(**v)
