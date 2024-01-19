""" remapper.py

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
import numpy as np

# Local Packages #
from .operation import BaseOperation


# Definitions #
# Classes #
class Remapper(BaseOperation):
    default_input_names: tuple[str, ...] = ("data", "map_matrix")
    default_output_names: tuple[str, ...] = ("remapped_data",)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        map_matrix: np.ndarray | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.axis: int = 1
        self.map_matrix = None

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                map_matrix=map_matrix,
                axis=axis,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        map_matrix: np.ndarray | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            map_matrix: The remap matrix to apply.
            axis: The axis to remap along.
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        if map_matrix is not None:
            self.map_matrix = map_matrix

        if axis is not None:
            self.axis = axis

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Setup
    def setup(self, map_matrix: np.ndarray | None = None, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        if map_matrix is not None:
            self.map_matrix = map_matrix

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, map_matrix: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to remap.
            map_matrix: The remap matrix to apply.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        if map_matrix is not None:
            self.map_matrix = map_matrix

        moved_data = np.moveaxis(data, self.axis, -1)
        remapped = moved_data @ self.map_matrix
        return np.moveaxis(remapped, -1, self.axis)
