""" recenter.py

"""
# Package Header #
from precog.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from typing import Any

# Third-Party Packages #
from baseobjects import MethodMultiplexer
import numpy as np

# Local Packages #
from ..operation import BaseOperation


# Definitions #
# Classes #
class Recenter(BaseOperation):
    default_input_names: tuple[str, ...] = ("data",)
    default_output_names: tuple[str, ...] = ("r_data",)
    default_find_center: str = "find_peak"
    default_recenter: str = "roll"

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        axis: int | tuple[int, int] | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.axis: int | None = 0

        self.find_center = MethodMultiplexer(instance=self, select=self.default_find_center)
        self.recenter = MethodMultiplexer(instance=self, select=self.default_recenter)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                axis=axis,
                *args,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        axis: int | tuple[int, int] | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        if axis is not None:
            self.axis = axis

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

    # Find Center
    def find_peak(self, data: np.ndarray):
        return data.argmax()

    def find_center_mass(self: np.ndarray):
        pass

    # Recenter
    def roll(self, data: np.ndarray):
        return np.roll(data, data.shape[self.axis] // 2 - self.find_center()[self.axis], axis=self.axis)

    def reflect_roll(self, data: np.ndarray):
        slices = [slice(s) for s in data.shape]
        axis_len = data.shape[self.axis]
        shift = axis_len // 2 - self.find_center()[self.axis]
        width = [(0, 0)] * len(data.shape)
        width[self.axis] = (shift, 0) if shift > 0 else (0, -shift)
        if shift > 0:
            width[self.axis] = (shift, 0)
        else:
            width[self.axis] = (0, -shift)
            slices[self.axis] = slice(1 - axis_len)

        return np.pad(data, width, mode="reflect")[tuple(slices)]

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        return self.recenter(data)
