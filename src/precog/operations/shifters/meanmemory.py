""" meanmemory.py

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
from abc import abstractmethod
from typing import Any

# Third-Party Packages #
from baseobjects.functions import MethodMultiplexer
import numpy as np

# Local Packages #
from ..operation import BaseOperation


# Definitions #
# Classes #
class MeanMemory(BaseOperation):
    default_input_names: tuple[str, ...] = ("data", "weight",)
    default_output_names: tuple[str, ...] = ("shifted_data",)
    default_apply_shift: str = "mean"

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        apply_shift: str | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        steps_up: bool = True,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.axis: int = 0
        self.mem_sum: np.ndarray | None = None
        self.mem_weight: np.ndarray | None = None

        self.apply_shift: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_apply_shift)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                apply_shift=apply_shift,
                axis=axis,
                *args,
                init_io=init_io,
                steps_up=steps_up,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        apply_shift: str | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        setup: bool = True,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            setup: Determines if setup will run during this construction
            **kwargs: Keyword arguments for inheritance.
        """
        if axis is not None:
            self.axis = axis

        if apply_shift is not None:
            self.apply_shift.select(apply_shift)

        # Construct Parent #
        super().construct(*args, init_io=init_io, setup=setup, **kwargs)

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

    # Shift
    def mean(self, data, weight) -> np.ndarray:
        temp_sum = self.mem_sum + np.cumsum(data, axis=self.axis)
        temp_sum2 = np.append((self.mem_sum, temp_sum[1:]))
        temp_weight = self.mem_weight + weight

        self.mem_sum = temp_sum
        # 1. compute the new mem_sum and new mem_weight (hold in temp).
        # 2. perform the shifting along desired axis (using old mem attributes).
        # 3. update the mem_sum and mem_weight.

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, weight: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        return self.apply_shift(data, weight)
