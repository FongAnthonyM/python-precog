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
# Constants
blank_arg = object()


# Classes #
class MeanMemory(BaseOperation):
    default_input_names: tuple[str, ...] = ("data",)
    default_output_names: tuple[str, ...] = ("shifted_data",)
    default_apply_shift: str = "mean"
    default_forget: str | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        apply_shift: str | None = None,
        forget_factor: float | None | object = blank_arg,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.axis: int = 0
        self.mem_sum: np.ndarray | None = None
        self.mem_weight: np.ndarray | None = None

        self.apply_shift: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_apply_shift)
        self.forget: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_forget)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                apply_shift=apply_shift,
                forget_factor=forget_factor,
                axis=axis,
                init_io=init_io,
                steps_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )
            
    @property
    def forget_factor(self) -> float:
        return self._forget_factor

    @forget_factor.setter
    def forget_factor(self, value) -> None:
        if self.forget.selected in {None, "constant_forget", "exponential_forget"}:
            if value is None:
                self.forget.select("constant_forget")
            else:
                self.forget.select("exponential_forget")
        self._forget_factor = value

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        apply_shift: str | None = None,
        forget_factor: float | None | object = blank_arg,
        axis: int | None = None,
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

        if apply_shift is not None:
            self.apply_shift.select(apply_shift)

        if forget_factor is not blank_arg:
            self.forget_factor = forget_factor

        # Construct Parent #
        super().construct(*args, init_io=init_io, steps_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    def create_previous_sum(self, shape, dtype):
        self.previous_sum = np.zeros(shape, dtype)
        if self.previous_sum_fill_value != 0:
            self.previous_sum.fill(self.previous_sum_fill_value)
        return self.previous_sum

    # Setup
    def setup(self, previous_sum: np.ndarray, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        self.previous_sum = previous_sum

    # Forgetting
    def constant_forget(self, index, **kwargs):
        return 1 / (self.mem_count + (index + 1))

    def exponential_forget(self, **kwargs):
        return self.forget_factor

    # Shift
    def mean(self, data) -> np.ndarray:
        axis_swap = (self.axis, -1)
        axis_unswap = (-1, self.axis)
        data = np.moveaxis(data, axis_swap[0], axis_swap[1])
        shifted_data = np.zeros_like(data)

        for i in range(data.shape[-1]):
            shifted_data[..., i] = data[..., i] - self.previous_sum
            self.previous_sum += shifted_data[..., i] * self.forget(index=i)
        self.mem_count += data.shape[-1]

        shifted_data = np.moveaxis(shifted_data, axis_unswap[0], axis_unswap[1])

        return shifted_data

    def mean_alternative(self, data) -> np.ndarray:
        shifted_data = np.empty_like(data)
        slices = [slice(None)] * len(data.shape)
        for i in range(data.shape[self.axis]):
            slices[self.axis] = i
            t_slices = tuple(slices)
            shifted_data[t_slices] = data[t_slices] - self.previous_sum
            self.previous_sum += shifted_data[t_slices] * self.forget(index=i)
        self.mem_count += data.shape[-1]
        return shifted_data

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
