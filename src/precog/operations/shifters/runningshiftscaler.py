""" runningshiftscaler.py

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
class RunningShiftScaler(BaseOperation):
    default_input_names: tuple[str, ...] = ("data",)
    default_output_names: tuple[str, ...] = ("shifted_data",)
    default_rescale_shift: str = "mean"
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

        self.previous_mean: np.ndarray | None = None
        self.previous_mean_fill_value: float | int = 0

        self.previous_std: np.ndarray | None = None
        self.previous_std_fill_value: float | int = 1

        self.previous_count: int = 0

        self._forget_factor: float | None = None

        self.forget: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_forget)
        self.rescale_shift: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_rescale_shift)

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
            self.rescale_shift.select(apply_shift)

        if forget_factor is not blank_arg:
            self.forget_factor = forget_factor

        # Construct Parent #
        super().construct(*args, init_io=init_io, steps_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    def create_previous_mean(self, shape, dtype):
        self.previous_mean = np.expand_dims(np.zeros(shape, dtype), self.axis)
        if self.previous_mean_fill_value != 0:
            self.previous_mean.fill(self.previous_mean_fill_value)
        return self.previous_mean

    # Setup
    def setup(self, previous_sum: np.ndarray, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        self.previous_mean = previous_sum

    # Forgetting
    def constant_forget(self, index, **kwargs):
        return 1 / (self.previous_count + (index + 1))

    def exponential_forget(self, **kwargs):
        return self.forget_factor

    # Shift
    def shift_mean(self, data) -> np.ndarray:
        shifted_data = np.empty_like(data)
        slices = [slice(None)] * len(data.shape)
        for i in range(data.shape[self.axis]):
            slices[self.axis] = i
            t_slices = tuple(slices)
            shifted_data[t_slices] = data[t_slices] - self.previous_mean
            self.previous_mean = self.previous_mean + shifted_data[t_slices] * self.forget(index=i)
        self.previous_count += data.shape[-1]
        return shifted_data

    def shift_decaying_mean(self, data) -> np.ndarray:
        # https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
        # a3µn−3 + a2(1−a)xn−2 + a(1−a)xn−1 + (1−a)xn
        slices = [slice(None)] * len(data.shape)
        slices[self.axis] = -1
        t_slices = tuple(slices)

        t_shape = [1] * len(data.shape)
        t_shape[self.axis] = -1

        n_sample = data.shape[self.axis]
        # Calculate Weights
        weights = np.empty(n_sample)
        weights.fill(self.forget_factor)
        weights = np.power(weights, np.arange(n_sample))[::-1] * (1 - weights)

        # Calculate Decayed Means
        # a^n * u0 + a^(n-1) * (1-a) + x
        new_mean = (self.forget_factor ** n_sample) * self.previous_mean + weights.reshape(t_shape) * data
        self.previous_mean = np.expand_dims(new_mean[t_slices], self.axis)

        # Shift Data
        shifted_data = data - new_mean
        return shifted_data

    def mean_and_std(self, data) -> np.ndarray:
        scaled_data = np.empty_like(data)
        slices = [slice(None)] * len(data.shape)
        for i in range(data.shape[self.axis]):
            slices[self.axis] = i
            t_slices = tuple(slices)
            forget_factor = self.forget(index=i)

            # This standardizes the data using values up to (but not including) the current index i
            shifted_data = data[t_slices] - self.previous_mean
            scaled_data[t_slices] = shifted_data / self.previous_std

            # This updates the statistics using values up to (and including) the current index i
            self.previous_mean = self.previous_mean + shifted_data * forget_factor
            self.previous_std = forget_factor * (self.previous_std + (1-forget_factor) * shifted_data**2)
        self.previous_count += data.shape[-1]
        return scaled_data

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        if self.previous_mean is None:
            shape = list(data.shape)
            shape.pop(self.axis)
            self.create_previous_mean(shape, data.dtype)
        return self.rescale_shift(data)
