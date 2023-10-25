""" linelengthoperation.py

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
from typing import Any, Callable

# Third-Party Packages #
import numpy as np
from scipy.signal import convolve
from scipy.signal import hann

# Local Packages #
from .basefeature import BaseFeature


# Definitions #
# Classes #
class LineLengthOperation(BaseFeature):
    default_axis: int = 0
    default_squared_estimator: bool = False
    default_window_len: int = 0
    default_window_type: Callable = hann

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        steps_up: bool = True,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.axis: int = self.default_axis
        self.squared_estimator: bool = self.default_squared_estimator
        self.window_len: int = self.default_window_len
        self.window_type: Callable = self.default_window_type

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                squared_estimator=squared_estimator,
                window_len=window_len,
                window_type=window_type,
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
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
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

        if squared_estimator is not None:
            self.squared_estimator = squared_estimator

        if window_len is not None:
            self.window_len = window_len

        if window_type is not None:
            self.window_type = window_type

        # Construct Parent #
        super().construct(*args, init_io=init_io, setup=setup, **kwargs)

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to create features from.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        data_ll = np.abs(np.diff(data, axis=self.axis))

        if self.squared_estimator:
            data_ll = data_ll ** 2

        if self.window_len > 0:
            data_ll = convolve(
                data_ll,
                self.window_type(self.window_len).reshape(-1, 1),
                mode='same')

            if self.squared_estimator:
                data_ll = np.sqrt(data_ll)

        return data_ll
