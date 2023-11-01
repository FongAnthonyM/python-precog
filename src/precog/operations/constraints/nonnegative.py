""" nonnegative.py

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
from baseobjects.functions import CallableMultiplexObject, MethodMultiplexer
import numpy as np

# Local Packages #
from ..operation import BaseOperation


# Definitions #
# Classes #
class NonNegative(BaseOperation):
    default_input_names: tuple[str, ...] = ("data",)
    default_output_names: tuple[str, ...] = ("nn_data",)
    default_non_negative: str = "clip"

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.non_negative: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_non_negative)
        self.non_negative_kwargs: dict = {}

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                non_negative=non_negative,
                non_negative_kwargs=non_negative_kwargs,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        *args: str | None,
        init_io: bool = True,
        sets_up: Any = True,
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
        if non_negative is not None:
            self.non_negative.select(non_negative)

        if non_negative_kwargs is not None:
            self.non_negative_kwargs.clear()
            self.non_negative_kwargs.update(non_negative_kwargs)

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Non-Negative
    def clip(self, data: np.ndarray, threshold: float = 0, **kwargs: Any) -> np.ndarray:
        return data.clip(min=threshold)  # threshold >= 0

    def abs(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.abs(data)

    def square(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        return data**2

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None , *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        return self.non_negative(data, **self.non_negative_kwargs)
