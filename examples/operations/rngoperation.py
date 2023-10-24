""" rngoperation.py
An Operation which generates an array of random numbers.
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
import numpy as np

# Local Packages #
from precog.operations import BaseOperation


# Definitions #
# Classes #
class RNGOperation(BaseOperation):
    """An Operation which generates an array of random numbers.

    Attributes:
        inputs: The inputs manager of the Operation.
        outputs: The outputs manager of the Operation.
        execute: The method multiplexer which manages which execute method to run when called.
        input_names: The ordered tuple with the names of the inputs to an Operation.
        _output_names: The ordered tuple with the names of the outputs to an Operation.

        shape: The shape of the random array to generate.
        scale: The scale to apply to the random array.
        shift: The shift to apply to the random array.

    Args:
        shape: The shape of the random array to generate.
        scale: The scale to apply to the random array.
        shift: The shift to apply to the random array.
        *args: Arguments for inheritance.
        init_io: Determines if construct_io run during this construction.
        setup: Determines if setup will run during this construction.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """
    default_output_names: tuple[str, ...] = ("out_array",)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        shape: tuple[int, ...] = (),
        scale: float = 1,
        shift: float = 0.0,
        *args: Any,
        init_io: bool = True,
        steps_up: bool = True,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.shape: tuple[int, ...] = shape
        self.scale: float = scale
        self.shift: float = shift

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                init_io=init_io,
                steps_up=steps_up,
                **kwargs,
            )

    # Evaluate
    def evaluate(self, *args, **kwargs: Any) -> Any:
        """Generates an array of random numbers.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            A randomly generated array.
        """
        return np.random.rand(*self.shape) * self.scale - self.shift
