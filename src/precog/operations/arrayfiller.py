""" arrayfiller.py

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
class ArrayFiller(BaseOperation):
    default_input_names: tuple[str, ...] = ("a", "b")
    default_output_names: tuple[str, ...] = ("a",)

    # Evaluate
    def evaluate(self, a: np.ndarray | None = None, b: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        a[:] = b[:]
        return a
