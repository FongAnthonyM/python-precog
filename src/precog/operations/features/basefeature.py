""" basefeature.py

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
import numpy as np

# Local Packages #
from ..operation import BaseOperation


# Definitions #
# Classes #
class BaseFeature(BaseOperation):
    default_input_names: tuple[str, ...] = ("data",)
    default_output_names: tuple[str, ...] = ("features",)

    # Instance Methods #
    # Evaluate
    @abstractmethod
    def evaluate(self, data: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to create features from.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        pass

