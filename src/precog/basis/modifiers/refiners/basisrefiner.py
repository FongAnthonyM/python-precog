""" basisrefiner.py.py

"""
# Package Header #
from ....header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from abc import abstractmethod
from typing import ClassVar, Any

# Third-Party Packages #

# Local Packages #
from ....operations import BaseOperation


# Definitions #
# Classes #
class BasisRefiner(BaseOperation):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("basis",)
    default_output_names: ClassVar[tuple[str, ...]] = ("m_basis",)

    # Evaluate
    @abstractmethod
    def evaluate(self, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """