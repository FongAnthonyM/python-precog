"""basebasismodifier.py

"""
# Package Header #
from ...header import *

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
import numpy as np

# Local Packages #
from ..bases import ModelBasis
from ..basiscontainer import BasisContainer


# Definitions #
class BaseBasisModifier(BasisContainer):
    # Class Attributes #
    precision: ClassVar[float] = np.finfo(np.float64).precision

    # Instance Methods  #
    @abstractmethod
    def modify(self, *args: Any, **kwargs: Any) -> Any:
        """Modifies a given the basis with the given parameters.

        Args:
            *args: The arguments for modify the basis.
            **kwargs: The keyword arguments for modify the basis.

        Returns:
            Any: The updated object.
        """

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        """Updates the contained basis with the given parameters.

        Args:
            *args: The arguments for updating the basis.
            **kwargs: The keyword arguments for updating the basis.

        Returns:
            Any: The updated object.
        """
