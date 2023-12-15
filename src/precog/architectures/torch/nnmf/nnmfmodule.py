"""nnmfmodule.py

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
from typing import Any

# Third-Party Packages #
from torch import Tensor
from torch.nn.functional import linear

# Local Packages #
from .basennmfmodule import BaseNNMFModule


# Definitions #
class NNMFModule(BaseNNMFModule):
    # Instance Methods  #
    def reconstruct(self, *args, **kwargs) -> Tensor:
        """Creates a reconstruction by taking the product of W and H."""
        return linear(self.H, self.W)
