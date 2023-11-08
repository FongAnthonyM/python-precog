"""nnmflearner.py

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
from typing import Any

# Third-Party Packages #
import torch
from torch import Tensor
from torch.nn.functional import linear
from torch.nn import Parameter

# Local Packages #
from .basennmflearner import BaseNNMFLearner


# Definitions #
class NNMFLearner(BaseNNMFLearner):
    # Instance Methods  #
    def reconstruct(self, *args, **kwargs) -> Tensor:
        """Creates a reconstruction by taking the product of W and H."""
        return linear(self.H.tensor, self.W.tensor)

