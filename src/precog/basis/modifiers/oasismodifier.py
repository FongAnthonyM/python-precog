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
from typing import Any

# Third-Party Packages #
import torch
from oasis.functions import GetSn, estimate_time_constant, deconvolve

# Local Packages #
from ..modelbasis import ModelBasis
from .basebasismodifier import BaseBasisModifier


# Definitions #
# Classes #
class OASISModifier(BaseBasisModifier):
    default_state_variables: dict[str, Any] = {
        "factor_state_variables": None,
    }
    default_factor_state_variables = {
        "g": None,
        "baseline": None,
        "sn": None,
        "penalty": 0,  # int [0, 1]
        "nonneg": True,  # bool
        "optimize_g": True,
        "optimize_g_len": None,  # int
    }

    @ classmethod
    def create_state_variables(cls, factor_state_variables, **kwargs):
        factor_state_variables = [cls.default_factor_state_variables | k for k in factor_state_variables]
        super().create_state_variables(factor_state_variables=factor_state_variables, **kwargs)

    # Magic Methods #
    @property
    def H(self) -> ModelBasis:
        return self.bases["H"]

    @property
    def S(self) -> ModelBasis:
        return self.bases["S"]

    # Instance Methods  #
    def step(self) -> None:
        shape = self.H.tensor.shape
        H = self.H.tensor.detach().numpy()
        for f in range(shape[self.H.factor_axis]):
            state_variables = self.state_variables["factor_state_variables"][f]
            h = H[0, f, :]

            # Deconvolve Model
            c, s, b, g, sn = deconvolve(
                h,
                g=state_variables["g"],
                penalty=state_variables["penalty"],
                b=state_variables["baseline"],
                b_nonneg=state_variables["nonneg"],
                optimize_g=shape[-1] if state_variables["optimize_g"] else 0,
            )  # 0 <= opt_g <= H.shape[-1]

            state_variables["g"] = g
            state_variables["sn"] = sn
            state_variables["baseline"] = b

            self.H.tensor[0, f, :] = torch.as_tensor(c.clip(min=0) + self.precision)
            self.S.tensor[0, f, :] = torch.as_tensor(s.clip(min=0) + self.precision)
