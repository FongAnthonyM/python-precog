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
from typing import Any, Callable, Optional

# Third-Party Packages #

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module

# Local Packages #
from ..modelbasis import ModelBasis
from .basebasismodifier import BaseBasisModifier


# Definitions #
# Classes #
class AdaptiveMultiplicativeModifier(Optimizer, BaseBasisModifier):
    default_state_variables: dict[str, Any] = {
        "theta": None,
        "beta": None,
        "penalty": None,
        "pos": None,  # torch.zeros_like(theta, memory_format=torch.preserve_format)
        "neg": None,
        "step": 0,
    }

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        module: Module | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.module: Module | None = None

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(bases=bases, state_variables=state_variables, module=module, **kwargs)

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        module: Module | None = None,
        *args,
        **kwargs,
    ) -> None:
        if module is None:
            self.module = module

        # Construct Parent #
        super().construct(bases=bases, state_variables=state_variables, **kwargs)

    @torch.enable_grad()
    def closure(self):
        self.zero_grad()
        return self.module.forward()

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None,
        x: Tensor | None = None,
        theta: Tensor | None = None,
        beta: int | None = None,
        penalty: Tensor | None = None,
        pos: Tensor | None = None,
        neg: Tensor | None = None,
        step: int | None = None,
    ) -> None:
        """

        Args:
            closure:
            x:
            theta:
            beta:
            penalty:
            pos:
            neg:
            step:
        """
        if theta is None:
            theta = self.state_variables["theta"]
        else:
            self.state_variables["theta"] = theta

        if beta is None:
            beta = self.state_variables["beta"]
        else:
            self.state_variables["beta"] = beta
            
        if penalty is None:
            penalty = self.state_variables["penalty"]
        else:
            self.state_variables["penalty"] = penalty
            
        if pos is None:
            pos = self.state_variables["pos"]
        else:
            self.state_variables["pos"] = pos
            
        if neg is None:
            neg = self.state_variables["neg"]
        else:
            self.state_variables["neg"] = neg
            
        if step is None:
            step = self.state_variables["step"]

        # Make sure the closure is always called with grad enabled
        if closure is None:
            closure = self.closure
        else:
            closure = torch.enable_grad()(closure)

        # Cache the gradient status for reversion at the end of the function
        tensor = self.basis.tensor
        required_grad = tensor.requires_grad
        tensor.requires_grad = True

        ### FIRST PASS -- Accumulate Positive/Negative Gradient Contributions
        # Iterate over each parameter group (specifies order of optimization)

        # Iterate over models parameters within the group
        # if a gradient is not required then that parameter is "fixed"

        # Initialize temporary gradient components
        _neg = torch.zeros_like(tensor)
        _pos = torch.zeros_like(tensor)

        # Close the optimization loop by retrieving the
        # observed data and prediction
        WH = closure()

        # Multiplicative update coefficients for beta-divergence
        #      Marmin, A., Goulart, J.H.D.M. and Févotte, C., 2021.
        #      Joint Majorization-Minimization for Nonnegative Matrix
        #      Factorization with the $\beta $-divergence.
        #      arXiv preprint arXiv:2106.15214.
        if beta == 2:
            output_neg = x
            output_pos = WH
        elif beta == 1:
            output_neg = x / WH.add(self.precision)
            output_pos = torch.ones_like(WH)
        elif beta == 0:
            WH_eps = WH.add(self.precision)
            output_pos = WH_eps.reciprocal_()
            output_neg = output_pos.square().mul_(x)
        else:
            WH_eps = WH.add(self.precision)
            output_neg = WH_eps.pow(beta - 2).mul_(x)
            output_pos = WH_eps.pow_(beta - 1)

        # Numerator (negative factor) gradient
        # Retain graph so that backward can be run again using the
        # positive component.
        WH.backward(output_neg, retain_graph=True)
        __neg = (torch.clone(tensor.grad).relu_())
        tensor.grad.zero_()

        # Denominator (positive factor) gradient
        # The parameter gradient holds both components (positive - negative)
        WH.backward(output_pos)
        __pos = (torch.clone(tensor.grad).relu_())
        # p.grad.add_(-_neg)
        tensor.grad.zero_()

        # Include penalty
        __pos.add_(penalty)

        # Add to the running estimate
        _neg.add_(__neg)
        _pos.add_(__pos)

        # Initialize the state variables for gradient averaging.
        # Enables incremental learning.

        # Accumulate gradients
        neg.mul_(1 - theta).add_(_neg.mul_(theta))
        pos.mul_(1 - theta).add_(_pos.mul_(theta))

        # Avoid ill-conditioned, zero-valued multipliers
        neg.add_(self.precision)
        pos.add_(self.precision)

        # Multiplicative Update
        multiplier = neg.div(pos)
        tensor.mul_(multiplier)

        # Force the gradient requirement to off
        tensor.requires_grad = False

        # Reinstate the grad status from before
        tensor.requires_grad = required_grad
        
        # Update Step
        self.state_variables["step"] = step + 1 
        
        return None