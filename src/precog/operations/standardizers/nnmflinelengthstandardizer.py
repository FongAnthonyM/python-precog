""" baseoperation.py
An abstract class which defines an Operation, an easily definable data processing block with inputs and outputs.
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
from typing import Any, Callable

# Third-Party Packages #

# Local Packages #
from ..operation import OperationGroup
from ..features import LineLength
from ..shiftrescalers import RunningShiftScaler, blank_arg
from ..constraints import NonNegative


# Definitions #
# Classes #
class NNMFLineLengthStandardizer(OperationGroup):
    default_input_names: tuple[str, ...] = ("data",)
    default_output_names: tuple[str, ...] = ("features",)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        shift_scale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                squared_estimator=squared_estimator,
                window_len=window_len,
                window_type=window_type,
                shift_scale=shift_scale,
                forget_factor=forget_factor,
                non_negative=non_negative,
                non_negative_kwargs=non_negative_kwargs,
                axis=axis,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        shift_scale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
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
        new_setup_kwargs = dict(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            shift_scale=shift_scale,
            forget_factor=forget_factor,
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
            axis=axis,
        )

        setup_kwargs = ({} if setup_kwargs is None else setup_kwargs) | new_setup_kwargs

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Setup
    def setup(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        shift_scale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        axis: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """A method for setting up the object before it runs operation."""
        # Create Operations
        self.operations["line_length"] = line_length = LineLength(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            axis=axis,
        )

        self.operations["shift_scale"] = shift_scaler = RunningShiftScaler(
            shift_rescale=shift_scale,
            forget_factor=forget_factor,
            axis=axis,
        )

        self.operations["non_negative"] = non_negative_op = NonNegative(
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
        )

        # Set Input
        self.inputs["data"] = line_length.inputs["data"]

        # Inner IO
        shift_scaler.inputs["data"] = line_length.outputs["features"]
        non_negative_op.inputs["data"] = shift_scaler.outputs["ss_data"]

        # Set Output
        self.outputs["features"] = non_negative_op.outputs["nn_data"]
