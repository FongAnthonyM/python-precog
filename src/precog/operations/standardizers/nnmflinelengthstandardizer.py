""" nnmflinelengthstandardizer.py
An abstract class which defines an Operation, an easily definable data processing block with inputs and outputs.
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
        setup_kwargs = {} if setup_kwargs is None else setup_kwargs
        setup_kwargs["create_kwargs"] = setup_kwargs.get("create_kwargs", {}) | dict(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            shift_scale=shift_scale,
            forget_factor=forget_factor,
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
            axis=axis,
        )

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Operations
    def create_operations(
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
        # Create Operations
        self.operations["line_length"] = LineLength(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            axis=axis,
        )

        self.operations["shift_scale"] = RunningShiftScaler(
            shift_rescale=shift_scale,
            forget_factor=forget_factor,
            axis=axis,
        )

        self.operations["non_negative"] = NonNegative(
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
        )

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Operations
        line_length = self.operations["line_length"]
        shift_scaler = self.operations["shift_scale"]
        non_negative_op = self.operations["non_negative"]

        # Set Input
        self.inputs["data"] = line_length.inputs["data"]

        # Inner IO
        shift_scaler.inputs["data"] = line_length.outputs["features"]
        non_negative_op.inputs["data"] = shift_scaler.outputs["ss_data"]

        # Set Output
        self.outputs["features"] = non_negative_op.outputs["nn_data"]

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
        create: bool = True,
        create_kwargs: dict[str, Any] | None = None,
        link: bool = True,
        link_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """A method for setting up the object before it runs operation."""
        new_create_kwargs = dict(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            shift_scale=shift_scale,
            forget_factor=forget_factor,
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
            axis=axis,
        )

        create_kwargs = new_create_kwargs | ({} if create_kwargs is None else create_kwargs)

        super().setup(create=create, create_kwargs=create_kwargs, link=link, link_kwargs=link_kwargs, **kwargs)
