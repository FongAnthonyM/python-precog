""" filterbank.py

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
from itertools import chain
from typing import ClassVar, Any

# Third-Party Packages #
import numpy as np
from scipy.signal import filtfilt

# Local Packages #
from ..operation import BaseOperation
from .basefilterbuilder import Filter, BaseFilterBuilder


# Definitions #
# Classes #
class FilterBank(BaseOperation):

    default_filter_builders: list[BaseFilterBuilder, ...] = []
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("filter_data",)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        builders: list | None = None,
        filters: list | None = None,
        sample_rate: float | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.axis: int = 0

        self.sample_rate: float | None = None
        self.filter_builders: list[BaseFilterBuilder, ...] = self.default_filter_builders.copy()
        self.filters: list[Filter] = []

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                builders=builders,
                filters=filters,
                sample_rate=sample_rate,
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
        builders: list | None = None,
        filters: list | None = None,
        sample_rate: float | None = None,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            axis: The axis to remap along.
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        if builders is not None:
            self.filter_builders.clear()
            self.filter_builders.extend(builders)

        if filters is not None:
            self.filters.clear()
            self.filters.extend(filters)

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if axis is not None:
            self.axis = axis

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Create Filters
    def create_filters(self):
        self.filters.clear()
        self.filters.extend(chain.from_iterable(
            builder.create_filters_iter(sample_rate=self.sample_rate) for builder in self.filter_builders
        ))

    # Setup
    def setup(
        self,
        sample_rate: float | None = None,
        axis: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """A method for setting up the object before it runs operation."""
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if axis is not None:
            self.axis = axis

        self.create_filters()

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to remap.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        for filter_ in self.filters:
            data = filter_.filter(x=data, axis=self.axis)
        return data
