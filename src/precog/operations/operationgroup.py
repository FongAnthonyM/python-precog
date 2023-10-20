""" baseoperation.py

"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from abc import abstractmethod
from typing import Any
from warnings import warn

# Third-Party Packages #
from baseobjects.collections import OrderableDict

# Local Packages #
from .baseoperation import BaseOperation


# Definitions #
# Classes #
class OperationGroup(BaseOperation):
    default_execute: str | None = None
    default_input_names: tuple[str, ...] = ()
    default_output_names: tuple[str, ...] = ()

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        *args: Any,
        init_io: bool = True,
        steps_up: bool = True,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.operations: OrderableDict[str, BaseOperation] = OrderableDict()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                init_io=init_io,
                steps_up=steps_up,
            )

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @output_names.setter
    def output_names(self, value: tuple[str, ...]) -> None:
        if self.execute.selected in {None, "execute_one_output", "execute_multiple_outputs"}:
            if len(value) == 1:
                self.execute.select("execute_one_output")
            else:
                self.execute.select("execute_multiple_outputs")
        self._output_names = value

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        *args: Any,
        init_io: bool = True,
        setup: bool = True,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            setup: Determines if setup will run during this construction
            **kwargs: Keyword arguments for inheritance.
        """
        # Construct Parent #
        super().construct(*args, **kwargs)

        if init_io:
            self.construct_io()

        if setup:
            self.setup()

    # IO
    def construct_io(self, *args, **kwargs) -> None:
        """Constructs the io for this object.

        Args:
            *args: The arguments for constructing the io.
            **kwargs: The keyword arguments for constructing the io.
        """
        self.inputs.create_io(self.input_names, *args, **kwargs)
        self.outputs.create_io(self.output_names, *args, **kwargs)

    def parse_output(self, *args) -> dict[str, Any]:
        """Parses the output from the evaluate method and returns a dictionary of the outputs.

        Args:
            *args: The order tuple of the outputs to load into a dictionary.

        Returns:
            The dictionary of outputs.
        """
        return dict(zip(self.output_names, args))

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operations."""
        pass

    # Evaluate
    def evaluate(self, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        for operation in self.operations:
            operation.execute()

    # Execute
    def execute_one_output(self) -> None:
        """Evaluates from the inputs and puts the single result to the outputs."""
        self.outputs.put_all(self.parse_output(self.evaluate(**self.inputs.get_all())))
        
    def execute_multiple_outputs(self) -> None:
        """Evaluates from the inputs and puts the multiple results to the outputs."""
        self.outputs.put_all(self.parse_output(*self.evaluate(**self.inputs.get_all())))
