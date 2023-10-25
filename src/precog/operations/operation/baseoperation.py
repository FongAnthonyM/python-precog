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
from typing import Any

# Third-Party Packages #
from baseobjects.functions import CallableMultiplexObject, MethodMultiplexer

# Local Packages #
from .io import IOManager


# Definitions #
# Classes #
class BaseOperation(CallableMultiplexObject):
    """An abstract class which defines an Operation, an easily definable data processing block with inputs and outputs.

    In subclasses the "evaluate" method must be defined as it is data processing element of this object. Additionally,
    the "input_names" and "output_names" must be defined to ensure the IO is mapped properly. "input_names" must match
    the keyword arguments of the "evaluate" method. "output_names" are names for each of the elements of the outputs
    tuple of the "evaluate" method.

    "evaluate" can be called directly which will run without using the Operation IO. This is useful for processing data
    without using the Object IO architecture.

    To use the Object IO architecture "execute" should be called. "execute" first gets the inputs from the inputs
    manager and passes it to "evaluate" method. After evaluating, the output from "evaluate" is then put into the
    outputs manager to be used later.

    "execute" is also MethodMultiplexer, meaning that its call is delegated to different specified method. This gives
    Operation the flexibility to change the "execute" method's implementation during runtime.

    Class Attributes:
        default_execute: The default name of the method to use for execution.
        default_input_names: The default ordered tuple with the names of the inputs to an Operation.
        default_output_names: The default ordered tuple with the names of the outputs to an Operation.

    Attributes:
        inputs: The inputs manager of the Operation.
        outputs: The outputs manager of the Operation.
        execute: The method multiplexer which manages which execute method to run when called.
        input_names: The ordered tuple with the names of the inputs to an Operation.
        _output_names: The ordered tuple with the names of the outputs to an Operation.

    Args:
        *args: Arguments for inheritance.
        init_io: Determines if construct_io run during this construction.
        sets_up: Determines if setup will run during this construction.
        setup_kwargs: The keyword arguments for the setup method.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """
    default_execute: str | None = None
    default_input_names: tuple[str, ...] = ()
    default_output_names: tuple[str, ...] = ()
    execute_output_names: set[str, ...] = {None, "execute_no_output", "execute_one_output", "execute_multiple_outputs"}

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.inputs: IOManager = IOManager()
        self.outputs: IOManager = IOManager()

        self.execute: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_execute)

        self.input_names = self.default_input_names

        self._output_names: tuple[str, ...] = ()
        self.output_names = self.default_output_names

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(*args, init_io=init_io, steps_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """The order tuple of the names of the outputs"""
        return self._output_names

    @output_names.setter
    def output_names(self, value: tuple[str, ...]) -> None:
        if self.execute.selected in self.execute_output_names:
            if len(value) == 1:
                self.execute.select("execute_one_output")
            elif value:
                self.execute.select("execute_multiple_outputs")
            else:
                self.execute.select("execute_no_outputs")
        self._output_names = value

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
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
        # Construct Parent #
        super().construct(*args, **kwargs)

        if init_io:
            self.construct_io()

        if sets_up:
            self.setup(**setup_kwargs)

    # IO
    def construct_io(self, *args, **kwargs) -> None:
        """Constructs the io for this object.

        Args:
            *args: The arguments for constructing the io.
            **kwargs: The keyword arguments for constructing the io.
        """
        self.inputs.create_io(self.input_names, *args, **kwargs)
        self.outputs.create_io(self.output_names, *args, **kwargs)

    def output_as_dict(self, output: tuple[Any, ...]) -> dict[str, Any]:
        """Parses the output tuple from the evaluate method and returns a dictionary of the outputs.

        Args:
            output: The output of the evaluate method.

        Returns:
            The dictionary of outputs.
        """
        return dict(zip(self.output_names, output))

    def output_as_tuple(self, output: dict[str, Any]) -> tuple[Any, ...]:
        """Parses the output dictionary from the outputs and returns a tuple of the outputs.

        Args:
            output: The output from the outputs object.

        Returns:
            The tuple of outputs.
        """
        return tuple(output.get(name) for name in self.output_names)

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

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
        pass

    # Execute
    def execute_no_output(self) -> None:
        """Evaluates from the inputs and does not output."""
        self.evaluate(**self.inputs.get_all())

    def execute_one_output(self) -> None:
        """Evaluates from the inputs and puts the single result to the outputs."""
        self.outputs.put_all(self.output_as_dict((self.evaluate(**self.inputs.get_all()),)))
        
    def execute_multiple_outputs(self) -> None:
        """Evaluates from the inputs and puts the multiple results to the outputs."""
        self.outputs.put_all(self.output_as_dict(self.evaluate(**self.inputs.get_all())))

    def execute_dict_output(self) -> None:
        """Evaluates from the inputs and puts the output dict directly to the outputs."""
        self.outputs.put_all(self.evaluate(**self.inputs.get_all()))
