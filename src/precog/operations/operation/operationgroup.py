""" exampleoperationgroup.py
An Operation which contains several Operation objects to execute.
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
from collections.abc import Mapping
from typing import Any

# Third-Party Packages #
from baseobjects.collections import OrderableDict

# Local Packages #
from .baseoperation import BaseOperation


# Definitions #
# Classes #
class OperationGroup(BaseOperation):
    """An Operation which contains several Operation objects to execute.

    OperationGroup is more of an abstract class because IO and contained Operations must be defined, but an
    OperationGroup object can still function properly without defining those.

    The IO and/or contained Operations can be defined in either the "construction_io" or the "setup" methods. Operations
    could also be defined outside the OperationGroup class and be added into the OrderableDict "operation" directly.

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
        operations: The ordered dictionary of Operation to execute and the order to execute them in.

    Args:
        operations: The dictionary of Operation to add to the OperationGroup.
        *args: Arguments for inheritance.
        init_io: Determines if construct_io run during this construction.
        sets_up: Determines if setup will run during this construction.
        setup_kwargs: The keyword arguments for the setup method.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """
    default_execute: str | None = "execute_all"

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        operations: Mapping[str, BaseOperation] | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
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
                *args,
                operations=operations,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        operations: Mapping[str, BaseOperation] | None = None,
        *args: Mapping[str, BaseOperation] | None,
        init_io: Any = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            operations: The dictionary of Operation to add to the OperationGroup.
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        if operations is not None:
            self.operations.update(operations)

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Operations
    def create_operations(self, *args: Any, **kwargs) -> None:
        """Creates the inner operations.

        Args:
            *args: The arguments for creating the inner operations.
            **kwargs: The keyword arguments for creating the inner operations.
        """

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        """Links the inner operations' IO.

        Args:
            *args: The arguments for creating linking the inner operations' IO.
            **kwargs: The keyword arguments for creating linking the inner operations' IO.
        """

    # Setup
    def setup(
        self,
        *args: Any,
        create: bool = True,
        create_kwargs: dict[str, Any] | None = None,
        link: bool = True,
        link_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates the inner operations and links their IO.

        Args:
            *args: The arguments for setup.
            create: Determines if the inner operation will be created.
            create_kwargs: The keyword arguments for creating the inner operations.
            link: Determines if the inner IO will be linked between operations.
            link_kwargs: The keyword arguments for creating linking the inner operations' IO.
            **kwargs: The keyword arguments for setup.
        """
        if create:
            self.create_operations(**(create_kwargs if create_kwargs is not None else {}))

        if link:
            self.link_inner_io(**(link_kwargs if link_kwargs is not None else {}))

    # Evaluate
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        self.inputs.put_all(**kwargs)
        self.execute()
        outputs = self.outputs.get_items(self.output_names)
        return outputs if len(outputs) > 1 else outputs[0]

    # Execute
    def execute_all(self) -> None:
        """Executes all operation within this operation group."""
        for operation in self.operations.values():
            operation.execute()
