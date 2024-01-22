"""nnmfspiketrainer.py

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
from collections.abc import Mapping
from typing import ClassVar, Any

# Third-Party Packages #

# Local Packages #
from ..operations import BaseOperation, OperationGroup
from ..operations.operation.io import IOManager, IORouter
from ..architectures.torch import BaseNNMFModule, NNMFDModule
from ..basis import ModelBasis
from ..basis.modifiers import AdaptiveMultiplicativeModifier
from ..trainers.bases import BaseTrainerOperation


# Definitions #
# Classes #
class EnsembleTrainer(OperationGroup, BaseTrainerOperation):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names:  ClassVar[tuple[str, ...]] = ("bases",)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        subtrainers: dict[str, BaseTrainerOperation] | None = None,
        *args: Any,
        operations: Mapping[str, BaseOperation] | None = None,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                subtrainers=subtrainers,
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                operations=operations,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        subtrainers: dict[str, BaseTrainerOperation] | None = None,
        *args: Any,
        operations: Mapping[str, BaseOperation] | None = None,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        **kwargs: Any,
    ) -> None:
        # New Setup #
        if subtrainers is not None:
            self.subtrainers.update(subtrainers)
            operations.update(subtrainers)

        # Construct Parent #
        super().construct(
            bases=bases,
            state_variables=state_variables,
            create_defaults=create_defaults,
            bases_kwargs=bases_kwargs,
            operations=operations,
            init_io=init_io,
            sets_up=sets_up,
            setup_kwargs=setup_kwargs,
            **kwargs,
        )

    # State Variables
    def get_state_variables(self) -> dict[str, Any]:
        state_vars = super().get_state_variables()
        state_vars["subtrainers"] = {n: s.state_variables for n, s in self.subtrainers.items()}
        return state_vars

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Set Input
        self.inputs["data"] = IORouter(inputs={n: op.inputs["data"] for n, op in self.operations})

        # Set Output
        W_refiner.outputs["r_bases"] = self.outputs["bases"]


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
        if create_kwargs is None:
            create_kwargs = {}

        if "W_modifier" not in self.operations:
            create_kwargs["W_modifier_kwargs"] = self.create_W_modifier_kwargs(
                **(create_kwargs.get("W_modifier_kwargs", {})),
            )
            
        if "H_modifier" not in self.operations:
            create_kwargs["H_modifier_kwargs"] = self.create_H_modifier_kwargs(
                **(create_kwargs.get("H_modifier_kwargs", {})),
            )
        
        super().setup(
            *args, 
            create=create,
            create_kwargs=create_kwargs,
            link=link,
            link_kwargs=link_kwargs,
            **kwargs,
        )
