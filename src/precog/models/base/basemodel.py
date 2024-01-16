"""basemodel.py

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
from typing import ClassVar, Any

# Third-Party Packages #
from baseobjects import BaseObject, BaseComponent

# Local Packages #
from ...basis import ModelBasis, BasisContainer
from ...architectures import BaseArchitecture
from ...trainers import BaseTrainer


# Definitions #
# Classes #
class BaseModel(BaseObject):
    # Class Attributes #
    _part_types: ClassVar[tuple[str]] = ("local", "architecture", "trainer", "submodels")
    default_state_variables: ClassVar[dict[str, Any]] = {}
    default_bases: ClassVar[dict[str, dict[str, dict | tuple[type, dict[str, Any]]]]] = {}
    default_architecture: ClassVar[tuple[type[BaseArchitecture], dict[str, Any]]] = ()
    default_trainer: ClassVar[tuple[type[BaseTrainer], dict[str]]] = ()
    default_submodels: ClassVar[dict[str, tuple[type["BaseModel"], dict[str, Any]]]] = {}

    # Attributes #
    submodel_type: type["BaseModel"]
    local_state_variables: dict[str, Any]
    local_bases: dict[str, ModelBasis]
    _atoms: dict[str, Any]
    architecture: BaseArchitecture | None = None
    trainer: BaseTrainer | None = None

    submodels: dict[str, "BaseModel"]

    # Properties
    @property
    def state_variables(self) -> dict[str, Any]:
        return self.get_state_variables()

    @property
    def bases(self) -> dict[str, Any]:
        return self.get_bases()

    @property
    def atoms(self) -> dict[str, Any]:
        return self.get_atoms()

    # Magic Methods  #
    # Construction/Destruction
    def __init__(
        self,
        bases: dict[str, ModelBasis | dict[str, Any]] | None = None,
        state_variables: dict[str, Any] | None = None,
        architecture: Any = None,
        trainer: Any = None,
        submodels: dict[str, "BaseModel"] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs,
    ) -> None:
        # Attributes #
        self.local_state_variables = {}
        self.local_bases = {}
        self._atoms = {}

        self.submodels = {}

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                bases=bases,
                state_variables=state_variables,
                architecture=architecture,
                trainer=trainer,
                submodels=submodels,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis | dict[str, Any]] | None = None,
        state_variables: dict[str, Any] | None = None,
        architecture: Any = None,
        trainer: Any = None,
        submodels: dict[str, "BaseModel"] | None = None,
        **kwargs,
    ) -> None:
        super().construct(**kwargs)

        # State Variables
        if state_variables is None:
            state_variables = self.default_state_variables
        else:
            state_variables = {
                n: self.default_state_variables.get(n, {}) | state_variables.get(n, {}) for n in self._part_types
            }
        
        if (local_state_variables := state_variables.get("local", None)) is not None:
            self.local_state_variables.update(local_state_variables)
        
        # Bases
        d_bases = self.construct_default_bases()
        bases = d_bases if bases is None else {n: d_bases.get(n, {}) | bases.get(n, {}) for n in self._part_types}

        if (local_bases := bases.get("local", None)) is not None:
            self.local_bases.update(local_bases)
        
        # Contained Objects
        if architecture is not None:
            self.architecture = architecture
        else:
            self.construct_default_architecture(
                bases=bases.get("architecture", None), 
                state_variables=state_variables.get("architecture", None),
            )

        if trainer is not None:
            self.trainer = trainer
        else:
            self.construct_default_trainer(
                bases=bases.get("trainer", None), 
                state_variables=state_variables.get("trainer", None),
            )

        self.construct_default_submodels()
        if submodels is not None:
            self.submodels.update(submodels)

    # State Variables
    def get_submodel_state_variables(self) -> dict[str, Any]:
        return {name: submodel.state_variables for name, submodel in self.submodels.items()}

    def get_state_variables(self) -> dict[str, Any]:
        return {
            "local": self.local_state_variables,
            "architecture": self.architecture.state_variables,
            "trainer": self.trainer.state_variables,
            "submodels": self.get_submodel_state_variables(),
        }

    # Bases
    def create_bases(self, bases: dict[str, dict] | tuple[type, dict[str, Any]]) -> dict[str, dict | ModelBasis]:
        if isinstance(bases, dict):
            return {n: self.create_bases(b) for n, b in bases.items()}
        else:
            return bases[0](**bases[1])

    def construct_default_bases(self) -> dict[str, dict]:
        return self.create_bases(self.default_bases)

    def get_submodel_bases(self) -> dict[str, Any]:
        return {name: submodel.bases for name, submodel in self.submodels.items()}

    def get_bases(self) -> dict[str, Any]:
        return {
            "local": self.local_bases,
            "architecture": self.architecture.bases,
            "trainer": self.trainer.bases,
            "submodels": self.get_submodel_bases(),
        }

    # Atoms
    def get_submodels_atoms(self) -> dict[str, Any]:
        return {name: submodel for name, submodel in self.submodels.items()}

    def get_atoms(self) -> dict[str, Any]:
        return self._atoms | self.get_submodels_atoms()
    
    # Architecture
    def construct_default_architecture(self, **kwargs) -> None:
        self.architecture = self.default_architecture[0](**(self.default_architecture[1] | kwargs))

    def build_architecture(self, **kwargs) -> None:
        pass

    # Trainer
    def construct_default_trainer(self, **kwargs) -> None:
        self.trainer = self.default_trainer[0](**(self.default_trainer[1] | kwargs))

    def build_trainer(self, **kwargs) -> None:
        pass

    # Submodels
    def construct_default_submodels(self) -> None:
        self.submodels.update({n: t(**k) for n, (t, k) in self.default_submodels.items()})

    def construct_subomodels_from_architecture(self) -> None:
        self.submodels.update(
            {n: self.submodel_type(architecture=a) for n, a in self.architecture.subarchitectures.items()},
        )

    def construct_subomodels_from_trainer(self) -> None:
        self.submodels.update({n: self.submodel_type(trainer=t) for n, t in self.trainer.subtrainers.items()})
