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
from baseobjects import BaseObject

# Local Packages #
from ...basis import ModelBasis


# Definitions #
# Classes #
class BaseModel(BaseObject):
    # Class Attributes #
    default_bases: ClassVar[dict[str, tuple[type, dict[str, Any]]]] = {}
    default_architecture: ClassVar[tuple[type, dict[str, Any]]] = ()
    default_trainer: ClassVar[tuple[type, dict[str]]] = ()
    default_submodels: ClassVar[dict[str, tuple[type, dict[str, Any]]]] = {}

    # Attributes #
    architecture_bases: set[str] = set()
    trainer_bases: set[str] = set()

    _bases: dict[str, ModelBasis]
    _atoms: dict[str, Any]
    architecture: Any = None
    trainer: Any = None

    submodels: dict[str, "BaseModel"]

    # Properties
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
        architecture: Any = None,
        trainer: Any = None,
        submodels: dict[str, "BaseModel"] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs,
    ) -> None:
        # Attributes #
        self.architecture_bases = self.architecture_bases.copy()
        self.trainer_bases = self.trainer_bases.copy()

        self._bases = {}
        self._atoms = {}

        self.submodels = {}

        # Parent Attributes #
        super().__init__(init=False)

        # Construct #
        if init:
            self.construct(bases=bases, architecture=architecture, trainer=trainer, submodels=submodels, **kwargs)

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis | dict[str, Any]] | None = None,
        architecture: Any = None,
        trainer: Any = None,
        submodels: dict[str, "BaseModel"] | None = None,
        init: bool = True,
        **kwargs,
    ) -> None:
        super().construct(**kwargs)

        self.construct_default_bases()
        if bases is not None:
            self.bases.update(bases)

        if architecture is not None:
            self.architecture = architecture
        else:
            self.construct_default_architecture()

        if self.architecture is not None:
            self.set_architecture_bases()

        if trainer is not None:
            self.trainer = trainer
        else:
            self.construct_default_trainer()

        if self.trainer is not None:
            self.set_trainer_bases()

        self.construct_default_submodels()
        if submodels is not None:
            self.submodels.update(submodels)

    # Bases
    def construct_default_bases(self) -> None:
        self._bases.update({n: t(**k) for n, (t, k) in self.default_bases.items()})

    def get_submodel_bases(self) -> dict[str, Any]:
        return {name: submodel.bases for name, submodel in self.submodels.items()}

    def get_bases(self) -> dict[str, Any]:
        return self._bases | self.get_submodel_bases()

    # Atoms
    def get_submodels_atoms(self) -> dict[str, Any]:
        return {name: submodel for name, submodel in self.submodels.items()}

    def get_atoms(self) -> dict[str, Any]:
        return self._atoms | self.get_submodels_atoms()
    
    # Architecture
    def construct_default_architecture(self) -> None:
        self.architecture = self.default_architecture[0](**self.default_architecture[1])

    def set_architecture_bases(self) -> None:
        for name in self.architecture_bases:
            self.architecture.set_basis(name, self._bases[name])

    # Trainer
    def construct_default_trainer(self) -> None:
        self.trainer = self.default_trainer[0](**self.default_trainer[1])

    def set_trainer_bases(self) -> None:
        for name in self.trainer_bases:
            self.trainer.set_basis(name, self._bases[name])

    # Submodels
    def construct_default_submodels(self) -> None:
        self.submodels.update({n: t(**k) for n, (t, k) in self.default_submodels.items()})
