"""compositemodel.py

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
from copy import deepcopy
from typing import Any

# Third-Party Packages #

# Local Packages #
from ..modelbasis import ModelBasis
from .basemodel import BaseModel


# Definitions #
# Classes #
class CompositeModel(BaseModel):
    default_bases: tuple[tuple[str, type, dict[str, Any]], ...] = ()
    default_models: tuple[tuple[str, type, dict[str, Any]], ...] = ()
    default_basis_map: dict[str, dict[str, tuple[str, ...] | str]] = {}

    # Magic Methods  #
    # Construction/Destruction
    def __init__(
        self,
        bases: dict[str, Any] | None = None,
        models: dict[str, BaseModel] | None = None,
        *args: Any,
        basis_map: dict[str, dict[str, tuple[str, ...] | str]] | None = None,
        map_bases: bool = False,
        init=True,
        **kwargs,
    ) -> None:
        # New Attributes #
        self.basis_map: dict[str, dict[str, tuple[str, ...] | str]] = deepcopy(self.default_basis_map)

        self.atoms: dict[str, Any] = {}
        self.bases: dict[str, ModelBasis] = {}
        self.models: dict[str, BaseModel] = {}

        # Parent Attributes #
        super().__init__(init=False)

        # Construct #
        if init:
            self.construct(
                bases=bases,
                models=models,
                basis_map=basis_map,
                map_bases=map_bases,
                **kwargs,
            )

    def construct(
        self,
        bases: dict[str, Any] | None = None,
        models: dict[str, BaseModel] | None = None,
        *args: Any,
        basis_map: dict[str, dict[str, tuple[str, ...] | str]] | None = None,
        map_bases: bool = False,
        **kwargs: Any,
    ) -> None:
        if basis_map is not None:
            self.basis_map.clear()
            self.basis_map.update(basis_map)

        super().construct(bases=bases, **kwargs)

        self.create_default_models()
        if models is not None:
            self.models.update(bases)

        if map_bases:
            self.map_bases()

    def create_default_models(self):
        self.bases.update({n: t(**k) for n, t, k in self.default_models})

    def map_bases(self, basis_map: dict[str, dict[str, tuple[str, ...] | str]] | None = None) -> None:
        if basis_map is not None:
            self.basis_map.clear()
            self.basis_map.update(basis_map)

        for basis_name, models in self.basis_map.items():
            for model_name, model_basis_names in models.items():
                if isinstance(model_basis_names, str):
                    self.models[model_name].bases[model_basis_names] = self.bases[basis_name]
                else:
                    for model_basis_name in model_basis_names:
                        self.models[model_name].bases[model_basis_name] = self.bases[basis_name]
