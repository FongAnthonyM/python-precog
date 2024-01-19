""" basiscontainer.py.py

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
from collections import ChainMap
from typing import ClassVar, Any

# Third-Party Packages #
from baseobjects import BaseObject

# Local Packages #
from .bases import ModelBasis


# Definitions #
# Classes #
class BasisContainer(BaseObject):
    # Class Attributes #
    default_state_variables: ClassVar[dict[str, Any]] = {}
    default_bases: ClassVar[dict[str, tuple[type[ModelBasis], dict[str, Any]]]] = {}

    # Class Methods #
    @classmethod
    def create_state_variables(cls, **kwargs) -> dict[str, Any]:
        return cls.default_state_variables | kwargs

    # Attributes #
    state_variables: dict[str, Any]
    bases: dict[str, ModelBasis]
    all_bases: ChainMap[str, ModelBasis]

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Attributes #
        self.state_variables = self.create_state_variables()
        self.bases = {}
        self.all_bases = ChainMap(self.bases)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # Assign New #
        if state_variables is not None:
            self.state_variables.update(state_variables)

        # Construct Parent #
        super().construct(*args, **kwargs)

        # Construct New #
        if create_defaults:
            self.construct_default_bases(bases_kwargs=bases_kwargs)
        if bases is not None:
            self.bases.update(bases)

    # Bases
    def construct_default_bases(self, bases_kwargs: dict[str, dict[str, Any]] | None = None) -> None:
        if bases_kwargs is None:
            self.bases.update({n: t(**k) for n, (t, k) in self.default_bases.items()})
        else:
            self.bases.update({n: t(**(k | bases_kwargs.get(n, {}))) for n, (t, k) in self.default_bases.items()})
