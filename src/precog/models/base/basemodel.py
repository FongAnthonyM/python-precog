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
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject

# Local Packages #
from ...basis import ModelBasis


# Definitions #
# Classes #
class BaseModel(BaseObject):
    default_bases: tuple[tuple[str, type, dict[str, Any]], ...] = ()

    # Magic Methods  #
    # Construction/Destruction
    def __init__(self, bases: dict[str, Any] | None = None, *, init=True, **kwargs) -> None:
        # New Attributes #
        self.atoms: dict[str, Any] = {}
        self.bases: dict[str, ModelBasis] = {}

        # Parent Attributes #
        super().__init__(init=False)

        # Construct #
        if init:
            self.construct(bases=bases, **kwargs)

    # Instance Methods  #
    # Constructors/Destructors
    def construct(self, bases: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> None:
        super().construct(**kwargs)

        self.create_default_bases()
        if bases is not None:
            self.bases.update(bases)

    def create_default_bases(self):
        self.bases.update({n: t(**k) for n, t, k in self.default_bases})
