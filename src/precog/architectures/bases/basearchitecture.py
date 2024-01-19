""" basearchitecture.py.py

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

# Third-Party Packages #

# Local Packages #
from ...basis import BasisContainer


# Definitions #
# Classes #
class BaseArchitecture(BasisContainer):
    """An abstract bases class for model architectures."""
    # Attributes #
    subarchitectures: dict[str, "BaseArchitecture"]
