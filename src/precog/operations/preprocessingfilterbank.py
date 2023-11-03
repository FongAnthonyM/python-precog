""" preprocessingfilterbank.py

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


# Third-Party Packages #


# Local Packages #
from .filters import BaseFilterBuilder, FilterBank, ButterworthFilterBuilder, NotchFilterBuilder


# Definitions #
# Classes #
class PreprocessingFilterBank(FilterBank):
    default_filter_builders: list[BaseFilterBuilder, ...] = [
        NotchFilterBuilder(),
        ButterworthFilterBuilder(pass_frequency=512.0, stop_frequency=512.0 * 1.1, butter_type="low"),
        ButterworthFilterBuilder(pass_frequency=1.0, stop_frequency=1.0 * 0.9, butter_type="high"),
    ]

