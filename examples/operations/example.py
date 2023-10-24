#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_hdf5objects.py
Description:
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

# Third-Party Packages #

# Local Packages #
from examples.operations.exampleoperationgroup import ExampleOperationGroup


# Definitions #
# Main #
if __name__ == "__main__":
    group = ExampleOperationGroup()
    out = group.evaluate()
    assert out is False
