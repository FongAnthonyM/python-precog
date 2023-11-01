#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_hdf5objects.py
Description:
"""
# Package Header #
from src.precog.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import datetime
import pathlib
import timeit

# Third-Party Packages #
import pytest
import numpy as np

# Local Packages #
from precog.operations.standardizers import NNMFLineLengthStandardizer


# Definitions #
# Functions #
@pytest.fixture
def tmp_dir(tmpdir):
    """A pytest fixture that turn the tmpdir into a Path object."""
    return pathlib.Path(tmpdir)


# Classes #
class ClassTest:
    """Default class tests that all classes should pass."""

    class_ = None
    timeit_runs = 2
    speed_tolerance = 200

    def get_log_lines(self, tmp_dir, logger_name):
        path = tmp_dir.joinpath(f"{logger_name}.log")
        with path.open() as f_object:
            lines = f_object.readlines()
        return lines


class TestNNMFLineLengthStandardizer(ClassTest):

    def test_random_execute(self):
        samples = 102400
        channels = 512
        t_data = np.random.rand(samples, channels)

        standardizer = NNMFLineLengthStandardizer(
            forget_factor=10**-6,
        )
        out = standardizer.evaluate(data=t_data)

        assert np.all(out >= 0)

    def test_random_execute_decay_mean(self):
        samples = 102400
        channels = 512
        t_data = np.random.rand(samples, channels)

        standardizer = NNMFLineLengthStandardizer(
            forget_factor=10**-6,
            shift_scale="shift_decaying_mean",
        )
        out = standardizer.evaluate(data=t_data)

        assert np.all(out >= 0)


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
