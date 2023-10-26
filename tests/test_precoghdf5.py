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
#from src.precog.precoghdf5 import *


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


class TestPrECoGModelsFile(ClassTest):
    class_ = PrECoGModelsFile

    @pytest.mark.parametrize("mode", ["r", "r+", "a"])
    def test_new_object(self, mode, tmp_path):
        with self.class_(file=tmp_path / "test.h5", mode=mode) as f_obj:
            assert f_obj is not None
        assert True

    @pytest.mark.parametrize("mode", ["r", "r+", "a"])
    def test_load_whole_file(self, mode, tmp_path):
        with self.class_(file=tmp_path / "test.h5", mode=mode, load=True) as f_obj:
            assert f_obj is not None
        assert True

    def test_print_map(self):
        self.class_.default_map.print_tree()
        assert True

    def test_validate_file(self, tmp_path):
        file_path = tmp_path / "test.h5"
        assert self.class_.validate_file_type(file_path)

    def test_create_file(self, tmp_path):
        file_path = tmp_path / "test.h5"
        f_obj = self.class_(file=file_path, mode="a", create=True, construct=True)
        f_obj.print_contents()
        f_obj["montage"]["bipolar"]
        f_obj["montage"].components["reference"].reref()
        f_obj.close()
        assert True


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
