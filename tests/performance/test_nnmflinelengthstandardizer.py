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
import cProfile
import datetime
import io
import os
import pathlib
import pickle
import pstats
from pstats import Stats, f8, func_std_string
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
class StatsMicro(Stats):
    def print_stats(self, *amount):
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = "  \n"
        for func in self.top_level:
            print(indent, func_get_function_name(func), file=self.stream)

        print(indent, self.total_calls, "function calls", end=" ", file=self.stream)
        if self.total_calls != self.prim_calls:
            print("(%d primitive calls)" % self.prim_calls, end=" ", file=self.stream)
        print("in %.3f microseconds" % (self.total_tt * 1000000), file=self.stream)
        print(file=self.stream)
        width, list = self.get_print_list(amount)
        if list:
            print('ncalls'.rjust(16), end='  ', file=self.stream)
            print('tottime'.rjust(12), end='  ', file=self.stream)
            print('percall'.rjust(12), end='  ', file=self.stream)
            print('cumtime'.rjust(12), end='  ', file=self.stream)
            print('percall'.rjust(12), end='  ', file=self.stream)
            print('filename:lineno(function)', file=self.stream)
            for func in list:
                self.print_line(func)
            print(file=self.stream)
            print(file=self.stream)
        return self

    def print_line(self, func):  # hack: should print percentages
        cc, nc, tt, ct, callers = self.stats[func]
        c = str(nc)
        if nc != cc:
            c = c + "/" + str(cc)
        print(c.rjust(16), end="  ", file=self.stream)
        print(f8(tt * 1000000).rjust(12), end="  ", file=self.stream)
        if nc == 0:
            print(" " * 12, end="  ", file=self.stream)
        else:
            print(f8(tt / nc * 1000000).rjust(12), end=" ", file=self.stream)
        print(f8(ct * 1000000).rjust(12), end="  ", file=self.stream)
        if cc == 0:
            print(" " * 12, end="  ", file=self.stream)
        else:
            print(f8(ct / cc * 1000000).rjust(12), end=" ", file=self.stream)
        print(func_std_string(func), file=self.stream)


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
        samples = 10240
        channels = 256
        t_data = np.random.normal(loc=7, scale=3, size=(samples, channels))

        standardizer = NNMFLineLengthStandardizer(
            forget_factor=10**-6,
        )

        pr = cProfile.Profile()
        pr.enable()

        out = standardizer.evaluate(data=t_data)

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = StatsMicro(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        assert np.all(out >= 0)

    def test_random_execute_decay_mean(self):
        samples = 1024
        channels = 512
        t_data = np.random.rand(samples, channels)

        standardizer = NNMFLineLengthStandardizer(
            forget_factor=10**-6,
            shift_scale="shift_decaying_mean",
        )

        pr = cProfile.Profile()
        pr.enable()

        out = standardizer.evaluate(data=t_data)

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME
        ps = StatsMicro(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        assert np.all(out >= 0)


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
