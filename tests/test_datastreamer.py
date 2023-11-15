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
from ucsfbids import Subject
from h5py import Dataset

# Local Packages #
from src.precog.operations import CDFSStreamer


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


class TestCDFSStreamer(ClassTest):
    subjects_root = pathlib.Path("/data_store0/human/converted_clinical")
    subject_id = "EC0212"

    def test_evaluate_stream(self):
        from xltektools.xltekucsfbids import IEEGXLTEK
        start = datetime.datetime(1970, 1, 6, 0, 0, tzinfo=datetime.timezone.utc)
        stop = datetime.datetime(1970, 1, 6, 1, 0, tzinfo=datetime.timezone.utc)

        bids_subject = Subject(name=self.subject_id, parent_path=self.subjects_root)
        session = bids_subject.sessions["clinicalintracranial"]
        cdfs = session.modalities["ieeg"].require_cdfs()

        streamer = CDFSStreamer(cdfs=cdfs)
        streamer.setup(start=start, stop=stop, step=10, approx=True, tails=True)

        outs = [streamer.evaluate() for i in range(360)]

        assert True


# Main #
if __name__ == "__main__":
    # pytest.main(["-v", "-s"])
    t = TestCDFSStreamer()
    t.test_evaluate_stream()
