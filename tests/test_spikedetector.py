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

# Local Packages #
from src.precog.pipelines import SpikeDetector


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


class TestSpikeDetector(ClassTest):
    subjects_root = pathlib.Path("/data_store0/human/converted_clinical")
    subject_id = "EC0212"

    def test_constrcution(self):
        detector = SpikeDetector(preprocessing={"sample_rate": 1024})
        bases = detector.model.get_bases()
        state_variables = detector.model.get_state_variables()
        assert detector is not None

    def test_evaluate_stream(self):
        # Import Package
        from xltektools.xltekucsfbids import IEEGXLTEK

        # Select Subject
        bids_subject = Subject(name=self.subject_id, parent_path=self.subjects_root)
        session = bids_subject.sessions["clinicalintracranial"]
        ieeg = session.modalities["ieeg"]
        cdfs = ieeg.require_cdfs()
        cdfs.open(mode="r", load=True)

        # Remap Channels
        sample_rate = cdfs.data.sample_rates[1]
        montage = ieeg.load_electrodes()

        max_channels = np.array(cdfs.data.shapes).max(0)[1]
        used_channels = len(montage["name"][:-4])
        remap = np.zeros((max_channels, used_channels))
        remap[:used_channels, :] = np.identity(used_channels)  # Remap Channels from Montage

        # Create Pipeline
        spike_detector = SpikeDetector(
            streamer={"cdfs": cdfs},
            remapper={"map_matrix": remap},
            preprocessing={"sample_rate": sample_rate},
            standardizer={"forget_factor": 10**-6, "shift_scale": "shift_decaying_mean"},
        )

        # Create Tensors
        window_size = int(sample_rate * 0.250)
        n_motifs = 10
        arc_bases = spike_detector.model.bases["architecture"]
        arc_bases["W"].create_tensor(size=(remap.shape[1], n_motifs, window_size))
        arc_bases["H"].create_tensor(size=(1, n_motifs, int(sample_rate * 10) - window_size + 1))

        # Select Time Range
        start = datetime.datetime(1970, 1, 7, 0, 5, 0, tzinfo=datetime.timezone.utc)
        stop = datetime.datetime(1970, 1, 7, 0, 9, 10, tzinfo=datetime.timezone.utc)

        streamer = spike_detector.operations["streamer"]
        streamer.setup(start=start, stop=stop, step=10, approx=True, tails=True)

        # Evaluate
        spike_detector.evaluate()

        assert True



# Main #
if __name__ == "__main__":
    # pytest.main(["-v", "-s"])
    t = TestSpikeDetector()
    t.test_evaluate_stream()