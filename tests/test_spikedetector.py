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
from src.precog.models import EnsembleModel
from src.precog.models.torch import NNMFDTorchModel
from src.precog.basis.torch import NonNegativeBasis
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

    def closest_square(self, n):
        n = int(n)
        i = int(np.ceil(np.sqrt(n)))
        while True:
            if (n % i) == 0:
                break
            i += 1
        assert n == (i * (n // i))
        return i, n // i

    def make_bipolar(self, lead_group):
        for l_name in lead_group:
            sel_lead = lead_group[l_name]
            n_contact = len(sel_lead['IDs'])
            if 'grid' in sel_lead['Type']:
                n_row, n_col = self.closest_square(n_contact)
            else:
                n_row, n_col = [n_contact, 1]

            CA = np.arange(len(sel_lead['Contacts'])).reshape((n_row, n_col), order='F')

            lead_group[l_name]['Contact_Pairs_ix'] = []

            if n_row > 1:
                for bp1, bp2 in zip(CA[:-1, :].flatten(), CA[1:, :].flatten()):
                    lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))

            if n_col > 1:
                for bp1, bp2 in zip(CA[:, :-1].flatten(), CA[:, 1:].flatten()):
                    lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))

        return lead_group

    def make_bipolar(self, montage):
        groups = []
        g_name = montage["name"][0].strip("1234567890")
        f_index = 0
        for i, row in enumerate(montage):
            if (new_name := row["name"].strip("1234567890")) != g_name:
                groups.append((g_name, montage[f_index:i]))
                g_name = new_name
                f_index = i
        groups.append((g_name, montage[f_index:len(montage)]))

        pb_groups = {}
        remap_contacts = []
        cn_contacts = 0
        for (name, group) in groups:
            type_ = group["group"][0]
            c_names = group["names"]
            n_contact = len(group)
            n_row, n_col = self.closest_square(n_contact) if 'grid' in type_ else (n_contact, 1)

            CA = np.arange(n_contact).reshape((n_row, n_col), order='F')

            pb_groups[name] = bp_contacts = []

            if n_row > 1:
                for bp1, bp2 in zip(CA[:-1, :].flatten(), CA[1:, :].flatten()):
                    bp_contacts.append((c_names[bp1], c_names[bp2], bp1, bp2))
                    remap_contacts.append((bp1 + cn_contacts, bp2 + cn_contacts))

            if n_col > 1:
                for bp1, bp2 in zip(CA[:, :-1].flatten(), CA[:, 1:].flatten()):
                    bp_contacts.append((c_names[bp1], c_names[bp2], bp1, bp2))
                    remap_contacts.append((bp1 + cn_contacts, bp2 + cn_contacts))

            cn_contacts += n_contact

        remap = np.zeros((len(montage), len(remap_contacts)))
        for i, (a, c) in enumerate(remap_contacts):
            remap[a, i] = 1
            remap[c, i] = -1

        return pb_groups, remap

    def test_construction(self):
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
        b_groups, remap = self.make_bipolar(montage)

        # max_channels = np.array(cdfs.data.shapes).max(0)[1]
        # used_channels = len(montage["name"][:-4])
        # remap = np.zeros((max_channels, used_channels))
        # remap[:used_channels, :] = np.identity(used_channels)  # Remap Channels from Montage

        # Create Tensor Info
        window_size = int(sample_rate * 0.250)
        n_motifs = 10
        w_size = (remap.shape[1], n_motifs, window_size)
        h_size = (1, n_motifs, int(sample_rate * 10) - window_size + 1)

        # Create Models
        submodels = {}

        submodels["first_model"] = NNMFDTorchModel(
            architecture={"W": NonNegativeBasis(size=w_size), "H": NonNegativeBasis(size=h_size)},
            trainer={"state_variables": {"W_modifier", {}, "H_modifier", {}}},
        )

        model = EnsembleModel(submodels=submodels)

        # Create Pipeline
        spike_detector = SpikeDetector(
            model=model,
            streamer={"cdfs": cdfs},
            remapper={"map_matrix": remap},
            preprocessing={"sample_rate": sample_rate},
            standardizer={"forget_factor": 10**-6, "burn_in": 10},  # Todo: Handle burn in (automatic)
        )

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
