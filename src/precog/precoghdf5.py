"""precoghdf5.py
A HDF5 file which contains data for EEG data.
"""
# Package Header #
from .header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import pathlib
import datetime
from typing import Any

# Third-Party Packages #
from classversioning import VersionType, TriNumberVersion, Version
from dspobjects.time import Timestamp, nanostamp
import h5py
from hdf5objects.hdf5bases import HDF5Map, HDF5Dataset, GroupMap
from hdf5objects.dataset import BaseTimeSeriesMap
from hdf5objects import BaseHDF5Map, BaseHDF5

# Local Packages #


# Definitions #
# Classes #
class PrECoGModelGroupMap(GroupMap):
    default_attribute_names = {
        "convolutional_window_size": "convolutional_window_size",
        "beta": "beta",
        "rank": "rank",
        "motif_normalization": "motif_normalization",
        "motif_recentering": "motif_recentering",
        "motif_additive_noise": "motif_additive_noise",
        "motif_jitter_noise": "motif_jitter_noise",
        "oasis_tau_init": "oasis_tau_init",
        "oasis_tau_optimize": "oasis_tau_optimize",
        "penalty_l1_motif": "penalty_l1_motif",
        "penalty_l1_expression": "penalty_l1_expression",
        "penalty_ortho_motif": "penalty_ortho_motif",
        "penalty_ortho_expression": "penalty_ortho_expression",
        "penalty_ortho_cross": "penalty_ortho_cross",
        "learning_rate_motif_init": "learning_rate_motif_init",
        "learning_rate_expression_init": "learning_rate_expression_init",
        "iterations_motif_update": "iterations_motif_update",
        "iterations_expression_update": "iterations_expression_update",
    }
    default_map_names = {
        "h_matrix": "h_matrix",
        "w_matrix": "w_matrix",
    }
    default_maps = {
        "h_matrix": BaseTimeSeriesMap(shape=(0, 0, 0, 0), maxshape=(None, None, None, None)),
        "w_matrix": BaseTimeSeriesMap(shape=(0, 0, 0, 0), maxshape=(None, None, None, None)),
    }


class ModelsGroupMap(GroupMap):
    """A group for containing models."""

    default_map_names = {"learner_0": "learner_0"}
    default_maps = {"learner_0": PrECoGModelGroupMap()}


class PrECoGModelsFileMap(BaseHDF5Map):
    """A map for PrECOG files."""

    default_attribute_names = BaseHDF5Map.default_attribute_names | {
        "subject_id": "subject_id",
        "start": "start",
        "end": "end",
    }
    default_map_names = {"models": "models"}
    default_maps = {"models": ModelsGroupMap()}


class PrECoGModelsFile(BaseHDF5):
    """A HDF5 file which contains data for PrECoG Models.

    Class Attributes:
        _registration: Determines if this class will be included in class registry.
        _VERSION_TYPE: The type of versioning to use.
        FILE_TYPE: The file type name of this class.
        VERSION: The version of this class.
        default_map: The HDF5 map of this object.
    """

    _registration: bool = True
    _VERSION_TYPE: VersionType = VersionType(name="PrECoGModel", class_=TriNumberVersion)
    VERSION: Version = TriNumberVersion(0, 0, 0)
    FILE_TYPE: str = "PrECoGModel"
    default_map: HDF5Map = PrECoGModelsFileMap()
