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
from hdf5objects import BaseGroupComponent
from hdf5objects.hdf5bases import DatasetMap, HDF5Dataset, GroupMap
from hdf5objects.dataset import BaseTimeSeriesMap
from hdf5objects import BaseHDF5Map, BaseHDF5

# Local Packages #


# Todo: Include Bipolor Electrodes
# Todo: Include Streaming Parameters
# Todo: Include LineLength and Normalizor Parameter
# Definitions #
# Classes #
# Montage
class BipolorMontageMap(DatasetMap):
    default_attribute_names = {
        "anode_index": "anode_index",
        "cathode_index": "cathode_index",
    }
    default_attributes = {
        "anode_index": 0,
        "cathode_index": 1,
    }


class RereferenceComponent(BaseGroupComponent):
    def reref(self, ) -> None:
        self.composite["name"]
        # Todo: Add code (output coordinates and labels)


class MontageGroupMap(GroupMap):
    default_attribute_names = {
        "": "",
    }
    default_map_names = {
        "anatomy_labels": "anatomy_labels",
        "coordinates": "coordinates",
        "bipolor_map": "bipolar_map",
    }
    default_maps = {
        "anatomy_labels": DatasetMap(shape=(0,), maxshape=(None,)),
        "coordinates": DatasetMap(shape=(0, 0), maxshape=(None, 3)),
        "bipolor_map": BipolorMontageMap(shape=(0, 0), maxshape=(None, 2)),
        "bipolor_map_2": BipolorMontageMap(shape=(0, 0), maxshape=(None, 2)),
    }
    default_component_types = {
        "reference": (RereferenceComponent, {"bipolar_map"}),
    }


# Model
class PrECoGModelGroupMap(GroupMap):
    default_attribute_names = {
        "sampling_rate": "sampling_rate",
        "convolutional_window_size": "convolutional_window_size",
        "input_normalization": "input_normalization",
        "beta": "beta",
        "rank": "rank",
        "motif_normalization": "motif_normalization",
        "motif_recentering": "motif_recentering",
        "motif_additive_noise": "motif_additive_noise",
        "motif_jitter_noise": "motif_jitter_noise",
        "oasis_tau_rise": "oasis_tau_rise",
        "oasis_tau_decay": "oasis_tau_decay",
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
    default_attributes = {
        "sampling_rate": 1024.0,
        "convolutional_window_size": 256,
        "input_normalization": 0.0,
        "beta": 1.0,
        "rank": 10,
        "motif_normalization": "l1",
        "motif_recentering": "max",
        "motif_additive_noise": 0.0,
        "motif_jitter_noise": 0.0,
        "oasis_tau_rise": 10,
        "oasis_tau_decay": 200,
        "oasis_tau_optimize": True,
        "penalty_l1_motif": 0.0,
        "penalty_l1_expression": 0.0,
        "penalty_ortho_motif": 0.0,
        "penalty_ortho_expression": 0.0,
        "penalty_ortho_cross": 0.0,
        "learning_rate_motif_init": 86400,
        "learning_rate_expression_init": 0.0,
        "iterations_motif_update": 1,
        "iterations_expression_update": 3,
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


# File
class PrECoGModelsFileMap(BaseHDF5Map):
    """A map for PrECOG files."""

    default_attribute_names = BaseHDF5Map.default_attribute_names | {
        "subject_id": "subject_id",
        "start": "start",
        "end": "end",
    }
    default_map_names = {"montage": "montage", "models": "models"}
    default_maps = {"montage": MontageGroupMap(), "models": ModelsGroupMap()}


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
