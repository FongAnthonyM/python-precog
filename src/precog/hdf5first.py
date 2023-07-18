"""hdf5eeg.py
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
class ModelGroup(GroupMap):


class ModelFileMap(BaseHDF5Map):
    """A map for HDF5EEG files."""

    default_attribute_names = {
        "file_type": "FileType",
        "file_version": "FileVersion",
        "subject_id": "subject_id",
        "age": "age",
        "sex": "sex",
        "species": "species",
        "start": "start",
        "end": "end",
    }
    default_map_names = {"data": "EEG Array"}
    default_maps = {"data": BaseTimeSeriesMap()}


class HDF5EEG(BaseHDF5):
    """A HDF5 file which contains data for EEG data.

    Class Attributes:
        _registration: Determines if this class will be included in class registry.
        _VERSION_TYPE: The type of versioning to use.
        FILE_TYPE: The file type name of this class.
        VERSION: The version of this class.
        default_map: The HDF5 map of this object.

    Attributes:
        _subject_id: The ID of the EEG subject data.
        _subject_dir: The directory where subjects data are stored.

    Args:
        file: Either the file object or the path to the file.
        s_id: The subject id.
        s_dir: The directory where subjects data are stored.
        start: The start time of the data, if creating.
        init: Determines if this object will construct.
        **kwargs: The keyword arguments for the open method.
    """

    _registration: bool = False
    _VERSION_TYPE: VersionType = VersionType(name="HDF5EEG", class_=TriNumberVersion)
    VERSION: Version = TriNumberVersion(0, 0, 0)
    FILE_TYPE: str = "EEG"
    default_map: HDF5Map = HDF5EEGMap()