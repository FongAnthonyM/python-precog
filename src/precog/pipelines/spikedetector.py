""" spikedetector.py.py

"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
from abc import abstractmethod
from typing import Any, Callable

# Third-Party Packages #
import numpy as np

# Local Packages #
from ..operations.operation import OperationGroup
from ..operations.streamers import CDFSStreamer
from ..operations.remapper import Remapper
from ..operations.preprocessingfilterbank import PreprocessingFilterBank
from ..operations.standardizers import NNMFLineLengthStandardizer


# Definitions #
# Classes #
class SpikeDetector(OperationGroup):
    streamer_type = CDFSStreamer
    remapper_type = Remapper
    preprocessing_type = PreprocessingFilterBank
    standardizer_type = NNMFLineLengthStandardizer
    learner_type = None

    # Operations
    def create_operations(
        self,
        streamer_kwargs: dict[str, Any] | None = None,
        remapper_kwargs: dict[str, Any] | None = None,
        preprocessing_kwargs: dict[str, Any] | None = None,
        standardizer_kwargs: dict[str, Any] | None = None,
        learner_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Create Operations
        self.operations["data_streamer"] = self.streamer_type(**(streamer_kwargs or {}))
        self.operations["data_remapper"] = self.remapper_type(**(remapper_kwargs or {}))
        self.operations["preprocessing"] = self.preprocessing_type(**(preprocessing_kwargs or {}))
        self.operations["standardizer"] = self.standardizer_type(**(standardizer_kwargs or {}))
        self.operations["leaner"] = self.learner_type(**(learner_kwargs or {}))

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Operations
        data_streamer = self.operations["data_streamer"]
        data_remapper = self.operations["data_remapper"]
        preprocessing = self.operations["preprocessing"]
        standardizer = self.operations["standardizer"]
        learner = self.operations["leaner"]

        # Set Input


        # Inner IO
        data_remapper.inputs["data"] = data_streamer.outputs["data"]
        preprocessing.inputs["data"] = data_remapper.inputs["remapped_data"]
        standardizer.inputs["data"] = preprocessing.outputs["filtered_data"]
        learner.inputs["data"] = standardizer.outputs["features"]

        # Set Output
