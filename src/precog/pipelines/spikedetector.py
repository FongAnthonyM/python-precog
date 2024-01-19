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
from collections.abc import Mapping
from typing import Any, Callable

# Third-Party Packages #
import numpy as np

# Local Packages #
from ..operations.operation import BaseOperation, OperationGroup
from ..operations.streamers import CDFSStreamer
from ..operations.remapper import Remapper
from ..operations.preprocessingfilterbank import PreprocessingFilterBank
from ..operations.standardizers import NNMFLineLengthStandardizer
from ..models import BaseModel
from ..models.torch import NNMFDTorchModel


# Definitions #
# Classes #
class SpikeDetector(OperationGroup):
    # Attributes #
    streamer_type = CDFSStreamer
    remapper_type = Remapper
    preprocessing_type = PreprocessingFilterBank
    standardizer_type = NNMFLineLengthStandardizer
    model_type = NNMFDTorchModel
    detector_type = None

    model: BaseModel | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        model: BaseModel | None = None,
        streamer: BaseOperation | dict[str, Any] | None = None,
        remapper: BaseOperation | dict[str, Any] | None = None,
        preprocessing: BaseOperation | dict[str, Any] | None = None,
        standardizer: BaseOperation | dict[str, Any] | None = None,
        detector: BaseOperation | dict[str, Any] | None = None,
        *args: Any,
        operations: Mapping[str, BaseOperation] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                model=model,
                streamer=streamer,
                remapper=remapper,
                preprocessing=preprocessing,
                standardizer=standardizer,
                detector=detector,
                operations=operations,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        model: BaseModel | None = None,
        streamer: BaseOperation | dict[str, Any] | None = None,
        remapper: BaseOperation | dict[str, Any] | None = None,
        preprocessing: BaseOperation | dict[str, Any] | None = None,
        standardizer: BaseOperation | dict[str, Any] | None = None,
        detector: BaseOperation | dict[str, Any] | None = None,
        *args: Any,
        operations: Mapping[str, BaseOperation] | None = None,
        init_io: Any = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            operations: The dictionary of Operation to add to the OperationGroup.
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        # New Assignment #
        if model is not None:
            self.model = model
        
        # Kwargs for Operation Creation
        create_kwargs = {}
        
        if isinstance(streamer, BaseOperation):
            self.operations["streamer"] = streamer
        elif isinstance(streamer, dict):
            create_kwargs["streamer_kwargs"] = streamer
            
        if isinstance(remapper, BaseOperation):
            self.operations["remapper"] = remapper
        elif isinstance(remapper, dict):
            create_kwargs["remapper_kwargs"] = remapper
            
        if isinstance(preprocessing, BaseOperation):
            self.operations["preprocessing"] = preprocessing
        elif isinstance(preprocessing, dict):
            create_kwargs["preprocessing_kwargs"] = preprocessing
            
        if isinstance(standardizer, BaseOperation):
            self.operations["standardizer"] = standardizer
        elif isinstance(standardizer, dict):
            create_kwargs["standardizer_kwargs"] = standardizer
            
        if isinstance(detector, BaseOperation):
            self.operations["detector"] = detector
        elif isinstance(detector, dict):
            create_kwargs["detector_kwargs"] = detector

        if setup_kwargs is None and create_kwargs:
            setup_kwargs = {"create_kwargs": create_kwargs}
        elif setup_kwargs is not None and (c_kwargs := setup_kwargs.get("create_kwargs")) is not None:
            setup_kwargs["create_kwargs"] = setup_kwargs | c_kwargs

        # Construct Parent #
        super().construct(
            operations=operations,
            init_io=init_io,
            sets_up=sets_up,
            setup_kwargs=setup_kwargs,
            **kwargs,
        )

    # Operations
    def create_detector(self, *args, **kwargs) -> BaseOperation:
        return self.model.trainer

    def create_operations(
        self,
        streamer_kwargs: dict[str, Any] | None = None,
        remapper_kwargs: dict[str, Any] | None = None,
        preprocessing_kwargs: dict[str, Any] | None = None,
        standardizer_kwargs: dict[str, Any] | None = None,
        detector_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Create Operations
        self.operations["streamer"] = self.streamer_type(**(streamer_kwargs or {}))
        self.operations["remapper"] = self.remapper_type(**(remapper_kwargs or {}))
        self.operations["preprocessing"] = self.preprocessing_type(**(preprocessing_kwargs or {}))
        self.operations["standardizer"] = self.standardizer_type(**(standardizer_kwargs or {}))
        self.operations["detector"] = self.create_detector(**(detector_kwargs or {}))

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Operations
        streamer = self.operations["streamer"]
        remapper = self.operations["remapper"]
        preprocessing = self.operations["preprocessing"]
        standardizer = self.operations["standardizer"]
        detector = self.operations["detector"]

        # Inner IO
        streamer.outputs["data"] = remapper.inputs["data"]
        remapper.outputs["remapped_data"] = preprocessing.inputs["data"]
        preprocessing.outputs["filter_data"] = standardizer.inputs["data"]
        standardizer.outputs["features"] = detector.inputs["data"]

        # Set Output
        self.outputs = detector.outputs  # Make an indirect assignment

    # Setup
    def setup(
        self,
        model_kwargs: dict[str, Any] | None = None,
        *args: Any,
        create: bool = True,
        create_kwargs: dict[str, Any] | None = None,
        link: bool = True,
        link_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates the inner operations and links their IO.

        Args:
            *args: The arguments for setup.
            create: Determines if the inner operation will be created.
            create_kwargs: The keyword arguments for creating the inner operations.
            link: Determines if the inner IO will be linked between operations.
            link_kwargs: The keyword arguments for creating linking the inner operations' IO.
            **kwargs: The keyword arguments for setup.4
        """
        if self.model is None:
            self.model = self.model_type(**(model_kwargs or {}))

        super().setup(
            *args,
            create=create,
            create_kwargs=create_kwargs,
            link=link,
            link_kwargs=link_kwargs,
            **kwargs,
        )