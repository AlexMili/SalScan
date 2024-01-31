#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    Abstract base class for models.
    This class serves as a base for different types of models used in saliency prediction.
    It provides a common interface for all derived models, ensuring consistency and
    reusability.

    Attributes:
    params_to_store (list): List of attributes of the model stored in the
                            evaluation logs by
                            `SalScan.Evaluate.ImageSaliencyEvaluation` and
                            `SalScan.Evaluate.VideoSaliencyEvaluation`.
    """

    params_to_store = None

    @abstractmethod
    def __init__(self):
        if self.params_to_store is None:
            raise ValueError("You must define class attribute 'params_to_store'")

    @abstractmethod
    def run(self, img, params):
        """
        Runs the model to process an image based on provided parameters. This method must
        be overwritten by all the subclasses.

        Args:
            img (any): The image to be processed by the model.
            params (dict): Parameters required by the model.

        Returns:
            Varies: The output is model-specific.
        """
        raise NotImplementedError
