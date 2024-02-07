from abc import abstractmethod
from typing import Callable

import numpy as np

from .AbstractDataset import AbstractDataset


class VideoAbstractDataset(AbstractDataset):
    """Video Datasets abstract class.

    This class cannot be instantiated and must be inherited from a child class. It's
    purpose is to define a structure and some basic functions in order to implement any
    video dataset class.

    Parameters:
        root_path (str): to indicate a path to the dataset.
        sequence_length (int): Number of frames loaded for each step (idx iteration).
        eval_next_frame (bool): If set to `True`, the target for prediction is not the
                                last frame in the loaded sequence, but rather the frame
                                immediately following it. For instance, in a Many-to-One
                                model with a sequence length of 5 (loaded frames are t0,
                                t1, t2, t3, t4), setting this to `False` means the target
                                corresponds to frame t4. On the other hand, if set to
                                `True`, the target corresponds to frame t5, which follows
                                the sequence. This option is only applicable to
                                Many-to-One models.
        transform (Callable): Transformation applied to video frames.

    Attributes:
        name: A string indicating the name of the dataset.
        has_label (bool): It describes if the stimuli are organized within the folder such
                            that each stimulus parent folder can be considered as its
                            label.
        stimuli: Pandas dataframe composed by the following information: "id",
                "stimulus", "salmap", "fixmap", "label". These information will
                be inserted with the `populate` method.
        stimuli_shape: A tuple of int indicating the number of stimuli in the current
                        dataset
        fixations_shape: A tuple of int indicating the number of fixations in the current
                        dataset
        type: A string indicating the type of the dataset. i.e. "video_saliency"
        structure (Optional): Dict containing dataset's folder structure.
    """

    @abstractmethod
    def __init__(
        self,
        root_path: str = None,
        sequence_length: int = 1,
        eval_next_frame: bool = False,
        transform: Callable[..., np.array] = None,
    ):
        super().__init__()
        self.root_path = root_path
        self.sequence_length = sequence_length
        self.eval_next_frame = 1 if eval_next_frame else 0
        self.transform = transform

    @abstractmethod
    def populate(self, force: bool = False):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    # is abstractmethod in parent class, therefore we overwrite it
    # even if we are not using it
    def _load_stimulus(self, path: str):
        raise NotImplementedError

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        if value < 1:
            raise ValueError("sequence_length must be greater or equal to 1")
        self._sequence_length = value

    @property
    def eval_next_frame(self):
        return self._eval_next_frame

    @eval_next_frame.setter
    def eval_next_frame(self, value):
        # current_frame is converted to int in while initialising class.
        if (isinstance(value, bool) is True) or (value in [0, 1]):
            self._eval_next_frame = value
        else:
            raise ValueError(
                "self.eval_next_frame must can be only set to one of [True, False, 1, 0]"
            )
