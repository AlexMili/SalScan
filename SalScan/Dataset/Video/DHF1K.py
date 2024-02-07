# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""DHF1K Dataset."""

import glob
import os
from typing import Callable, List, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from SalScan.Dataset.VideoAbstractDataset import VideoAbstractDataset
from SalScan.Utils import get_logger

logger = get_logger(__name__)


class DHF1KDataset(VideoAbstractDataset):
    """
    DHF1K Dataset class for video saliency evaluation.

    It allows to download the dataset automatically and it handles the augmentation and
    retrieval of stimuli, saliency maps and fixation maps from the specified folder.
    Furthermore it allows to specify more technical parameters such as the sequence
    length, the model type (many-to-one or many-to-many) and the frame rate used for the
    evaluation.

    Parameters:
        root_path (str): Path to the dataset.
        sequence_length (int): Number of frames loaded for each step (idx iteration).
        sequence_target (int): Used for Many-to-One models, it means at which position
                                of sequence length the sequence target is positioned.
                                For example, given a 32 sequence length, if sequence
                                target is set to 16, it means the target will be the
                                frame in the middle of the sequence.
        evaluation_fps (int): The FPS of the retrieved stimuli. DHF1K defaults to 30
                                fps but we can evaluate at different frame rates. It
                                must be a value between 1 and 30 and it must be
                                divisible for evaluation_fps such that
                                `30 % evaluation_fps == 0`.
        many_to_many (bool): If True, model type is 'Many to Many'.
        eval_next_frame (bool): If set to `True`, the target for prediction is not the
                                last frame in the loaded sequence, but rather the frame
                                immediately following it. For instance, in a Many-to-One
                                model with a sequence length of 5 (loaded frames are t0,
                                t1, t2, t3, t4), setting this to `False` means the target
                                corresponds to frame t4. On the other hand, if set to
                                `True`, the target corresponds to frame t5, which follows
                                the sequence. This option is only applicable to
                                Many-to-One models.
        transform_stimuli (Callable): Transformation applied to video frames.
        transform_salmap (Callable): Transformation applied to saliency maps.
        transform_fixmap (Callable): Transformation applied to fixation maps.
        eval_set_only (bool): If True, retrieves videos only from the evaluation set.
        to_array (bool): If True, loaded frames are stacked on an array, otherwise will
                        be returned the list of the arrays.
        download (bool): If True, the dataset will be downloaded to the specified
                        `root_path`.

    Attributes:
        stimuli: Pandas dataframe composed by the following information: "id",
                "stimulus", "salmap", "fixmap", "label". These information will
                be inserted with the `populate` method.
        stimuli_shape: A tuple of int indicating the number of stimuli in the current
                        dataset
        name: "DHF1K"
        fps: indicates the fps used while filming videos. For DHF1K videos were shot
            at 30fps.
        has_label: Set to False since DHF1K folder structure does not allow
                    considering parent folders as labels.
        sliding_window: When `many_to_many` is set to `True`, sliding window is the
                        same set to `sequence_length`, otherwise is set to 1.
        start_idx: It is set to 601 when `eval_set_only` is set to `True` since videos
                    from 601 to 700 of DHF1K are used for evaluation. Otherwise is
                    set to 0.
        video: cv2.VideoCapture object containing the current video.
        tot_video_frames: set to the total amount of frames of a given `video`.
        frame_skip_interval: number of frames discarded to evaluate the video at
                            `evaluation_fps` instead of default `fps`. See inline
                            comments for a more detailed explanation.

    Reference:
        Wang, W., Shen, J., Xie, J., Cheng, M.-M., Ling, H., & Borji, A. (2021).
        Revisiting Video Saliency Prediction in the Deep Learning Era. IEEE Transactions
        on Pattern Analysis and Machine Intelligence.
    """

    name = "DHF1K"
    fps = 30
    has_label = False

    def __init__(
        self,
        root_path: str = None,
        sequence_length: int = 1,
        sequence_target: int = None,
        evaluation_fps: int = 30,
        many_to_many: bool = False,
        eval_next_frame: bool = False,
        transform_stimuli: Callable[..., np.array] = None,
        transform_salmap: Callable[..., np.array] = None,
        transform_fixmap: Callable[..., np.array] = None,
        eval_set_only: bool = True,
        to_array: bool = False,
        download: bool = False,
    ):
        # passing sequence lenght to check that is positive
        if sequence_target is not None and many_to_many is True:
            raise ValueError(
                "You cannot set sequence_target on many_to_many models: either set"
                " sequence_target=None or many_to_many=False"
            )
        if eval_next_frame is True and many_to_many is True:
            logger.warning(
                "\n WARN: many_to_many does not support next frame evaluation."
                " Setting eval_next_frame=False"
            )
            eval_next_frame = False

        if eval_next_frame is True and sequence_target is not None:
            raise ValueError(
                "The 'eval_next_frame' parameter is designed to shift the target frame "
                "to the one immediately following the sequence. However, this is "
                "incompatible with specifying a 'sequence_target', which is intended for "
                "use in models where the target frame is fixed within the sequence. To "
                "use 'eval_next_frame' ensure 'sequence_target' is set to None."
            )

        if evaluation_fps < 1 or evaluation_fps > self.fps:
            raise ValueError(
                f"evaluation_fps must be an integer between 1 and {self.fps}, which is"
                " the original video's evaluation_fps"
            )

        super().__init__(sequence_length=sequence_length)
        self.root_path = root_path
        if download is True:
            self._download()

        self.sequence_length = sequence_length
        self.many_to_many = many_to_many
        if self.many_to_many is True:
            self.sliding_window = self.sequence_length
        else:
            self.sliding_window = 1
        self.eval_next_frame = 1 if eval_next_frame else 0
        self.transform_stimuli = transform_stimuli
        self.transform_salmap = transform_salmap
        self.transform_fixmap = transform_fixmap
        self.to_array = to_array
        # if sequence target is not specified, default value is sequence lenght.
        self.sequence_target = (
            sequence_length if sequence_target is None else sequence_target
        )

        if eval_set_only:
            self.start_idx = 601
        else:
            self.start_idx = 1

        # Initializing two attributes that are going to be used for
        # loading video frames.

        # self._current_video during dataset iteration
        # is going to be set to the path of the last opened video
        self._current_video = None
        self._frame_idx = 0
        # self.video during dataset iteration
        # is going to be set to a VideoCapture object
        self.video = None
        self.tot_video_frames: int = None

        # The frame_skip_interval is calculated to convert the video to a different frame
        # rate (evaluation_fps) by selecting or skipping frames from the original frame
        # rate (self.fps). This is straightforward when the original fps is a multiple of
        # the target fps, as we can simply retrieve every nth frame. For example, for a
        # video with original fps of 30, to evaluate at 15 fps we select every other
        # frame (30/15), or every fifth frame for 6 fps (30/5). However, this approach is
        # not feasible for rates like 12 fps where the division results in a fraction
        # (30/12 is 2.5), since we cannot work with partial frames. Therefore, the logic
        # only works when the original fps can be evenly divided by the target fps to
        # ensure we pick whole frames at regular intervals. The following check raises an
        # error if the desired evaluation fps would require fractional frame selection,
        # which is not supported.
        if self.fps % evaluation_fps != 0:
            raise ValueError(
                f"The dataset default fps is {self.fps}. Please set an evaluation_fps such that "
                f"{self.fps}/evaluation_fps has no remainder."
            )
        # Given the above requirement, frame_skip_interval determines how many frames to
        # skip to achieve the desired evaluation fps. For instance, if the original fps
        # is 30 and the target is 10, frame_skip_interval is 3, meaning every third frame
        # is used in the resulting configuration.
        self.frame_skip_interval = int(self.fps / evaluation_fps)

    def populate(self) -> None:
        """
        Populates a dataframe with the paths of stimuli, fixation maps and saliency maps.
        """

        # define patterns and paths
        pattern = os.path.join(self.root_path, "video", "*.AVI")
        annotation_path = os.path.join(self.root_path, "annotation")
        videos = glob.glob(pattern)
        # filter only validation videos, which are video from 600 to 700
        videos = sorted(videos)[self.start_idx : 701]

        stimuli = []
        pbar_items = tqdm(videos)
        for video in pbar_items:
            # filtering file name with os invariant syntax
            # and removing the file format to load load both
            # stimulus and fixation path

            filename = os.path.basename(video).split(".")[0]
            fixmap_pattern = os.path.join(
                annotation_path, f"0{filename}", "fixation", "*.png"
            )
            salmap_pattern = os.path.join(annotation_path, f"0{filename}", "maps", "*.png")
            # fixations and stimuli need to be sorted because
            # since their order represent the frames, both
            # are going to be loaded by index
            fixmaps = sorted(glob.glob(fixmap_pattern))
            salmaps = sorted(glob.glob(salmap_pattern))

            sequence = []
            for i in range(len(fixmaps)):
                sequence.append([video, salmaps[i], fixmaps[i]])

            stimuli.extend(sequence)

        self.stimuli = pd.DataFrame(stimuli, columns=["video", "salmap", "fixmap"])

    def _new_video(self, path):
        """
        Loads a new video based on its path.

        Parameter:
            path (str): The path of the video file.
        """
        self._frame_idx = 0
        self.video = cv2.VideoCapture(path)
        self.tot_video_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def _load_video_frames(
        self, path: str, sequence_length: int
    ) -> Union[List[np.array], np.array]:
        """
        Loads a sequence of video frames. The number of frames is equal to sequence length

        This method loads a specified number of frames from a video by taking into account
        the evaluation frame rate and eventually by applying the specified aumentation.

        Parameters:
            path (str): The path of the video file.
            sequence_length (int): The number of frames to load in the sequence.

        Returns:
            Union[List[np.array], np.array]: Depending on self.to_array specified during
                                            class initialisation, it returns either a
                                            list of numpy arrays or a numpy array where
                                            video frames are stacked on the first
                                            dimension.
        """
        # when a new video needs to be loaded, we are setting frame_idx
        # to 0 to start loading from the first frame
        if path != self._current_video:
            self._new_video(path)
        # ovewriting the last video path with the current one
        self._current_video = path
        # starting to read frames from first frame if is the first
        # time the video path is present or from the frame after the
        # last loaded one if the video is the same as in the previous
        # data loading
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
        frames = []

        # total_frames_to_read: Specifies the total number of frames that must be read
        # from the video in order to obtain a sequence of frames with the desired length
        # at the adjusted frame rate. This calculation accounts for the
        # frame_skip_interval, which determines how frequently frames are selected based
        # on the target fps. For example, if the sequence_length is 10, and the
        # frame_skip_interval is 3, total_frames_to_read will be 30, meaning that 30
        # frames are read to obtain 10 frames at the adjusted frame rate.
        total_frames_to_read = sequence_length * self.frame_skip_interval
        for idx in range(total_frames_to_read):
            _, frame = self.video.read()
            # Use the modulo operator to select frames at regular intervals based on the
            # frame_skip_interval. The modulo operation here checks if the current index
            # is a multiple of the frame_skip_interval. i.e. if frame_skip_interval
            # is 3, frames at indexes 0, 3, 6, 9, etc., will be used, and the others
            # are discarded
            if idx % self.frame_skip_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Check if a transformation function for stimuli is defined
                if self.transform_stimuli is not None:
                    # Apply the transformation function to the current frame
                    frame = self.transform_stimuli(frame)
                # When the sequence length is greater than 1, the frame is appended to
                # the frames list. This is done only when multiple frames need to be
                # processed together in order to avoid unnecessary list handling for
                # models requiring a sequence length equal to one.
                if self.sequence_length > 1:
                    frames.append(frame)

        # last frames to be loaded
        if (self.tot_video_frames - self._frame_idx) == sequence_length:
            self.video.release()

        if self.to_array is True:
            frames = np.stack(frames, axis=0)
        # Return the appropriate structure based on the sequence length. If only a single
        # frame is processed (sequence_length <= 1), return the frame directly. Otherwise,
        # return the array/list of frames.
        if self.sequence_length > 1:
            return frames
        else:
            return frame

    def _load_saliencymaps(self, paths: List[str]) -> List[np.array]:
        """
        Loads saliency maps from the specified paths by eventually applying the
        specified data augmentation.

        Parameters:
            paths (List[str]): A list of paths to the saliency map files.

        Returns:
            List[np.array]: A list of loaded saliency maps.
        """
        sal_maps = []
        for path in paths:
            sal_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            # sal_map = sal_map.astype(float) / 255.0
            if self.transform_salmap is not None:
                sal_map = self.transform_salmap(sal_map)
            sal_map = sal_map.astype(float)

            # normalise saliency_map between 0 and 1
            sal_map = cv2.normalize(sal_map, sal_map, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            sal_maps.append(sal_map)

        return sal_maps

    def _load_fixationmaps(self, paths: List[str]) -> List[np.array]:
        """
        Loads fixations maps from the specified paths by eventually applying the
        specified data augmentation.
        Args:
            paths (List[str]): A list of paths to the fixation map files.

        Returns:
            List[np.array]: A list of loaded fixation maps.
        """
        fix_maps = []
        for path in paths:
            fix_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if self.transform_fixmap is not None:
                fix_map = self.transform_fixmap(fix_map)
            fix_map = fix_map.astype(float)
            # For AUC metric calculations, fixation map values must be binary (0s and 1s).
            # This is because the AUC metric counts true positives when the sum of
            # fixation coordinates and corresponding values on a discretized saliency map
            # equals 2. This sum can only reach 2 with binary values. Although
            # intuitively fixation points might be set to 1, some saliency datasets
            # represent them with continuous values ranging from 0 to 1. Therefore, we
            # binarize the map by treating any non-zero value as a fixation (1) and zeros
            # remain unchanged (0).
            fix_map = fix_map > 0
            fix_maps.append(fix_map)
        return fix_maps

    def __iter__(self):
        """
        Iterator for the dataset.

        Yields:
            A tuple containing loaded video frames, saliency maps, fixation maps, and
            None (as a placeholder for future labels).
        """
        # Calculate the last index for iteration. This takes into account the sequence
        # length and adjusts if we need to include the evaluation of the next frame
        # (self.eval_next_frame). For example, with 100 stimuli and a sequence length
        # of 5, we stop at the 95th stimulus to ensure the last sequence includes the
        # 99th (final) stimulus.
        last_index = len(self.stimuli) - self.sequence_length + 1 - self.eval_next_frame

        # The step size for iterating through stimuli depends on the sliding window size
        # (which is equal to the sequence length in a many-to-many setting, otherwise 1)
        # and the frame skipping interval (self.frame_skip_interval).
        step_size = self.sliding_window * self.frame_skip_interval

        # Iterate through the stimuli. Each iteration step considers a sequence of frames
        # determined by the step size and the sequence length.
        for idx in range(0, last_index, step_size):
            # Calculate the target index within the sequence. The target index is
            # determined by the current index 'idx' and the position of the target
            # frame within the sequence (self.sequence_target), adjusted by the frame
            # skip interval. We subtract '1' to align with Python's zero-based indexing,
            # as pandas loc indexer is right-inclusive.
            target_index = idx + (self.sequence_target - 1) * self.frame_skip_interval
            # The upper_bound_idx represents the ending index for the sequence of frames
            # being considered. This is calculated similarly to target_index but uses
            # self.sequence_length to determine the farthest frame to include in the
            # sequence. This index is used to define the range of frames to be processed
            # or extracted.
            upper_bound_idx = idx + (self.sequence_length - 1) * self.frame_skip_interval
            # Example: For a model with a sequence_length of 32, a sequence_target of 16,
            # idx starting at 0, and a frame_skip_interval of 1:
            # - target_index would be 0 + (16 - 1) * 1 = 15. This means the target frame
            #   is the 16th frame in the sequence.
            # - upper_bound_idx would be 0 + (32 - 1) * 1 = 31. This means the sequence
            #   includes frames from index 0 to 31.

            # Example with a frame_skip_interval of 2:
            # For a model with sequence_length of 32, sequence_target of 16, idx starting
            # at 0, and frame_skip_interval of 2:
            # - target_index would be 0 + (16 - 1) * 2 = 30. This means the target frame
            #   is the 16th frame in the sequence, but considering every second frame
            #   (due to skipping), it's physically located at index 30 in the original
            #   frame sequence.
            # - upper_bound_idx would be 0 + (32 - 1) * 2 = 62. This means the sequence
            #   includes frames from index 0 to 62, skipping every alternate frame.

            # we slice videos considering the whole sequence length window.
            # if the evaluation_fps is lower than the default one, we consider a frame
            # every self.frame_skip_interval.
            videos = self.stimuli.loc[
                idx : upper_bound_idx : self.frame_skip_interval, "video"
            ]

            # if the sequence_length of frames to load contains two filenames (videos)
            # it means that we have finished loading the frames of the current
            # videos. sequence_length observation will be skipped (continue) and
            # self._frame_idx is set to 0 (beginning of new video)
            if len(pd.unique(videos)) > 1:
                continue

            filename = videos.iloc[0]
            frames = self._load_video_frames(filename, self.sequence_length)

            # the following mechanism allows to retrieve a 'sequence length' amount
            # of targets we the model is many to many: when 'self.many_to_many' is True,
            # we are going to retrieve all the labels from the current index (lower_index)
            # to target_index. When self.many_to_many is False on the other hand only the
            # target index is retrieved
            if self.many_to_many is True:
                lower_index = idx
            else:
                lower_index = target_index

            salmaps = self._load_saliencymaps(
                self.stimuli.loc[
                    lower_index : target_index
                    + self.eval_next_frame : self.frame_skip_interval,
                    "salmap",
                ].tolist(),
            )

            fixmaps = self._load_fixationmaps(
                self.stimuli.loc[
                    lower_index : target_index
                    + self.eval_next_frame : self.frame_skip_interval,
                    "fixmap",
                ].tolist(),
            )

            # Adjusting the frame index (self._frame_idx) for the next iteration.
            # This adjustment is contingent on the nature of the model in use.
            # For many-to-one (not many-to-many), the frame shift is simply 1, indicating
            # a step to the next frame without skipping or considering a sequence.
            # In a many-to-many model, the next evaluation starts after the current
            # sequence. The shift in frame index (step) accounts for the processed
            # sequence and frame skipping behavior. This ensures that each new iteration
            # begins at the correct position in the video, avoiding reprocessing of
            # frames and adhering to the desired evaluation_fps rate.
            # Example 1: sequence_length = 32, frame_skip_interval = 1
            # In this case, for a many-to-many model, the frame index will be incremented
            # by 32. This means after processing a 32-frame sequence, the next sequence
            # starts 32 frames later.

            # Example 2: sequence_length = 32, frame_skip_interval = 2
            # Here, the frame index increment will be 32 * 2 = 64.
            # This reflects the processing of every alternate frame in a 32-frame
            # sequence, effectively placing the start of the next sequence 64 frames
            # ahead in the original video.

            if self.many_to_many is True:
                step = self.sequence_length * self.frame_skip_interval
            else:
                step = 1
            self._frame_idx += step

            yield frames, salmaps, fixmaps, None

    def __len__(self):
        """
        Returns the length of the dataset.

        The length depends on whether the model is many-to-many or many-to-one, as well
        as on the sequence length and on the evaluation frame rate.

        Returns:
            int: The length of the dataset.
        """
        if self.many_to_many:
            return int(
                len(self.stimuli) / (self.sequence_length * self.frame_skip_interval)
            )
        else:
            return len(self.stimuli)

    def _download(self):
        """
        Downloads the DHF1K dataset on your machine.

        If the dataset is not present at the specified root path, this method will
        download it. Requires 'unar' installed in the system for unarchiving the dataset.
        """
        logger.warning(
            "\n 🔴 You need to have unar installed in your terminal (https://theunarchiver.com/command-line)"
        )
        if os.path.isdir(self.root_path):
            logger.warning(
                f"\n 🔴 The output directory {self.root_path} is already present on your "
                "disk and the dataset won't be downloaded. \n"
                f"In order to silence this warning message set the {self.name} "
                "parameter download=False \n"
            )
        else:
            os.makedirs(os.path.join(self.root_path, "trainSet"))
            logger.warning(
                "\n 🔴 Once you download the dataset, turn-off the auto-download feature:"
                " the administrator limited the amount of bandwidth you can use.\n"
            )
            self._download_dataset(
                "https://drive.google.com/drive/folders/1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ?usp=drive_link",
                self.root_path,
            )
