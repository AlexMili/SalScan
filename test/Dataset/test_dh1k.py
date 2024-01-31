import os
import numpy as np
from tqdm import tqdm
import unittest

from SalScan.Dataset.Video.DHF1K import DHF1KDataset


class Testing(unittest.TestCase):
    def setUp(self):
        base_path = os.path.expanduser("~")
        self.path = os.path.join(base_path, "datasets", "DHF1K")
        self.frame_size = []
        self.right_edge_index = None

    def test_populate(self):
        dataset = DHF1KDataset(self.path, sequence_length=1)
        dataset.populate()

    def test_Many_2_Many(self):
        dataset = DHF1KDataset(self.path, sequence_length=7, many_to_many=True)
        dataset.populate()
        for items in tqdm(dataset):
            frames, salmaps, fixmaps, _ = items
            assert len(frames) == len(salmaps) == len(fixmaps), self.fail(
                "Different lengths"
            )

    def test_seq_len_1(self):
        dataset = DHF1KDataset(self.path, sequence_length=1, eval_set_only=False)
        dataset.populate()
        idx_shift = dataset.sequence_length - 1 + dataset.eval_next_frame
        try:
            for idx, items in enumerate(tqdm(dataset)):
                video_frames = items[0]
                self.frame_size.append(len(video_frames))

                self.right_edge_index = idx + idx_shift
        except KeyError:
            self.fail("Failed test: test_seq_len_1")

        self.assertEqual(len(np.unique(self.frame_size)), 1)
        self.assertEqual(self.right_edge_index, len(dataset) + idx_shift - 1)

    def test_seq_len_1_next(self):
        dataset = DHF1KDataset(self.path, sequence_length=1, eval_next_frame=True)
        dataset.populate()
        idx_shift = dataset.sequence_length - 1 + dataset.eval_next_frame

        try:
            for idx, items in enumerate(tqdm(dataset)):
                video_frames = items[0]
                self.frame_size.append(len(video_frames))

                self.right_edge_index = idx + idx_shift
        except KeyError:
            self.fail("Failed test: test_seq_len_1_next")

        self.assertEqual(len(np.unique(self.frame_size)), 1)
        self.assertEqual(self.right_edge_index, len(dataset) + idx_shift - 1)

    # for sequence_length greater than 1 you cannot test the last index retrieved
    # because some indexes are actually skipped internally by the __iter__ method
    # via a continue statement
    def test_seq_len_5(self):
        dataset = DHF1KDataset(self.path, sequence_length=5)
        dataset.populate()
        try:
            for items in tqdm(dataset):
                video_frames = items[0]
                self.frame_size.append(len(video_frames))

        except KeyError:
            self.fail("Failed test: test_seq_len_5")

        self.assertEqual(len(np.unique(self.frame_size)), 1)

    def test_seq_len_5_next(self):
        dataset = DHF1KDataset(self.path, sequence_length=5, eval_next_frame=True)
        dataset.populate()
        try:
            for items in tqdm(dataset):
                video_frames = items[0]
                self.frame_size.append(len(video_frames))

        except KeyError:
            self.fail("Failed test: test_seq_len_5_next")

        self.assertEqual(len(np.unique(self.frame_size)), 1)


if __name__ == "__main__":
    unittest.main()
