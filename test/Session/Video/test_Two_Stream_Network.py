import os

# Datasets
from SalScan.Dataset.Video.DHF1K import DHF1KDataset

# Models
from SalScan.Model.Saliency.Two_Stream_Network import Two_Stream_Network

# Session
from SalScan.Session.Saliency.VideoSession import VideoSaliencySession

# Data augumentation (per eval)
from SalScan.Transforms import Padding, padding_fixation

path2dhf = os.path.join("/home/ec2-user/datasets/DHF1K")
dataset = DHF1KDataset(
    path2dhf,
    sequence_length=7,
    many_to_many=True,
    transform_stimuli=Padding(channels=3),
    transform_salmap=Padding(channels=1, normalize=True),
    transform_fixmap=padding_fixation,
    to_array=True,
)
dataset.populate()


options = {
    "dataset": dataset,
    "model": Two_Stream_Network,
    "params": {
        "model": {},
    },
}

if __name__ == "__main__":
    session = VideoSaliencySession(options)
    session.evaluate()
