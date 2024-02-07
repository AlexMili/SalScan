"""
Time to evaluate all frames: 351 secs
"""
from os import path as osp
from timeit import default_timer as timer
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from SalScan.Model.Saliency.UniSal import UniSal
from SalScan.Transforms import Preprocess_UniSal
from SalScan.Utils import load_video, release_video

EXPORT_RESULT: bool = True
CDIR: str = osp.expanduser("~")
VIDEO_PATH: str = osp.join(CDIR, "video.mp4")

if __name__ == "__main__":
    start = timer()
    transform = Preprocess_UniSal(out_size=(224, 384))
    params = {
        "mobilnet_weights": "path/to/mobilnet_v2.pth.tar",
        "decoder_weights": "path/to/weights.pth",
        "sequence_length": 6,
    }
    model = UniSal(params)
    video_capture = load_video(VIDEO_PATH)
    output: Optional[cv2.VideoWriter] = None

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if EXPORT_RESULT is True:
        filename: str = osp.basename(VIDEO_PATH)
        export_path: str = osp.join(CDIR, f"result_{filename}")
        print(f"Exporting to {export_path}")  # noqa
        output = cv2.VideoWriter(
            osp.join(CDIR, f"result_{filename}"),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (width, height),
        )

    # To store frames for batch prediction
    frame_buffer = []
    batch_size = 6

    for frame_index in tqdm(range(frame_count), total=frame_count):
        ret, frame = video_capture.read()

        # Break the loop if there are no more frames
        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_t = transform(frame).numpy()

        frame_buffer.append(frame_t)

        if len(frame_buffer) == batch_size:
            imgs = np.stack(frame_buffer, axis=0)
            predictions = model.run(imgs, {})

            if EXPORT_RESULT is True and output is not None:
                for pred in predictions:
                    salmap = cv2.resize(pred, (width, height))
                    tmp_salmap = salmap.astype("uint8")
                    frame_raw_salmap_output = np.dstack(
                        [tmp_salmap, tmp_salmap, tmp_salmap]
                    )
                    output.write(frame_raw_salmap_output)

            # Clear the frame buffer for the next batch
            frame_buffer.clear()

    # If there are any remaining frames, we process them
    if len(frame_buffer) > 0:
        imgs = np.stack(frame_buffer, axis=0)
        predictions = model.run(imgs, {})

        if EXPORT_RESULT is True and output is not None:
            for pred in predictions:
                salmap = cv2.resize(pred, (width, height))
                tmp_salmap = salmap.astype("uint8")
                frame_raw_salmap_output = np.dstack(
                    [tmp_salmap, tmp_salmap, tmp_salmap]
                )
                output.write(frame_raw_salmap_output)

    print(f"{timer() - start:.2f}s")  # noqa

    # Release the video capture object
    release_video(video_capture)
    release_video(output)
