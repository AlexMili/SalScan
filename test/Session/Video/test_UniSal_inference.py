"""
Time to evaluate all frames: 351 secs
"""


from timeit import default_timer as timer

import cv2
import numpy as np

from SalScan.Model.Saliency.UniSal import UniSal
from SalScan.Transforms import Preprocess_UniSal

VIDEO_PATH = "video.mp4"

if __name__ == "__main__":
    start = timer()
    transform = Preprocess_UniSal(out_size=(224, 384))
    params = {
        "mobilnet_weights": "unisal/unisal/models/weights/mobilenet_v2.pth.tar",
        "decoder_weights": "unisal/training_runs/pretrained_unisal/weights_best.pth",
        "sequence_length": 6,
    }
    model = UniSal(params)
    video_capture = cv2.VideoCapture(VIDEO_PATH)

    frame_count = 0
    frame_buffer = []  # To store frames for batch prediction
    batch_size = 6

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break  # Break the loop if there are no more frames

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame).numpy()

        frame_buffer.append(frame)
        frame_count += 1

        if len(frame_buffer) == batch_size:
            # Perform prediction on the batch of 6 frames
            # Replace 'predict_batch' with your actual prediction code

            imgs = np.stack(frame_buffer, axis=0)
            predictions = model.run(imgs, {})

            # Process the predictions as needed
            # ...

            # Clear the frame buffer for the next batch
            frame_buffer.clear()
            print(frame_count)

    # If there are any remaining frames (less than 6), process them
    if frame_buffer:
        # Perform prediction on the remaining frames
        # Replace 'predict_batch' with your actual prediction code
        imgs = np.stack(frame_buffer, axis=0)
        predictions = model.run(imgs, {})

    print(timer() - start)

    # Release the video capture object
    video_capture.release()
    cv2.destroyAllWindows()
