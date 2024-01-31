""" Two Stream Network saliency model.
"""

import os
from typing import List

import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv3D,
    TimeDistributed,
    UpSampling2D,
)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.vgg16 import VGG16

from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Metric.Saliency import EPSILON
from SalScan.Utils import get_logger

logger = get_logger(__name__)


class Two_Stream_Network(AbstractModel):
    """
    Spatial-Temporal Two-Stream Network for video saliency prediction.
    This model generates saliency maps from video frames using a 7 frames context window.
    This model is Many-to-Many.

    Parameters:
        options (dict): Configuration options for the model, must include `shape_r` and
                        `shape_c` which determine input and output shapes.

    Attributes:
        name (str): The name of the model, 'Two_Stream_Network'.
        type (str): The type of the model, 'video_saliency'.
        wrapper (str): The wrapper type, 'python'.
        params_to_store (list): List of attributes of the model stored in the
                                evaluation logs by
                                `SalScan.Evaluate.ImageSaliencyEvaluation` and
                                `SalScan.Evaluate.VideoSaliencyEvaluation`. It includes
                                `shape_r` and `shape_c`.
        shape_r (int): Height of the input stimuli.
        shape_c (int): Width of the input stimuli.
        shape_r_out (int): Height of the output stimuli, corresponds to `int(shape_r) / 8`
        shape_c_out (int): Width of the output stimuli, corresponds to `int(shape_c) / 8`
        nb_gaussian (int): Number of gaussians used to create the gaussians priors. Set
                            to 8.
        X_cb_st: Gaussian prior used in the static stream.
        X_cb_dy: Gaussian prior used in the dynamic stream.
        model: Two Stream Network Tensorflow model.
    """

    name = "Two_Stream_Network"
    type = "video_saliency"
    wrapper = "python"
    params_to_store = ["shape_r", "shape_c"]

    def __init__(self, options):
        super().__init__()

        files_weights = ["zk-twos-final-model.h5", "zk-twos-sf-model.h5"]
        for filename in files_weights:
            if (
                os.path.exists(os.path.join("SalScan", "Model", "Weights", filename))
                is False
            ):
                raise FileNotFoundError(
                    f"\n ❌ {filename} not present inside {os.path.join('SalScan', 'Model', 'Weights')}. You have to download the two "
                    "models \nfrom https://github.com/zhangkao/IIP_TwoS_Saliency#pre-trained-models and insert them inside the folder."
                )

        self._config = {**options}
        self.shape_r = self._config.get("shape_r", 480)
        self.shape_c = self._config.get("shape_c", 640)
        self.shape_r_out = int(self.shape_r / 8)
        self.shape_c_out = int(self.shape_c / 8)
        self.nb_gaussian = 8
        self.X_cb_st = self.get_guasspriors_3d(
            "st", 1, 7, self.shape_r_out, self.shape_c_out, self.nb_gaussian
        )
        self.X_cb_dy = self.get_guasspriors_3d(
            "dy", 1, 7, self.shape_r_out, self.shape_c_out, self.nb_gaussian
        )

        self.model = self.salcnn_TwoS_Net(
            time_dims=7,
            img_cols=self.shape_c,
            img_rows=self.shape_r,
            img_channels=3,
            pre_sf_path=os.path.join(
                "SalScan", "Model", "Weights", "zk-twos-final-model.h5"
            ),
        )

    def run(self, imgs: np.array, params) -> List[np.ndarray]:
        """
        Generates the saliency maps.

        Args:
            img (np.array): The input array which is a stack of 7 frames.
            params (dict): model's run parameters
        """
        imgs = imgs[None, ...]
        inputs = [imgs, self.X_cb_st, self.X_cb_dy]
        saliency_maps = self.model.predict(x=inputs, batch_size=1, verbose=0)
        # subset salmap trained to minize KLD (considered by the authors the best metric)
        saliency_maps = saliency_maps[0].reshape(7, self.shape_r_out, self.shape_c_out, 1)

        pp_saliency_maps = []
        for saliency_map in saliency_maps:
            saliency_map = self.postprocess_predictions(
                saliency_map, self.shape_r, self.shape_c
            )
            pp_saliency_maps.append(saliency_map)
        return pp_saliency_maps

    def salcnn_TwoS_Net(
        self, time_dims=7, img_rows=480, img_cols=640, img_channels=3, pre_sf_path=""
    ):
        video_inputs = Input(
            shape=(time_dims, img_rows, img_cols, img_channels), name="video_input"
        )
        # Subtract the mean value for static stream
        x_st_input = TimeDistributed(
            Conv2D(
                3,
                (1, 1),
                padding="same",
                kernel_initializer=Constant(value=(1, 0, 0, 0, 1, 0, 0, 0, 1)),
                bias_initializer=Constant(value=(-103.939, -116.779, -123.68)),
            ),
            name="sal_st_sub_mean",
        )(video_inputs)

        # BGR to gray image, three channels
        # Subtract the mean value for dynamic stream
        x_dy_input = TimeDistributed(
            Conv2D(
                3,
                (1, 1),
                padding="same",
                kernel_initializer=Constant(
                    value=(
                        0.114,
                        0.114,
                        0.114,
                        0.587,
                        0.587,
                        0.587,
                        0.299,
                        0.299,
                        0.299,
                    )
                ),
                use_bias=False,
            ),
            name="sal_dy_bgr2gray",
        )(video_inputs)
        x_dy_input = TimeDistributed(
            Conv2D(
                3,
                (1, 1),
                padding="same",
                kernel_initializer=Constant(value=(1, 0, 0, 0, 1, 0, 0, 0, 1)),
                bias_initializer=Constant(value=(-103.939, -116.779, -123.68)),
            ),
            name="sal_dy_sub_mean",
        )(x_dy_input)

        # SF-Net model
        sfnet = self.salcnn_SF_Net(
            img_rows=img_rows, img_cols=img_cols, img_channels=img_channels
        )

        pretrained_ssn_net_path = os.path.join(
            "SalScan", "Model", "Weights", "zk-twos-sf-model.h5"
        )
        sfnet.load_weights(os.path.join(pretrained_ssn_net_path), by_name=True)

        x_sf_st = TimeDistributed(sfnet, name="sf_net_st")(x_st_input)
        x_sf_dy = TimeDistributed(sfnet, name="sf_net_dy")(x_dy_input)

        # St-net model
        x_st = TimeDistributed(
            Conv2D(
                256, (3, 3), activation="relu", padding="same", name="sal_st_conv2d_1"
            ),
            name="sal_st_conv2d_11",
        )(x_sf_st)
        x_st = TimeDistributed(
            Conv2D(
                256, (3, 3), activation="relu", padding="same", name="sal_st_conv2d_2"
            ),
            name="sal_st_conv2d_22",
        )(x_st)

        # Dy_net model
        x_dy = Conv3D(
            256, (3, 3, 3), activation="relu", padding="same", name="sal_dy_conv3d_1"
        )(x_sf_dy)
        x_dy = Conv3D(
            256, (3, 3, 3), activation="relu", padding="same", name="sal_dy_conv3d_2"
        )(x_dy)

        # CGP layer
        cb_inputs_st = Input(
            shape=(time_dims, self.shape_r_out, self.shape_c_out, self.nb_gaussian),
            name="cb_input_st",
        )
        cb_x_st = TimeDistributed(
            Conv2D(
                64, (3, 3), activation="relu", padding="same", name="sal_st_cb_conv2d_1"
            ),
            name="sal_st_cb_conv2d_11",
        )(cb_inputs_st)
        priors_st = TimeDistributed(
            Conv2D(
                64, (3, 3), activation="relu", padding="same", name="sal_st_cb_conv2d_2"
            ),
            name="sal_st_cb_conv2d_22",
        )(cb_x_st)

        cb_inputs_dy = Input(
            shape=(time_dims, self.shape_r_out, self.shape_c_out, self.nb_gaussian),
            name="cb_input_dy",
        )
        cb_x_dy = TimeDistributed(
            Conv2D(
                64, (3, 3), activation="relu", padding="same", name="sal_dy_cb_conv2d_1"
            ),
            name="sal_dy_cb_conv2d_11",
        )(cb_inputs_dy)
        priors_dy = TimeDistributed(
            Conv2D(
                64, (3, 3), activation="relu", padding="same", name="sal_dy_cb_conv2d_2"
            ),
            name="sal_dy_cb_conv2d_22",
        )(cb_x_dy)

        x_st = layers.concatenate([x_st, priors_st], axis=-1, name="sal_st_cb_cat")
        x_dy = layers.concatenate([x_dy, priors_dy], axis=-1, name="sal_dy_cb_cat")

        x_input = [video_inputs, cb_inputs_st, cb_inputs_dy]

        x_st = TimeDistributed(
            Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                name="sal_st_conv2d_3_cb",
            ),
            name="sal_st_conv2d_33_cb",
        )(x_st)
        x_st_out = TimeDistributed(
            Conv2D(1, (3, 3), activation="relu", padding="same", name="sal_st_conv2d_4"),
            name="sal_st_conv2d_44",
        )(x_st)

        x_dy = Conv3D(
            256, (3, 3, 3), activation="relu", padding="same", name="sal_dy_conv3d_3_cb"
        )(x_dy)
        x_dy_out = Conv3D(
            1, (3, 3, 3), activation="relu", padding="same", name="sal_dy_conv3d_4"
        )(x_dy)

        # Fu_net model
        x_fu = layers.concatenate([x_st_out, x_dy_out], axis=-1, name="funet_cat")

        x_fu = TimeDistributed(
            Conv2D(64, (3, 3), activation="relu", padding="same", name="sal_fu_conv2d_1"),
            name="sal_fu_conv2d_1",
        )(x_fu)
        x_fu = TimeDistributed(
            Conv2D(
                128, (3, 3), activation="relu", padding="same", name="sal_fu_conv2d_2"
            ),
            name="sal_fu_conv2d_2",
        )(x_fu)
        x_fu_out = TimeDistributed(
            Conv2D(1, (3, 3), activation="relu", padding="same", name="sal_fu_conv2d_3"),
            name="sal_fu_conv2d_3",
        )(x_fu)

        model = Model(
            inputs=x_input,
            outputs=[x_fu_out, x_fu_out, x_fu_out],
            name="salcnn_fu_model",
        )
        sfnet.trainable = False

        for layer in model.layers[:4]:
            layer.trainable = False

        return model

    def salcnn_SF_Net(self, img_rows=480, img_cols=640, img_channels=3):
        sal_input = Input(shape=(img_rows, img_cols, img_channels))
        input_shape = (img_rows, img_cols, img_channels)

        # cnn = salcnn_VGG16(include_top=False, weights='imagenet', input_tensor=sal_input, input_shape=input_shape)
        cnn = VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=sal_input,
            input_shape=input_shape,
        )

        # C2 = cnn.get_layer(name='block2_pool').output
        C3 = cnn.get_layer(name="block3_pool").output
        C4 = cnn.get_layer(name="block4_pool").output
        C5 = cnn.get_layer(name="block5_conv3").output

        # C2_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c2')(C2)
        C3_1 = Conv2D(256, (1, 1), activation="relu", padding="same", name="sal_fpn_c3")(
            C3
        )
        C4_1 = Conv2D(256, (1, 1), activation="relu", padding="same", name="sal_fpn_c4")(
            C4
        )
        C5_1 = Conv2D(256, (1, 1), activation="relu", padding="same", name="sal_fpn_c5")(
            C5
        )

        C5_1_up = UpSampling2D((2, 2), interpolation="bilinear", name="sal_fpn_p5_up")(
            C5_1
        )
        C4_1_up = UpSampling2D((2, 2), interpolation="bilinear", name="sal_fpn_p4_up")(
            C4_1
        )
        x = layers.concatenate(
            [C3_1, C4_1_up, C5_1_up], axis=-1, name="sal_fpn_merge_concat"
        )
        model = Model(inputs=[sal_input], outputs=[x], name="salcnn_sf_fpn")
        return model

    def get_guasspriors_3d(
        self, type="st", b_s=2, time_dims=7, shape_r=60, shape_c=80, channels=8
    ):
        if type == "dy":
            ims = self.dy_get_gaussmaps(shape_r, shape_c, channels)
        else:
            ims = self.st_get_gaussmaps(shape_r, shape_c, channels)

        ims = np.expand_dims(ims, axis=0)
        ims = np.repeat(ims, time_dims, axis=0)

        ims = np.expand_dims(ims, axis=0)
        ims = np.repeat(ims, b_s, axis=0)
        return ims

    def st_get_gaussmaps(self, height, width, nb_gaussian):
        e = height / width
        e1 = (1 - e) / 2
        e2 = e1 + e

        mu_x = np.repeat(0.5, nb_gaussian, 0)
        mu_y = np.repeat(0.5, nb_gaussian, 0)

        sigma_x = e * np.array(np.arange(1, 9)) / 16
        sigma_y = sigma_x

        x_t = np.dot(
            np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width))
        )
        y_t = np.dot(
            np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width))
        )

        x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
        y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

        gaussian = (
            1
            / (2 * np.pi * sigma_x * sigma_y + EPSILON)
            * np.exp(
                -(
                    (x_t - mu_x) ** 2 / (2 * sigma_x**2 + EPSILON)
                    + (y_t - mu_y) ** 2 / (2 * sigma_y**2 + EPSILON)
                )
            )
        )

        return gaussian

    def dy_get_gaussmaps(self, height, width, nb_gaussian):
        e = height / width
        e1 = (1 - e) / 2
        e2 = e1 + e

        mu_x = np.repeat(0.5, nb_gaussian, 0)
        mu_y = np.repeat(0.5, nb_gaussian, 0)

        sigma_x = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 1 / 2, 1 / 2])
        sigma_y = e * np.array([1 / 16, 1 / 8, 3 / 16, 1 / 4, 1 / 8, 1 / 4, 3 / 8, 1 / 2])

        x_t = np.dot(
            np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width))
        )
        y_t = np.dot(
            np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width))
        )

        x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
        y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

        gaussian = (
            1
            / (2 * np.pi * sigma_x * sigma_y + EPSILON)
            * np.exp(
                -(
                    (x_t - mu_x) ** 2 / (2 * sigma_x**2 + EPSILON)
                    + (y_t - mu_y) ** 2 / (2 * sigma_y**2 + EPSILON)
                )
            )
        )

        return gaussian

    def postprocess_predictions(self, pred, shape_r, shape_c):
        predictions_shape = pred.shape
        rows_rate = shape_r / predictions_shape[0]
        cols_rate = shape_c / predictions_shape[1]

        if rows_rate > cols_rate:
            new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
            pred = cv2.resize(pred, (new_cols, shape_r))
            img = pred[
                :,
                ((pred.shape[1] - shape_c) // 2) : (
                    (pred.shape[1] - shape_c) // 2 + shape_c
                ),
            ]
        else:
            new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
            pred = cv2.resize(pred, (shape_c, new_rows))
            img = pred[
                ((pred.shape[0] - shape_r) // 2) : (
                    (pred.shape[0] - shape_r) // 2 + shape_r
                ),
                :,
            ]

        return img / np.max(img) * 255
