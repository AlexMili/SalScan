"""
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
"""
import math
import os
import pprint
from collections import OrderedDict
from itertools import product
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.nn import init
from torch.nn.parameter import Parameter

from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Utils import normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UniSal(AbstractModel):
    """
    Unified Image and Video Saliency Modeling (UniSal) for video saliency prediction.
    This model generates saliency maps from video frames using a configurable
    context-window. The model is a Many-to-Many.

    Attributes:
        options (dict): Configuration options for the model, must include
                        `mobilnet_weights` and `decoder_weights`

    Attributes:
        name (str): The name of the model, 'UniSal'.
        type (str): The type of the model, 'video_saliency'.
        wrapper (str): The wrapper type, 'python'.
        params_to_store (list): List of attributes of the model stored in the
                                evaluation logs by
                                `SalScan.Evaluate.ImageSaliencyEvaluation` and
                                `SalScan.Evaluate.VideoSaliencyEvaluation`. Includes
                                `mobilnet_weights`, `decoder_weights`, and
                                `sequence_length`.
        model_cfg (dict): Configuration for the model sources and weights.
        device (bool): Device used to perform the prediction. Can be either `cpu` or `gpu`
                        depending on `torch.cuda.is_available()`.
        model: UniSal Pytorch model.
    """

    name = "UniSal"
    type = "video_saliency"
    wrapper = "python"
    params_to_store = ["mobilnet_weights", "decoder_weights", "sequence_length"]
    model_cfg = {"sources": ["DHF1K", "Hollywood", "UCFSports", "SALICON"]}

    def __init__(self, options):
        super().__init__()

        self._config = {**options}
        # weight are in unisal/training_runs/pretrained_unisal/weights_best.pth
        mobilnet_weights = self._config.get("mobilnet_weights", "")
        decoder_weights = self._config.get("decoder_weights", "")

        if os.path.exists(mobilnet_weights) is False:
            raise FileNotFoundError(
                "\n ❌ 'mobilnet_weights' either an empty or non valid directory."
            )
        if os.path.exists(decoder_weights) is False:
            raise FileNotFoundError(
                "\n ❌ 'decoder_weights' either an empty or non valid directory."
            )

        self.device = DEVICE
        self.model_cfg["mobilnet_weights"] = mobilnet_weights
        self.model = UNISAL(**self.model_cfg).to(self.device)
        self.model.load_state_dict(torch.load(decoder_weights, map_location=self.device))

    def run(self, imgs: np.array, params) -> List[np.ndarray]:
        """
        Generates the saliency maps.

        Args:
            img (np.array): The input stimulus.
            params (dict): model's run parameters
        """
        imgs = torch.from_numpy(imgs)[None, ...].permute(0, 1, 2, 3, 4)
        imgs = imgs.to(self.device)
        salmaps = self.model(imgs)
        # size (1, seq_len, 1, heigth, width) -> (6, 224, 384) -> list
        out = []
        for smap in torch.unbind(salmaps, dim=1):
            # Postprocess prediction
            smap = smap.exp()
            smap = torch.squeeze(smap)
            smap = smap.detach().cpu().numpy()
            smap = normalize(smap, method="range") * 255
            smap = smap.astype(np.uint8)
            out.append(smap)

        return out


# Set default backbone CNN kwargs
default_cnn_cfg = {
    "widen_factor": 1.0,
    "pretrained": True,
    "input_channel": 32,
    "last_channel": 1280,
}

# Set default RNN kwargs
default_rnn_cfg = {
    "kernel_size": (3, 3),
    "gate_ksize": (3, 3),
    "dropout": (False, True, False),
    "drop_prob": (0.2, 0.2, 0.2),
    "mobile": True,
}


class DomainBatchNorm2d(nn.Module):
    """
    Domain-specific 2D BatchNorm module.

    Stores a BN module for a given list of sources.
    During the forward pass, select the BN module based on self.this_source.
    """

    def __init__(self, num_features, sources, momenta=None, **kwargs):
        """
        num_features: Number of channels
        sources: List of sources
        momenta: List of BatchNorm momenta corresponding to the sources.
            Default is 0.1 for each source.
        kwargs: Other BatchNorm kwargs
        """
        super().__init__()
        self.sources = sources

        # Process momenta input
        if momenta is None:
            momenta = [0.1] * len(sources)
        self.momenta = momenta
        if "momentum" in kwargs:
            del kwargs["momentum"]

        # Instantiate the BN modules
        for src, mnt in zip(sources, self.momenta):
            self.__setattr__(
                f"bn_{src}", nn.BatchNorm2d(num_features, momentum=mnt, **kwargs)
            )

        # Prepare the self.this_source attribute that will be updated at runtime
        # by the model
        self.this_source = None

    def forward(self, x):
        return self.__getattr__(f"bn_{self.this_source}")(x)


class UNISAL(nn.Module):
    """
    UNISAL model. See paper for more information.

    Arguments:
        rnn_input_channels: Number of channels of the RNN input.
        rnn_hidden_channels: Number of channels of the RNN hidden state.
        cnn_cfg: Dictionary with kwargs for the backbone CNN.
        rnn_cfg: Dictionary with kwargs for the RNN.
        res_rnn: Whether to add the RNN features with a residual connection.
        bypass_rnn: Whether to bypass the RNN for static inputs.
            Requires res_rnn.
        drop_probs: Dropout probabilities for
            [backbone CNN outputs, Skip-2x and Skip-4x].
        gaussian_init: Method to initialize the learned Gaussian parameters.
            If "manual", 16 pre-defined Gaussians are initialized.
        n_gaussians: Number of Gaussians if gaussian_init is "random".
        smoothing_ksize: Size of the Smoothing kernel.
        bn_momentum: Momentum of the BatchNorm running estimates for dynamic
            batches.
        static_bn_momentum: Momentum of the BatchNorm running estimates for
            static batches.
        sources: List of datasets.
        ds_bn: Domain-specific BatchNorm (DSBN).
        ds_adaptation: Domain-specific Adaptation.
        ds_smoothing: Domain-specific Smoothing.
        ds_gaussians: Domain-specific Gaussian prior maps.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        rnn_input_channels=256,
        rnn_hidden_channels=256,
        cnn_cfg=None,
        rnn_cfg=None,
        res_rnn=True,
        bypass_rnn=True,
        drop_probs=(0.0, 0.6, 0.6),
        gaussian_init="manual",
        n_gaussians=16,
        smoothing_ksize=41,
        bn_momentum=0.01,
        static_bn_momentum=0.1,
        sources=("DHF1K", "Hollywood", "UCFSports", "SALICON"),
        ds_bn=True,
        ds_adaptation=True,
        ds_smoothing=True,
        ds_gaussians=True,
        verbose=1,
        mobilnet_weights: str = None,
    ):
        super().__init__()

        # Check inputs
        if gaussian_init not in ("random", "manual"):
            raise ValueError("gaussian_init's value is not recognized")
        # Bypass-RNN requires residual RNN connection
        if bypass_rnn is True and res_rnn is False:
            raise ValueError("bypass_rnn cannot be True and res_rnn False")

        # Manual Gaussian initialization generates 16 Gaussians
        if n_gaussians > 0 and gaussian_init == "manual":
            n_gaussians = 16

        self.rnn_input_channels = rnn_input_channels
        self.rnn_hidden_channels = rnn_hidden_channels
        this_cnn_cfg = default_cnn_cfg.copy()
        this_cnn_cfg.update(cnn_cfg or {})
        self.cnn_cfg = this_cnn_cfg
        this_rnn_cfg = default_rnn_cfg.copy()
        this_rnn_cfg.update(rnn_cfg or {})
        self.rnn_cfg = this_rnn_cfg
        self.bypass_rnn = bypass_rnn
        self.res_rnn = res_rnn
        self.drop_probs = drop_probs
        self.gaussian_init = gaussian_init
        self.n_gaussians = n_gaussians
        self.smoothing_ksize = smoothing_ksize
        self.bn_momentum = bn_momentum
        self.sources = sources
        self.ds_bn = ds_bn
        self.static_bn_momentum = static_bn_momentum
        self.ds_adaptation = ds_adaptation
        self.ds_smoothing = ds_smoothing
        self.ds_gaussians = ds_gaussians
        self.verbose = verbose

        self.cnn_cfg["mobilnet_weights"] = mobilnet_weights

        # Initialize backbone CNN
        self.cnn = MobileNetV2(**self.cnn_cfg)

        # Initialize Post-CNN module with optional dropout
        post_cnn = [
            (
                "inv_res",
                InvertedResidual(
                    self.cnn.out_channels + n_gaussians,
                    rnn_input_channels,
                    1,
                    1,
                    bn_momentum=bn_momentum,
                ),
            )
        ]
        if self.drop_probs[0] > 0:
            post_cnn.insert(
                0, ("dropout", nn.Dropout2d(self.drop_probs[0], inplace=False))
            )
        self.post_cnn = nn.Sequential(OrderedDict(post_cnn))

        # Initialize Bypass-RNN if training on dynamic data
        if sources != ("SALICON",) or not self.bypass_rnn:
            self.rnn = ConvGRU(
                rnn_input_channels,
                hidden_channels=[rnn_hidden_channels],
                batchnorm=self.get_bn_module,
                **self.rnn_cfg,
            )
            self.post_rnn = self.conv_1x1_bn(rnn_hidden_channels, rnn_input_channels)

        # Initialize first upsampling module US1
        self.upsampling_1 = nn.Sequential(
            OrderedDict(
                [
                    ("us1", self.upsampling(2)),
                ]
            )
        )

        # Number of channels at the 2x scale
        channels_2x = 128

        # Initialize Skip-2x module
        self.skip_2x = self.make_skip_connection(
            self.cnn.feat_2x_channels, channels_2x, 2, self.drop_probs[1]
        )

        # Initialize second upsampling module US2
        self.upsampling_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "inv_res",
                        InvertedResidual(
                            rnn_input_channels + channels_2x,
                            channels_2x,
                            1,
                            2,
                            batchnorm=self.get_bn_module,
                        ),
                    ),
                    ("us2", self.upsampling(2)),
                ]
            )
        )

        # Number of channels at the 4x scale
        channels_4x = 64

        # Initialize Skip-4x module
        self.skip_4x = self.make_skip_connection(
            self.cnn.feat_4x_channels, channels_4x, 2, self.drop_probs[2]
        )

        # Initialize Post-US2 module
        self.post_upsampling_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "inv_res",
                        InvertedResidual(
                            channels_2x + channels_4x,
                            channels_4x,
                            1,
                            2,
                            batchnorm=self.get_bn_module,
                        ),
                    ),
                ]
            )
        )

        # Initialize domain-specific modules
        for source_str in self.sources:
            source_str = f"_{source_str}".lower()

            # Initialize learned Gaussian priors parameters
            if n_gaussians > 0:
                self.set_gaussians(source_str)

            # Initialize Adaptation
            self.__setattr__(
                "adaptation" + (source_str if self.ds_adaptation else ""),
                nn.Sequential(*[nn.Conv2d(channels_4x, 1, 1, bias=True)]),
            )

            # Initialize Smoothing
            smoothing = nn.Conv2d(
                1, 1, kernel_size=smoothing_ksize, padding=0, bias=False
            )
            with torch.no_grad():
                gaussian = self._make_gaussian_maps(
                    smoothing.weight.data, torch.Tensor([[[0.5, -2]] * 2])
                )
                gaussian /= gaussian.sum()
                smoothing.weight.data = gaussian
            self.__setattr__(
                "smoothing" + (source_str if self.ds_smoothing else ""), smoothing
            )

        if self.verbose > 1:
            pprint.pprint(self.asdict(), width=1)  # noqa

    @property
    def this_source(self):
        """Return current source for domain-specific BatchNorm."""
        return self._this_source

    @this_source.setter
    def this_source(self, source):
        """Set current source for domain-specific BatchNorm."""
        for module in self.modules():
            if isinstance(module, DomainBatchNorm2d):
                module.this_source = source
        self._this_source = source

    def get_bn_module(self, num_features, **kwargs):
        """Return BatchNorm class (domain-specific or domain-invariant)."""
        momenta = [
            self.bn_momentum if src != "SALICON" else self.static_bn_momentum
            for src in self.sources
        ]
        if self.ds_bn:
            return DomainBatchNorm2d(
                num_features, self.sources, momenta=momenta, **kwargs
            )
        else:
            return nn.BatchNorm2d(num_features, **kwargs)

    # @staticmethod
    def upsampling(self, factor):
        """Return upsampling module."""
        return nn.Sequential(
            *[
                nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=False),
            ]
        )

    def set_gaussians(self, source_str, prefix="coarse_"):
        """Set Gaussian parameters."""
        suffix = source_str if self.ds_gaussians else ""
        self.__setattr__(
            prefix + "gaussians" + suffix, self._initialize_gaussians(self.n_gaussians)
        )

    def _initialize_gaussians(self, n_gaussians):
        """
        Return initialized Gaussian parameters.
        Dimensions: [idx, y/x, mu/logstd].
        """
        if self.gaussian_init == "manual":
            gaussians = torch.Tensor(
                [
                    list(product([0.25, 0.5, 0.75], repeat=2))
                    + [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)]
                    + [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)]
                    + [(0.5, 0.5)],
                    [(-1.5, -1.5)] * 9 + [(0, -1.5)] * 3 + [(-1.5, 0)] * 3 + [(0, 0)],
                ]
            ).permute(1, 2, 0)

        elif self.gaussian_init == "random":
            with torch.no_grad():
                gaussians = torch.stack(
                    [
                        torch.randn(n_gaussians, 2, dtype=torch.float) * 0.1 + 0.5,
                        torch.randn(n_gaussians, 2, dtype=torch.float) * 0.2 - 1,
                    ],
                    dim=2,
                )

        else:
            raise NotImplementedError

        gaussians = nn.Parameter(gaussians, requires_grad=True)
        return gaussians

    @staticmethod
    def _make_gaussian_maps(x, gaussians, size=None, scaling=6.0):
        """Construct prior maps from Gaussian parameters."""
        if size is None:
            size = x.shape[-2:]
            bs = x.shape[0]
        else:
            size = [size] * 2
            bs = 1
        dtype = x.dtype
        device = x.device

        gaussian_maps = []
        map_template = torch.ones(*size, dtype=dtype, device=device)
        meshgrids = torch.meshgrid(
            [
                torch.linspace(0, 1, size[0], dtype=dtype, device=device),
                torch.linspace(0, 1, size[1], dtype=dtype, device=device),
            ]
        )

        for gaussian_idx, yx_mu_logstd in enumerate(torch.unbind(gaussians)):
            map = map_template.clone()
            for mu_logstd, mgrid in zip(yx_mu_logstd, meshgrids):
                mu = mu_logstd[0]
                std = torch.exp(mu_logstd[1])
                map *= torch.exp(-(((mgrid - mu) / std) ** 2) / 2)

            map *= scaling
            gaussian_maps.append(map)

        gaussian_maps = torch.stack(gaussian_maps)
        gaussian_maps = gaussian_maps.unsqueeze(0).expand(bs, -1, -1, -1)
        return gaussian_maps

    def _get_gaussian_maps(self, x, source_str, prefix="coarse_", **kwargs):
        """Return the constructed Gaussian prior maps."""
        suffix = source_str if self.ds_gaussians else ""
        gaussians = self.__getattr__(prefix + "gaussians" + suffix)
        gaussian_maps = self._make_gaussian_maps(x, gaussians, **kwargs)
        return gaussian_maps

    # @classmethod
    def make_skip_connection(
        self, input_channels, output_channels, expand_ratio, p, inplace=False
    ):
        """Return skip connection module."""
        hidden_channels = round(input_channels * expand_ratio)
        return nn.Sequential(
            OrderedDict(
                [
                    ("expansion", self.conv_1x1_bn(input_channels, hidden_channels)),
                    ("dropout", nn.Dropout2d(p, inplace=inplace)),
                    (
                        "reduction",
                        nn.Sequential(
                            *[
                                nn.Conv2d(hidden_channels, output_channels, 1),
                                self.get_bn_module(output_channels),
                            ]
                        ),
                    ),
                ]
            )
        )

    # @staticmethod
    def conv_1x1_bn(self, inp, oup):
        """Return pointwise convolution with BatchNorm and ReLU6."""
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            self.get_bn_module(oup),
            nn.ReLU6(inplace=True),
        )

    def forward(
        self,
        x,
        target_size=None,
        h0=None,
        return_hidden=False,
        source="DHF1K",
        static=None,
    ):
        """
        Forward pass.

        Arguments:
            x: Input batch of dimensions [batch, time, channel, h, w].
            target_size: (height, width) of the resized output.
            h0: Initial hidden state.
            return_hidden: Return [prediction, hidden_state].
            source: Data source of current batch. Must be in self.sources.
            static: Whether the current input is static. If None, this is
                inferred from the input dimensions or self.sources.
        """
        if target_size is None:
            target_size = x.shape[-2:]

        # Set the current source for the domain-specific BatchNorm modules
        self.this_source = source

        # Prepare other parameters
        source_str = f"_{source.lower()}"
        if static is None:
            static = x.shape[1] == 1 or self.sources == ("SALICON",)

        # Compute backbone CNN features and concatenate with Gaussian prior maps
        feat_seq_1x = []
        feat_seq_2x = []
        feat_seq_4x = []
        for t, img in enumerate(torch.unbind(x, dim=1)):
            im_feat_1x, im_feat_2x, im_feat_4x = self.cnn(img)

            im_feat_2x = self.skip_2x(im_feat_2x)
            im_feat_4x = self.skip_4x(im_feat_4x)

            if self.n_gaussians > 0:
                gaussian_maps = self._get_gaussian_maps(im_feat_1x, source_str)
                im_feat_1x = torch.cat((im_feat_1x, gaussian_maps), dim=1)

            im_feat_1x = self.post_cnn(im_feat_1x)
            feat_seq_1x.append(im_feat_1x)
            feat_seq_2x.append(im_feat_2x)
            feat_seq_4x.append(im_feat_4x)

        feat_seq_1x = torch.stack(feat_seq_1x, dim=1)

        # Bypass-RNN
        hidden, rnn_feat_seq, rnn_feat = (None,) * 3
        if not (static and self.bypass_rnn):
            rnn_feat_seq, hidden = self.rnn(feat_seq_1x, hidden=h0)

        # Decoder
        output_seq = []
        for idx, im_feat in enumerate(torch.unbind(feat_seq_1x, dim=1)):
            if not (static and self.bypass_rnn):
                rnn_feat = rnn_feat_seq[:, idx, ...]
                rnn_feat = self.post_rnn(rnn_feat)
                if self.res_rnn:
                    im_feat = im_feat + rnn_feat
                else:
                    im_feat = rnn_feat

            im_feat = self.upsampling_1(im_feat)
            im_feat = torch.cat((im_feat, feat_seq_2x[idx]), dim=1)
            im_feat = self.upsampling_2(im_feat)
            im_feat = torch.cat((im_feat, feat_seq_4x[idx]), dim=1)
            im_feat = self.post_upsampling_2(im_feat)

            im_feat = self.__getattr__(
                "adaptation" + (source_str if self.ds_adaptation else "")
            )(im_feat)

            im_feat = F.interpolate(im_feat, size=x.shape[-2:], mode="nearest")

            im_feat = F.pad(im_feat, [self.smoothing_ksize // 2] * 4, mode="replicate")
            im_feat = self.__getattr__(
                "smoothing" + (source_str if self.ds_smoothing else "")
            )(im_feat)

            im_feat = F.interpolate(
                im_feat, size=target_size, mode="bilinear", align_corners=False
            )

            im_feat = log_softmax(im_feat)
            output_seq.append(im_feat)
        output_seq = torch.stack(output_seq, dim=1)

        outputs = [output_seq]
        if return_hidden:
            outputs.append(hidden)
        if len(outputs) == 1:
            return outputs[0]
        return outputs


# Source: https://github.com/tonylins/pytorch-mobilenet-v2


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expand_ratio,
        omit_stride=False,
        no_res_connect=False,
        dropout=0.0,
        bn_momentum=0.1,
        batchnorm=None,
    ):
        super().__init__()
        self.out_channels = oup
        self.stride = stride
        self.omit_stride = omit_stride
        self.use_res_connect = not no_res_connect and self.stride == 1 and inp == oup
        self.dropout = dropout
        actual_stride = self.stride if not self.omit_stride else 1
        if batchnorm is None:

            def batchnorm(num_features):
                return nn.BatchNorm2d(num_features, momentum=bn_momentum)

        if actual_stride not in [1, 2]:
            raise ValueError("actual_stride must be 1 or 2")

        hidden_dim = round(inp * expand_ratio)
        if expand_ratio == 1:
            modules = [
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    actual_stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.append(nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
        else:
            modules = [
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    actual_stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                batchnorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                batchnorm(oup),
            ]
            if self.dropout > 0:
                modules.insert(3, nn.Dropout2d(self.dropout))
            self.conv = nn.Sequential(*modules)
            self._initialize_weights()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2(nn.Module):
    def __init__(
        self,
        widen_factor=1.0,
        pretrained=True,
        last_channel=None,
        input_channel=32,
        mobilnet_weights=None,
    ):
        super().__init__()
        self.widen_factor = widen_factor
        self.pretrained = pretrained
        self.last_channel = last_channel
        self.input_channel = input_channel

        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(self.input_channel * widen_factor)
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * widen_factor)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(
                            input_channel,
                            output_channel,
                            s,
                            expand_ratio=t,
                            omit_stride=True,
                        )
                    )
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t)
                    )
                input_channel = output_channel
        # building last several layers
        if self.last_channel is not None:
            output_channel = (
                int(self.last_channel * widen_factor)
                if widen_factor > 1.0
                else self.last_channel
            )
            self.features.append(conv_1x1_bn(input_channel, output_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.out_channels = output_channel
        self.feat_1x_channels = int(interverted_residual_setting[-1][1] * widen_factor)
        self.feat_2x_channels = int(interverted_residual_setting[-2][1] * widen_factor)
        self.feat_4x_channels = int(interverted_residual_setting[-4][1] * widen_factor)
        self.feat_8x_channels = int(interverted_residual_setting[-5][1] * widen_factor)

        if self.pretrained:
            state_dict = torch.load(
                mobilnet_weights,
                map_location=DEVICE,
            )
            self.load_state_dict(state_dict, strict=False)
        else:
            self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        feat_2x, feat_4x = None, None
        for idx, module in enumerate(self.features._modules.values()):
            x = module(x)
            if idx == 7:
                feat_4x = x.clone()
            elif idx == 14:
                feat_2x = x.clone()
            if idx > 0 and hasattr(module, "stride") and module.stride != 1:
                x = x[..., ::2, ::2]

        return x, feat_2x, feat_4x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# Inspired by:
# https://github.com/jacobkimmel/pytorch_convgru
# https://gist.github.com/halochou/acbd669af86ecb8f988325084ba7a749


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell.


    Arguments:
        input_ch: Number of channels of the input.
        hidden_ch: Number of channels of hidden state.
        kernel_size (tuple): Kernel size of the U and W operations.
        gate_ksize (tuple): Kernel size for the gates.
        bias: Add bias term to layers.
        norm: Normalization method. 'batch', 'instance' or ''.
        norm_momentum: BatchNorm momentum.
        affine_norm: Affine BatchNorm.
        batchnorm: External function that accepts a number of channels and
            returns a BatchNorm module (for DSBN). Overwrites norm and
            norm_momentum.
        drop_prob: Tuple of dropout probabilities for input, recurrent and
            output dropout.
        do_mode: If 'recurrent', the variational dropout is used, dropping out
            the same channels at every time step. If 'naive', different channels
            are dropped at each time step.
        r_bias, z_bias: Bias initialization for r and z gates.
        mobile: If True, MobileNet-style convolutions are used.
    """

    def __init__(
        self,
        input_ch,
        hidden_ch,
        kernel_size,
        gate_ksize=(1, 1),
        bias=True,
        norm="",
        norm_momentum=0.1,
        affine_norm=True,
        batchnorm=None,
        gain=1,
        drop_prob=(0.0, 0.0, 0.0),
        do_mode="recurrent",
        r_bias=0.0,
        z_bias=0.0,
        mobile=False,
        **kwargs,
    ):
        super().__init__()

        self.input_ch = input_ch
        self.hidden_ch = hidden_ch
        self.kernel_size = kernel_size
        self.gate_ksize = gate_ksize
        self.mobile = mobile
        self.kwargs = {"init": "xavier_uniform_"}
        self.kwargs.update(kwargs)

        # Process normalization arguments
        self.norm = norm
        self.norm_momentum = norm_momentum
        self.affine_norm = affine_norm
        self.batchnorm = batchnorm
        self.norm_kwargs = None
        if self.batchnorm is not None:
            self.norm = "batch"
        elif self.norm:
            self.norm_kwargs = {
                "affine": self.affine_norm,
                "track_running_stats": True,
                "momentum": self.norm_momentum,
            }

        # Prepare normalization modules
        if self.norm:
            self.norm_r_x = self.get_norm_module(self.hidden_ch)
            self.norm_r_h = self.get_norm_module(self.hidden_ch)
            self.norm_z_x = self.get_norm_module(self.hidden_ch)
            self.norm_z_h = self.get_norm_module(self.hidden_ch)
            self.norm_out_x = self.get_norm_module(self.hidden_ch)
            self.norm_out_h = self.get_norm_module(self.hidden_ch)

        # Prepare dropout
        self.drop_prob = drop_prob
        self.do_mode = do_mode
        if self.do_mode == "recurrent":
            # Prepare dropout masks if using recurrent dropout
            for idx, mask in self.yield_drop_masks():
                self.register_buffer(self.mask_name(idx), mask.to(DEVICE))
        elif self.do_mode != "naive":
            raise ValueError("Unknown dropout mode ", self.do_mode)

        # Instantiate the main weight matrices
        self.w_r = self._conv2d(self.input_ch, self.gate_ksize, bias=False)
        self.u_r = self._conv2d(self.hidden_ch, self.gate_ksize, bias=False)
        self.w_z = self._conv2d(self.input_ch, self.gate_ksize, bias=False)
        self.u_z = self._conv2d(self.hidden_ch, self.gate_ksize, bias=False)
        self.w = self._conv2d(self.input_ch, self.kernel_size, bias=False)
        self.u = self._conv2d(self.hidden_ch, self.gate_ksize, bias=False)

        # Instantiate the optional biases and affine paramters
        self.bias = bias
        self.r_bias = r_bias
        self.z_bias = z_bias
        if self.bias or self.affine_norm:
            self.b_r = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.b_z = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.b_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
        if self.affine_norm:
            self.a_r_x = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_r_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_z_x = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_z_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_h_x = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_h_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))

        self.gain = gain
        self.set_weights()

    def set_weights(self):
        """Initialize the parameters"""

        def gain_from_ksize(ksize):
            n = ksize[0] * ksize[1] * self.hidden_ch
            return math.sqrt(2.0 / n)

        with torch.no_grad():
            if not self.mobile:
                if self.gain < 0:
                    gain_1 = gain_from_ksize(self.kernel_size)
                    gain_2 = gain_from_ksize(self.gate_ksize)
                else:
                    gain_1 = gain_2 = self.gain
                init_fn = getattr(init, self.kwargs["init"])
                init_fn(self.w_r.weight, gain=gain_2)
                init_fn(self.u_r.weight, gain=gain_2)
                init_fn(self.w_z.weight, gain=gain_2)
                init_fn(self.u_z.weight, gain=gain_2)
                init_fn(self.w.weight, gain=gain_1)
                init_fn(self.u.weight, gain=gain_2)
            if self.bias or self.affine_norm:
                self.b_r.data.fill_(self.r_bias)
                self.b_z.data.fill_(self.z_bias)
                self.b_h.data.zero_()
            if self.affine_norm:
                self.a_r_x.data.fill_(1)
                self.a_r_h.data.fill_(1)
                self.a_z_x.data.fill_(1)
                self.a_z_h.data.fill_(1)
                self.a_h_x.data.fill_(1)
                self.a_h_h.data.fill_(1)

    def forward(self, x, h_tm1):
        # Initialize hidden state if necessary
        if h_tm1 is None:
            h_tm1 = self._init_hidden(x, cuda=x.is_cuda)

        # Compute gate components
        r_x = self.w_r(self.apply_dropout(x, 0, 0))
        r_h = self.u_r(self.apply_dropout(h_tm1, 1, 0))
        z_x = self.w_z(self.apply_dropout(x, 0, 1))
        z_h = self.u_z(self.apply_dropout(h_tm1, 1, 1))
        h_x = self.w(self.apply_dropout(x, 0, 2))
        h_h = self.u(self.apply_dropout(h_tm1, 1, 2))

        if self.norm:
            # Apply normalization
            r_x = self.norm_r_x(r_x)
            r_h = self.norm_r_h(r_h)
            z_x = self.norm_z_x(z_x)
            z_h = self.norm_z_h(z_h)
            h_x = self.norm_out_x(h_x)
            h_h = self.norm_out_h(h_h)

            if self.affine_norm:
                # Apply affine transformation
                r_x = r_x * self.a_r_x
                r_h = r_h * self.a_r_h
                z_x = z_x * self.a_z_x
                z_h = z_h * self.a_z_h
                h_x = h_x * self.a_h_x
                h_h = h_h * self.a_h_h

        # Compute gates with optinal bias
        if self.bias or self.affine_norm:
            r = torch.sigmoid(r_x + r_h + self.b_r)
            z = torch.sigmoid(z_x + z_h + self.b_z)
        else:
            r = torch.sigmoid(r_x + r_h)
            z = torch.sigmoid(z_x + z_h)

        # Compute new hidden state
        if self.bias or self.affine_norm:
            h = torch.tanh(h_x + r * h_h + self.b_h)
        else:
            h = torch.tanh(h_x + r * h_h)
        h = (1 - z) * h_tm1 + z * h

        # Optionally apply output dropout
        y = self.apply_dropout(h, 2, 0)

        return y, h

    @staticmethod
    def mask_name(idx):
        return "drop_mask_{}".format(idx)

    def set_drop_masks(self):
        """Set the dropout masks for the current sequence"""
        for idx, mask in self.yield_drop_masks():
            setattr(self, self.mask_name(idx), mask.to(DEVICE))

    def yield_drop_masks(self):
        """Iterator over recurrent dropout masks"""
        n_masks = (3, 3, 1)
        n_channels = (self.input_ch, self.hidden_ch, self.hidden_ch)
        for idx, p in enumerate(self.drop_prob):
            if p > 0:
                yield (idx, self.generate_do_mask(p, n_masks[idx], n_channels[idx]))

    @staticmethod
    def generate_do_mask(p, n, ch):
        """Generate a dropout mask for recurrent dropout"""
        with torch.no_grad():
            mask = Bernoulli(torch.full((n, ch), 1 - p)).sample() / (1 - p)
            mask = mask.requires_grad_(False).cpu()
            return mask

    def apply_dropout(self, x, idx, sub_idx):
        """Apply recurrent or naive dropout"""
        if self.training and self.drop_prob[idx] > 0 and idx != 2:
            if self.do_mode == "recurrent":
                x = x.clone() * torch.reshape(
                    getattr(self, self.mask_name(idx))[sub_idx, :], (1, -1, 1, 1)
                )
            elif self.do_mode == "naive":
                x = F.dropout2d(x, self.drop_prob[idx], self.training, inplace=False)
        else:
            x = x.clone()
        return x

    def get_norm_module(self, channels):
        """Return normalization module instance"""
        norm_module = None
        if self.batchnorm is not None:
            norm_module = self.batchnorm(channels)
        elif self.norm == "instance":
            norm_module = nn.InstanceNorm2d(channels, **self.norm_kwargs)
        elif self.norm == "batch":
            norm_module = nn.BatchNorm2d(channels, **self.norm_kwargs)
        return norm_module

    def _conv2d(self, in_channels, kernel_size, bias=True):
        """
        Return convolutional layer.
        Supports standard convolutions and MobileNet-style convolutions.
        """
        padding = tuple(k_size // 2 for k_size in kernel_size)
        if not self.mobile or kernel_size == (1, 1):
            return nn.Conv2d(
                in_channels, self.hidden_ch, kernel_size, padding=padding, bias=bias
            )
        else:
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv_dw",
                            nn.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                groups=in_channels,
                                bias=False,
                            ),
                        ),
                        ("sep_bn", self.get_norm_module(in_channels)),
                        ("sep_relu", nn.ReLU6()),
                        (
                            "conv_sep",
                            nn.Conv2d(in_channels, self.hidden_ch, 1, bias=bias),
                        ),
                    ]
                )
            )

    def _init_hidden(self, input_, cuda=True):
        """Initialize the hidden state"""
        batch_size, _, height, width = input_.data.size()
        prev_state = torch.zeros(batch_size, self.hidden_ch, height, width)
        if cuda:
            prev_state = prev_state.cuda()
        return prev_state


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_channels=None,
        hidden_channels=None,
        kernel_size=(3, 3),
        gate_ksize=(1, 1),
        dropout=(False, False, False),
        drop_prob=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        """
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Arguments:
            input_channels: Number of channels of the input.
            hidden_channels (list): List of hidden channels for each layer.
            kernel_size (tuple): Kernel size of the U and W operations.
            gate_ksize (tuple): Kernel size for the gates.
            dropout: Tuple of Booleans for input, recurrent and output dropout.
            drop_prob: Tuple of dropout probabilities for each selected dropout.
            kwargs: Additional parameters for the cGRU cells.
        """

        super().__init__()

        kernel_size = tuple(kernel_size)
        gate_ksize = tuple(gate_ksize)
        dropout = tuple(dropout)
        drop_prob = tuple(drop_prob)

        if len(hidden_channels) < 0:
            raise ValueError("hidden_channels cannot be negative")

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self._check_kernel_size_consistency(kernel_size)
        self._check_kernel_size_consistency(gate_ksize)
        self.kernel_size = self._extend_for_multilayer(kernel_size)
        self.gate_ksize = self._extend_for_multilayer(gate_ksize)
        self.dropout = self._extend_for_multilayer(dropout)
        drop_prob = self._extend_for_multilayer(drop_prob)
        self.drop_prob = [
            tuple(dp_ if do_ else 0.0 for dp_, do_ in zip(dp, do))
            for dp, do in zip(drop_prob, self.dropout)
        ]
        self.kwargs = kwargs

        cell_list = []
        for idx in range(self.num_layers):
            if idx < self.num_layers - 1:
                # Switch output dropout off for hidden layers.
                # Otherwise it would confict with input dropout.
                this_drop_prob = self.drop_prob[idx][:2] + (0.0,)
            else:
                this_drop_prob = self.drop_prob[idx]
            cell_list.append(
                ConvGRUCell(
                    self.input_channels[idx],
                    self.hidden_channels[idx],
                    self.kernel_size[idx],
                    drop_prob=this_drop_prob,
                    gate_ksize=self.gate_ksize[idx],
                    **kwargs,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden=None):
        """
        Args:
            input_tensor:
                5-D Tensor of shape (b, t, c, h, w)
            hidden:
                optional initial hiddens state

        Returns:
            outputs
        """
        if not hidden:
            hidden = [None] * self.num_layers

        outputs = []
        iterator = torch.unbind(input_tensor, dim=1)

        for t, x in enumerate(iterator):
            for layer_idx in range(self.num_layers):
                if self.cell_list[layer_idx].do_mode == "recurrent" and t == 0:
                    self.cell_list[layer_idx].set_drop_masks()
                (x, h) = self.cell_list[layer_idx](x, hidden[layer_idx])
                hidden[layer_idx] = h.clone()
            outputs.append(x.clone())
        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    def _extend_for_multilayer(self, param):
        if not isinstance(param, list):
            param = [param] * self.num_layers
        elif len(param) != self.num_layers:
            raise ValueError("param and self.num_layers must be equal")

        return param


def log_softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.log_softmax(x, dim=1)
    return x.view(x_size)
