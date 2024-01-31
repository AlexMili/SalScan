# -*- coding: utf-8 -*-

"""Module containing Session abstract class."""

from abc import ABC

from SalScan.Model.Saliency.CenterBias import CenterBias
from SalScan.Metric.Saliency import KLD, CC, SIM, NSS, AUC_JUDD, sAUC, IG
from SalScan.Session.AbstractSession import AbstractSession


class ImageSaliencySession(AbstractSession, ABC):
    """
    This class is run internally by `SalScan.Evaluate.ImageSaliencyEvaluation` and
    it is highly suggested to perform your evaluations using that class.

    This class performs an evaluation of an image saliency model against an image saliency
    dataset.
    It allows to specify which saliency metrics to use during the evaluation and also the
    baseline model used to compute the "information gain" (saliency metric).
    By default it scores the model using 6 metrics: `AUC Judd`, `Shuffled AUC`, `NSS`,
    `CC`, `KLD`, `SIM` and `IG`. The default baseline model used to compute IG
    (Information Gain) is `SalScan.Model.Saliency.CenterBias`.

    Parameters:
        options (dict): A dictionary containing configuration options for the
                        saliency session. This includes session parameters, metrics
                        for evaluation, and baseline model details.

    Attributes:
        type: Set to "saliency", it indicates the session type.
        dataset: dataset passed within the `options` argument. Must inherit from
                `SalScan.Dataset.AbstractDataset`
        model: Model passed within the `options` argument. Must inherit from
                `SalScan.Model.AbstractModel`
        name: Set to `self.model.name`
        metrics: Metrics passed within the `options` arguments. Default ones
                are: [`AUC_JUDD`, `sAUC`, `NSS`, `CC`, `KLD`, `SIM`, `IG`]
        baseline_model: Baseline model passed within the `options` argument.
                        Defaults to `SalScan.Model.Saliency.CenterBias`
        logs_directory: Parameter passed with the `options` arguments, which determines
                        where the evaluation logs are going to be stored.
        debug: Parameter passed with the `options` arguments. When True evaluation is
                going to be run only for the first two stimuli (for debugging purposes).

    Examples:
        >>> dataset = MIT1003Dataset("path/to/dataset")
        >>> dataset.populate()
        >>> model = UniSal_Static
        >>> unisal_params = {
                "mobilnet_weights": "path/to/weights.pht",
                "decoder_weights": "path/to/custom_weights_1.pth",
            }

        >>> session = ImageSaliencySession(
                "logs_directory": "results",
                "dataset": dataset,
                "model": model,
                "metrics": [NSS, CC, KLD],
                "params": {"model": unisal_params}
            )
        >>> session.evaluate()
    """

    type = "saliency"

    def __init__(self, options: dict) -> None:
        options.setdefault("params", {})
        options.setdefault("metrics", [AUC_JUDD, sAUC, NSS, CC, KLD, SIM, IG])
        options.setdefault("baseline_model", CenterBias)
        options["params"].setdefault("baseline_model", {"diameter": 0.15})
        super().__init__(options)
