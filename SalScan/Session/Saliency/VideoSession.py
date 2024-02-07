# -*- coding: utf-8 -*-

"""Module containing Video Session class."""

import csv
import glob
import inspect
import json
import os
from abc import ABC
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from SalScan.Metric.Saliency import AUC_JUDD, CC, KLD, NSS, SIM, sAUC
from SalScan.Model.Saliency.CenterBias import CenterBias
from SalScan.Utils import get_logger, is_json_serializable, normalize

from .ImageSession import AbstractSession

logger = get_logger(__name__)

# depending on the hardware used to install old tensorflow versions, the oneAPI Deep
# Neural Network Library (oneDNN) might not be integrated. In this case, a long warning
# message is logged every time a script invokes tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class VideoSaliencySession(AbstractSession, ABC):
    """
    This class is run internally by `SalScan.Evaluate.VideoSaliencyEvaluation` and
    it is highly suggested to perform your evaluations using that class.

    This class performs an evaluation of a video saliency model against a video saliency
    dataset and it allows to specify which saliency metrics to use (by default, are
    `AUC Judd`, `Shuffled AUC`, `NSS`, `CC`, `KLD` and `SIM`).

    Parameters:
        type (str): Set to "saliency" to indicate the session type.
        options (dict): A dictionary containing configuration options for the
                        saliency session. This includes session parameters, metrics
                        for evaluation, and baseline model details.

    Attributes:
        type: Set to "saliency", it indicates the session type.
        dataset: Dataset passed within the `options` argument. Must inherit from
                `SalScan.Dataset.VideoAbstractDataset`
        model: Model passed within the `options` argument. Must inherit from
                `SalScan.Model.AbstractModel`
        name: Set to `self.model.name`
        metrics: Metrics passed within the `options` arguments. Default ones
                are: [`AUC_JUDD`, `sAUC`, `NSS`, `CC`, `KLD`, `SIM`]
        logs_directory: Parameter passed with the `options` arguments, which determines
                        where the evaluation logs are going to be stored.
        debug: Parameter passed with the `options` arguments. When True evaluation is
                going to be run only for the first two stimuli (for debugging purposes).


    Examples:
        >>> dataset = DHF1KDataset(**dhf1k_kwargs)
        >>> dataset.populate()
        >>> model = UniSal
        >>> unisal_params = {
                "mobilnet_weights": "path/to/weights.pht",
                "decoder_weights": "path/to/custom_weights_1.pth",
                "sequence_length": 6
            }

        >>> session = VideoSaliencySession(
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
        options["params"].setdefault("session", {})
        options.setdefault("metrics", [AUC_JUDD, sAUC, NSS, CC, KLD, SIM])

        options.setdefault("baseline_model", CenterBias)
        options["params"].setdefault("baseline_model", {})
        super().__init__(options)

    # within: Optional[str] = "stimulus", between: Optional[str] = ""
    def evaluate(self) -> None:
        """
        Evaluates the video saliency models using the specified metrics. It iterates over
        the dataset, generates saliency maps, and computes metric scores.
        """

        metrics_names = []
        for metric in self.metrics:
            args = inspect.getfullargspec(metric).args
            metrics_names.append(metric.__name__)

        # val_metrics will be later used to build self._val_file
        val_metrics = {metric: defaultdict(list) for metric in metrics_names}

        pbar_items = tqdm(self.dataset)
        pbar_items.set_description(f"{self.name} - eval")
        if hasattr(self.dataset, "__iter__") is True:
            logger.warning(
                "\n❗ For __iter__ based datasets, the tqdm progress bar is an estimate: "
                "the process might finish slightly in advance or later."
            )
        for idx, items in enumerate(pbar_items):
            # Label reverse to the class of the object (i.e. dog)
            frames, gt_salmaps, gt_fixmaps, label = items

            params = {}
            if self.model.type == "saliency_baseline":
                if self.model.n_fix is None:
                    params["n_fix"] = np.sum(gt_fixmaps[0] > 0)
                else:
                    params["n_fix"] = self.model.n_fix

            params_b = {}
            if self.baseline_model.type == "saliency_baseline":
                if self.baseline_model.n_fix is None:
                    params_b["n_fix"] = np.sum(gt_fixmaps[0] > 0)
                else:
                    params_b["n_fix"] = self.baseline_model.n_fix

            gen_salmaps = self.model.run(frames, params)

            # since we need to deal with both Many2One and Many2Many models,
            # we cast the Many2One result into a list
            try:
                if getattr(self.dataset, "many_to_many") is False:
                    gen_salmaps = [gen_salmaps]
            except AttributeError:
                continue

            # checking the size of gt_images have a different size compared to gen_salmap,
            # we resize gen_salmap to the size of gt_images
            gt_h, gt_w = gt_salmaps[0].shape[-2:]
            if tuple([gt_h, gt_w]) != gen_salmaps[0].shape[-2:]:
                gen_salmaps = list(
                    map(lambda img: cv2.resize(img, (gt_w, gt_h)), gen_salmaps)
                )

            # enforcing normalisation to make sure that saliency map output
            # is between 0 and 1 despite some models normalise the output internally
            gen_salmaps = list(
                map(lambda img: normalize(img, method="range"), gen_salmaps)
            )

            for metric in self.metrics:
                args = inspect.getfullargspec(metric).args

                if "saliency_map" in args[0] and "fixation_map" in args[1]:
                    results = []
                    for gen_salmap, gtfixmap in zip(gen_salmaps, gt_fixmaps):
                        results.append(metric(gen_salmap, gtfixmap))

                    val_metrics[metric.__name__]["dataset"].extend(results)
                    # if self.dataset has labels we append (i.e. dog, cat) we append
                    # we append the result to the corresponding label lists.
                    if self.dataset.has_label is True:
                        val_metrics[metric.__name__][label].extend(results)

                elif "saliency_map" in args[0] and "saliency_map" in args[1]:
                    results = []
                    for gen_salmap, gt_salmap in zip(gen_salmaps, gt_salmaps):
                        results.append(metric(gen_salmap, gt_salmap))

                    val_metrics[metric.__name__]["dataset"].extend(results)
                    if self.dataset.has_label is True:
                        val_metrics[metric.__name__][label].extend(results)

                elif "baseline" in args[0]:
                    # the variable stimulus is only used to generate a baseline with
                    # the given dimension (we care about gt_salmaps[0] and not about
                    # its content)
                    stimulus = gt_salmaps[0]
                    # When evaluating a model which is set to baseline we do not
                    # evaluate the IG because its result is going to be 0
                    if self.model.name != self.baseline_model.name:
                        baseline_map = self.baseline_model.run(
                            img=stimulus, params=params_b
                        )
                        baseline_map = normalize(baseline_map, method="range")
                        results = []
                        for gen_salmap, gt_fixmap in zip(gen_salmaps, gt_fixmaps):
                            results.append(metric(baseline_map, gen_salmap, gt_fixmap))

                    else:
                        results = np.nan

                    val_metrics[metric.__name__]["dataset"].extend(results)
                    if self.dataset.has_label is True:
                        val_metrics[metric.__name__][label].extend(results)
                else:
                    raise NotImplementedError

            # In debugging mode, by setting idx to 1 (instead of 0), two values are
            # processed and appended to the metric lists, which guarantee unit-test
            # robustness and speed (unit-testing the whole evaluation could take
            # hours to days)
            if idx == 1 and self.debug is True:
                break

        # class_names is a list like ['dataset', 'class_1', ... , 'class_n']
        class_names = list(val_metrics[list(val_metrics.keys())[0]].keys())

        # after iteration val_pkl will be a list like:
        # ['metric', 'dataset', 'dog', 'cat'], ['NSS', NSS_dataset, NSS_dog, NSS_cat]
        # where the first value of each sublist is meant as an index
        self._val_file = [["metric"] + class_names]
        for metric in val_metrics.keys():
            metric_values = [metric]
            for class_name in class_names:
                class_values = val_metrics[metric][class_name]
                # np.mean() is the fastest even if class_values is a list
                metric_values.append(np.round(np.mean(class_values).item(), 3))

            self._val_file.append(metric_values)

        self.save()

    def save(self):
        """
        Saves evaluation results and session parameters to csv files.

        The method saves `self._val_file` and `self._raw_metrics` generated during the
        evaluation. Furthermore, it stores in the same directory a json file containing
        the parameters of the model.
        """
        session_parameters = self.model.parameters

        for key in list(session_parameters.keys()):
            if is_json_serializable(session_parameters[key]) is False:
                del session_parameters[key]

        session_path = os.path.join(
            self.logs_directory,
            self._config["dataset"].name.lower(),
            self._config["model"].name.lower(),
        )
        if os.path.exists(session_path) is False:
            os.makedirs(session_path)

        prior_exp = glob.glob(f"{session_path}/*")
        new_path = True
        # if no model experiments have been runned before
        if not prior_exp:
            exp_path = os.path.join(session_path, "params_setting_1")
            os.mkdir(exp_path)
        else:
            for exp in prior_exp:
                with open(os.path.join(exp, "params.json"), "r") as f:
                    exp_config = json.load(f)
                # cast values that are were given as list to tuples to avoid
                # experiment duplication:
                # {'dimension': (200, 200)} == {'dimension': [200, 200]} is False

                exp_config_casted = {}
                for k, v in exp_config.items():
                    if isinstance(v, tuple):
                        exp_config_casted[k] = list(v)
                    else:
                        exp_config_casted[k] = v

                if exp_config_casted == session_parameters:
                    exp_path = exp
                    new_path = False
                    break
        if new_path and prior_exp:
            # from glob.glob paths to list of filenames
            exp_folders = [exp.split("/")[-1] for exp in prior_exp]
            # getting the folder parameter rank and filtering hidden files
            exp_folders = [exp.split("_")[-1] for exp in exp_folders if "params" in exp]
            # getting the int of the latest folder: i.e. params_setting_4 -> 4
            last_setting = max([int(exp) for exp in exp_folders])
            # creating new path/folder
            exp_path = os.path.join(session_path, f"params_setting_{last_setting + 1}")
            os.mkdir(exp_path)

        logger.info(f"Storing evaluation files at {exp_path}")

        with open(os.path.join(exp_path, "val.csv"), "w") as f:
            write = csv.writer(f)
            write.writerows(self._val_file)

        with open(os.path.join(exp_path, "params.json"), "w") as f:
            json.dump(session_parameters, f)
