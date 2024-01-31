# -*- coding: utf-8 -*-

"""Module containing Session abstract class."""

from abc import ABC
from collections import defaultdict
import csv
import copy
import glob
import inspect
import json
import os
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd

from SalScan.Dataset.AbstractDataset import AbstractDataset
from SalScan.Model.AbstractModel import AbstractModel
from SalScan.Utils import normalize, get_logger

logger = get_logger(__name__)


def merge_dicts(new_dict: dict, dict_to_merge: dict) -> dict:
    """
    Merges two dictionaries, updating the first dictionary with values from the second.
    If a key exists in both dictionaries, the value from `dict_to_merge` is used.
    If a key in `dict_to_merge` is a dictionary, it's recursively merged.

    Args:
        new_dict (dict): The dictionary to be updated.
        dict_to_merge (dict): The dictionary with values to be merged into `new_dict`.

    Returns:
        dict: The updated dictionary after merging.
    """
    new_dict = copy.deepcopy(new_dict)

    for key in dict_to_merge:
        if key not in new_dict:
            new_dict[key] = copy.deepcopy(dict_to_merge[key])
        elif isinstance(dict_to_merge[key], dict):
            new_dict[key] = merge_dicts(new_dict[key], dict_to_merge[key])

    return new_dict


class AbstractSession(ABC):
    """
    An abstract base class used to create Sessions, which are objects used to evaluate
    models against datasets by using a set of metrics.

    This class must be inherited and not instantiated directly.

    Parameters:
        type (str): The type of the session (e.g., "saliency"). Should be overridden in
                    subclasses.
        _default_config (dict): The default configuration for the session. Can be
                                overridden by subclasses.
        options (dict): A dictionary of configuration options for the session.
                        These options are merged with the default configuration.

    Attributes:
        type: Session type like "saliency".
        dataset: dataset passed within the `options` argument. Must inherit from
                `SalScan.Dataset.AbstractDataset`
        model: model passed within the `options` argument. Must inherit from
                `SalScan.Model.AbstractModel`
        name: set to `self.model.name`
        metrics: metrics passed within the `options` arguments.
        baseline_model: baseline model passed within the `options` argument.
        logs_directory: Parameter passed with the `options` arguments, which determines
                        where the evaluation logs are going to be stored.
        debug: Parameter passed with the `options` arguments. When True evaluation is
        performed only on the first few stimuli (for debugging purposes).
    """

    _default_config: dict = {
        "dataset": None,  # Must be an object that inherited from AbstractDataset class
        "model": None,  # Model to use
        "logs_directory": os.path.join(os.getcwd(), "val_logs"),  # Where to store results
        "metrics": None,  # Metrics used for evaluation
        "debug": False,  # Evaluate for 2 steps, for debugging purpose
        "params": {
            "model": {},  # Model parameters
        },
    }

    type: str = "Unknown"

    def __init__(self, options):
        self.dataset = None
        self.model = None
        self.metrics = None
        self._config = None

        self._val_file = None
        self._raw_metrics = None

        self._config = merge_dicts(options, self._default_config)

        # MODEL
        if self._config["model"] is not None:
            if issubclass(self._config["model"], AbstractModel):
                self.model = self._config["model"](self._config["params"]["model"])
                self.model.parameters = self._config["params"]["model"]
                self.name = self.model.name
            else:
                raise ValueError(
                    "Provided model must inherit from AbstractModel. "
                    "i.e. see SalScan/Model/Saliency/UniSal.py"
                )

        # DATASET
        if self._config["dataset"] is not None:
            if isinstance(self._config["dataset"], AbstractDataset):
                self.dataset = self._config["dataset"]
            else:
                raise ValueError(
                    "Provided dataset must inherit from AbstractDataset. "
                    "i.e. see SalScan/Dataset/Image/CAT2000Dataset.py"
                )

        # BASELINE
        if self._config["baseline_model"] is not None:
            if issubclass(self._config["model"], AbstractModel):
                self.baseline_model = self._config["baseline_model"](
                    self._config["params"]["baseline_model"]
                )
            else:
                raise ValueError(
                    "Provided baseline-model must inherit from AbstractModel. "
                    "i.e. see SalScan/Model/Saliency/CenterBias.py"
                )

        # METRICS
        if self._config["metrics"] is not None:
            self.metrics = self._config["metrics"]

        # LOGS DIRECTORY
        self.logs_directory = self._config["logs_directory"]

        # DEBUG
        self.debug = self._config["debug"]

    def save(self):
        """
        Saves evaluation results and session parameters to csv files.

        The method saves `self._val_file` and `self._raw_metrics` generated during the
        evaluation. Furthermore, it stores in the same directory a json file containing
        the parameters of the model.
        """

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

        with open(os.path.join(exp_path, "raw.csv"), "w") as f:
            write = csv.writer(f)
            write.writerows(self._raw_metrics)

        with open(os.path.join(exp_path, "val.csv"), "w") as f:
            write = csv.writer(f)
            write.writerows(self._val_file)

        with open(os.path.join(exp_path, "params.json"), "w") as f:
            json.dump(self.model.parameters, f)

    # within: Optional[str] = "stimulus", between: Optional[str] = ""
    def evaluate(
        self,
    ) -> None:
        """
        Evaluates the session's model using configured metrics.

        This method processes the dataset, applies the model, and computes metrics for the
        results. It populates `self._raw_metrics` and `self._val_file` with evaluation
        results.

        Additional arguments can be added for specific evaluation criteria or metrics.
        """

        self.saliency_scores = pd.DataFrame([])

        metrics_names = []
        for metric in self.metrics:
            args = inspect.getfullargspec(metric).args
            metrics_names.append(metric.__name__)

        # self._raw_metrics is a class attibute since it will be saved with self.save()
        self._raw_metrics = [["stimulus_path"] + metrics_names]
        # val_metrics will be later used to build self._val_file
        val_metrics = {metric: defaultdict(list) for metric in metrics_names}

        pbar_items = tqdm(range(len(self.dataset)))
        pbar_items.set_description(f"{self.name} - eval")

        for idx in pbar_items:
            # Label reverse to the class of the object (i.e. dog)
            stimulus_path, stimulus, gtsalmap, gtfixmap, label = self.dataset[idx]

            gt_height, gt_width = gtfixmap.shape

            params = {}
            if self.model.type == "saliency_baseline":
                if self.model.n_fix is None:
                    params["n_fix"] = np.sum(gtfixmap > 0)
                else:
                    params["n_fix"] = self.model.n_fix

            params_b = {}
            if self.baseline_model.type == "saliency_baseline":
                if self.baseline_model.n_fix is None:
                    params_b["n_fix"] = np.sum(gtfixmap > 0)
                else:
                    params_b["n_fix"] = self.baseline_model.n_fix

            gensalmap = self.model.run(stimulus, params)
            # enforcing normalisation to make sure that saliency map output
            # is between 0 and 1 despite some models normalise the output internally
            gensalmap = normalize(gensalmap, method="range")
            if gensalmap.shape != gtfixmap.shape:
                gensalmap = cv2.resize(gensalmap, (gt_width, gt_height), cv2.INTER_CUBIC)

            # np.min(gensalmap) is 0, np.max(gensalmap) is 1
            # np.min(gtsalmap) is 0, np.max(gtsalmap) is 1
            # np.min(gtfixmap) is 0, np.max(gtfixmap) is 1

            gtsalmap = gtsalmap.astype("float64")
            gtfixmap = gtfixmap.astype("float64")

            # for each stimulus (img) we store here the metrics, temp stands for temporal
            temp_raw_metrics = []

            for metric in self.metrics:
                args = inspect.getfullargspec(metric).args

                if "saliency_map" in args[0] and "fixation_map" in args[1]:
                    result = metric(gensalmap, gtfixmap)

                    temp_raw_metrics.append(result)
                    val_metrics[metric.__name__]["dataset"].append(result)
                    if self.dataset.has_label is True:
                        val_metrics[metric.__name__][label].append(result)

                elif "saliency_map" in args[0] and "saliency_map" in args[1]:
                    result = metric(gensalmap, gtsalmap)

                    temp_raw_metrics.append(result)
                    val_metrics[metric.__name__]["dataset"].append(result)
                    if self.dataset.has_label is True:
                        val_metrics[metric.__name__][label].append(result)

                elif "baseline" in args[0]:
                    # within this elif we are going to evaluate the IG (information gain)
                    # of a model against a baseline. When evaluating a model which is
                    # also set to baseline we do not want to evaluate the IG because its
                    # result is going to be 0
                    if self.model.name != self.baseline_model.name:
                        baseline_map = self.baseline_model.run(
                            img=stimulus, params=params_b
                        )
                        baseline_map = normalize(baseline_map, method="range")
                        result = metric(baseline_map, gensalmap, gtfixmap)
                    else:
                        result = np.nan

                    temp_raw_metrics.append(result)
                    val_metrics[metric.__name__]["dataset"].append(result)
                    if self.dataset.has_label is True:
                        val_metrics[metric.__name__][label].append(result)
                else:
                    raise NotImplementedError

            self._raw_metrics.append([stimulus_path] + temp_raw_metrics)

            # In debugging mode, by setting idx to 1 (instead of 0), two values are
            # processed and appended to the metric lists, which guarantee unit-test
            # robustness and speed (unit-testing the whole evaluation could take
            # hours to days)
            if self.debug is True and idx == 1:
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
