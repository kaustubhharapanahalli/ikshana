import logging
from typing import Any, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.functional import (
    accuracy,
    auroc,
    confusion_matrix,
    f1_score,
    precision,
    recall,
    roc,
    specificity,
)


class Metrics:
    def __init__(
        self,
        task_type: Literal["binary", "multiclass", "multilabel"],
        number_of_classes: int,
        class_names: List,
        average: Literal["micro", "macro", "weighted", "none"] | None = "none",
        multidim_average: Literal["global", "samplewise"] | None = "global",
    ) -> None:
        self.task_type: Literal[
            "binary", "multiclass", "multilabel"
        ] = task_type
        self.number_of_classes: int = number_of_classes
        self.class_names: List = class_names
        self.average: Literal[
            "micro", "macro", "weighted", "none"
        ] | None = average
        self.multidim_average: Literal[
            "global", "samplewise"
        ] | None = multidim_average

    def accuracy_score(
        self, predictions: torch.Tensor, ground_truth: torch.Tensor
    ):
        score = accuracy(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
        )

        return score.item() * 100

    def auroc_score(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        threshold: int | None = None,
    ):
        score = auroc(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
            average=self.average,  # type: ignore
            thresholds=threshold,
        )

        return score

    def confusion_matrix_evaluation(
        self, predictions: torch.Tensor, ground_truth: torch.Tensor
    ):
        matrix = confusion_matrix(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
        )

        return matrix

    def f1_score_evaluation(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        score = f1_score(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
            average=self.average,
            multidim_average=self.multidim_average,
        )

        return score.item()

    def precision_score(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        score = precision(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
            average=self.average,
            multidim_average=self.multidim_average,
        )

        return score.item()

    def recall_score(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        score = recall(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
            average=self.average,
            multidim_average=self.multidim_average,
        )

        return score.item()

    def roc_score(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        score = roc(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
        )

        return score

    def specificity_score(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        score = specificity(
            preds=predictions,
            target=ground_truth,
            task=self.task_type,
            num_classes=self.number_of_classes,
            average=self.average,
            multidim_average=self.multidim_average,
        )

        return score.item()
