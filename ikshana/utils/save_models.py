import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")


class SaveBestModel:
    def __init__(
        self, dataset_name: str, best_valid_loss: float = float("inf")
    ):
        self.best_valid_loss = best_valid_loss
        self.dataset_name = dataset_name

    def __call__(
        self,
        current_valid_loss: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Any,
        criterion: Any,
        name: str,
    ):
        if not os.path.exists(os.path.join("checkpoints", self.dataset_name)):
            os.mkdir(os.path.join("checkpoints", self.dataset_name))

        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            logger.info(f"Saving best model for epoch: {epoch}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                os.path.join(
                    "checkpoints", self.dataset_name, f"best_model_{name}.pth"
                ),
            )


def save_model(
    epoch: int,
    model: torch.nn.Module,
    optimizer: Any,
    criterion: Any,
    name: str,
    dataset_name: str,
):
    logger.info(f"Saving model at {epoch}\n")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        os.path.join("checkpoints", dataset_name, f"model_{name}.pth"),
    )
