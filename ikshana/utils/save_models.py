import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SaveBestModel:
    def __init__(self, best_valid_loss: float = float("inf")):
        self.best_valid_loss = best_valid_loss

    def __call__(
        self,
        current_valid_loss: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Any,
        criterion: Any,
        name: str,
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            logger.info(f"Best validation loss: {self.best_valid_loss}")
            logger.info(f"Saving best model for epoch: {epoch+1}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                f"checkpoints/best_model_{name}.pth",
            )


def save_model(
    epoch: int,
    model: torch.nn.Module,
    optimizer: Any,
    criterion: Any,
    name: str,
):
    logger.info(f"Saving model at {epoch + 1}\n")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        f"checkpoints/model_{name}.pth",
    )
