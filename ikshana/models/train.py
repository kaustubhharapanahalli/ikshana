import logging
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from ikshana.data.visualization import Visualize
from ikshana.utils.save_models import SaveBestModel, save_model

logger = logging.getLogger(__name__)

TRAINING_PREDICTIONS, VALIDATION_PREDICTIONS = list(), list()


def train_function(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Any,
    loss_function: Any,
    device: str,
):
    model.train()
    logger.info("Training")
    train_running_loss, train_running_correct, counter = 0.0, 0, 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        predictions = model(image)

        loss = loss_function(predictions, labels)
        train_running_loss += loss.item()

        _, preds = torch.max(predictions.data, 1)
        TRAINING_PREDICTIONS.extend(preds.tolist())
        train_running_correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_accuracy = train_running_correct / len(train_loader.dataset)  # type: ignore

    return epoch_loss, epoch_accuracy * 100


def validate_function(
    model: nn.Module,
    valid_loader: DataLoader | Subset,
    loss_function: Any,
    device: str,
):
    model.eval()
    logger.info("Validation")
    valid_running_loss, valid_running_correct, counter = 0.0, 0, 0

    with torch.inference_mode():
        for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):  # type: ignore
            counter += 1

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_function(predictions, labels)

            valid_running_loss += loss.item()

            _, preds = torch.max(predictions.data, 1)
            VALIDATION_PREDICTIONS.extend(preds.tolist())
            valid_running_correct += (preds == labels).sum().item()

    epoch_loss = valid_running_loss / counter
    epoch_accuracy = valid_running_correct / len(valid_loader.dataset)  # type: ignore

    return epoch_loss, epoch_accuracy * 100


def train_model(
    model: nn.Module,
    data_loaders: Dict,
    loss_function: Any,
    optimizer: Any,
    device: str,
    epochs: int,
    vis: Visualize,
    dataset: Dict,
):
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []
    save_best_model = SaveBestModel()

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train_function(
            model,
            data_loaders["train_dataloader"],
            optimizer,
            loss_function,
            device,
        )
        valid_epoch_loss, valid_epoch_acc = validate_function(
            model, data_loaders["validation_dataloader"], loss_function, device
        )

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)
        statement = (
            f"Training loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:.3f},"
            + f" Validation loss: {valid_epoch_loss:.3f}, Validation accuracy: {valid_epoch_acc:.3f}"
        )
        logger.info(statement)

        save_best_model(
            valid_epoch_loss,
            epoch,
            model,
            optimizer,
            loss_function,
            "directions",
        )
        if epoch % 5 == 0:
            save_model(
                epoch,
                model,
                optimizer,
                loss_function,
                f"epoch_{epoch}_directions",
            )

    vis.save_model_plots(train_acc, valid_acc, train_loss, valid_loss)
    logger.info("TRAINING COMPLETE\n")

    vis.plot_correct_incorrect_classifications(
        dataset["train_dataset"], TRAINING_PREDICTIONS, "training"
    )

    vis.plot_correct_incorrect_classifications(
        dataset["validation_dataset"], VALIDATION_PREDICTIONS, "validation"
    )
