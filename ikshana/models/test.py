import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# from ikshana.data.visualization import Visualize
from ikshana.utils.metrics import classification_report_function

logger = logging.getLogger(__name__)


def test_model(
    model: nn.Module,
    model_path: str,
    test_loader: DataLoader,
    device: str,
    dataset,
):
    class_names = dataset["test_dataset"].classes
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    logging.info("Model Testing")
    test_running_correct = 0
    counter = 0

    with torch.inference_mode():
        for idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            _, preds = torch.max(predictions.data, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            test_running_correct += (preds == labels).sum().item()

    test_acc = 100 * (test_running_correct) / len(test_loader.dataset)  # type: ignore
    logger.info(f"Test Accuracy: {test_acc}")
    logger.info(
        classification_report_function(
            np.array(all_labels), np.array(all_preds), list(class_names)
        )
    )
    logger.info(f"Predictions: {all_preds}")
    logger.info(f"Labels     : {all_labels}")


def infer_image(model: nn.Module, model_path: str, image: torch.Tensor):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        prediction = model(image)
        _, pred = torch.max(prediction.data, 1)

    return pred
