import copy
import logging
import os
import random

import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
from torchvision import transforms

from ikshana.data.generate_dataset import generate_classification_dataset
from ikshana.data.loader import BaseDataLoader
from ikshana.data.transformation import TransformBase
from ikshana.data.visualization import Visualize
from ikshana.models.resnet import ResNet18
from ikshana.models.test import test_model
from ikshana.models.train import train_model

logger = logging.getLogger(__name__)
###############################################################################
######################### HYPER-PARAMETER DEFINITIONS #########################
###############################################################################
EPOCHS = 200
RANDOM_SEED = 42
BATCH_SIZE = 8
DEVICE = "cpu"
LEARNING_RATE = 0.001
VALID_SPLIT = 0.1

if torch.backends.mps.is_available():  # type: ignore
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

torch.manual_seed(RANDOM_SEED)

###############################################################################
############ DEFINITION OF DATASET PATHS AND INITIAL VISUALIZATION ############
###############################################################################
path = os.path.join("datasets", "Gender01")
train_path = os.path.join("datasets", "Gender01", "train")
test_path = os.path.join("datasets", "Gender01", "test")

vis = Visualize("datasets", "Gender01", "Classification")
vis.visualize_category_data(4, "categorical_samples")

###############################################################################
############# AUGMENTATION AND TRANSFORMATION OF DATA DEFINITIONS $############
###############################################################################
augmentations = [
    transforms.ColorJitter(),
    transforms.GaussianBlur(kernel_size=(5, 9)),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomEqualize(),
]
transform_data = TransformBase(224, augmentation_stratgies=augmentations)
train_transform = transform_data.train_transform()
test_transform = transform_data.test_transform()

transform_data = TransformBase(224)
train_transform_visualize = transform_data.train_transform()
test_transform_visualize = transform_data.test_transform()

data_transforms = {
    "train_transform": train_transform,
    "validation_transform": test_transform,
    "test_transform": test_transform,
}

###############################################################################
############################## DATASET CREATION ###############################
###############################################################################
(
    train_dataset,
    validation_dataset,
    test_dataset,
) = generate_classification_dataset(path, data_transforms)

dataset = {
    "train_dataset": train_dataset,
    "validation_dataset": validation_dataset,
    "test_dataset": test_dataset,
}

class_names = test_dataset.classes
num_classes = len(class_names)

rand_num = random.randint(0, len(train_dataset))
random_sample = train_dataset[rand_num]
Visualize.visualize_individual_image(
    random_sample[0], class_names[random_sample[1]], rand_num, True
)

###############################################################################
################################# DATA LOADER #################################
###############################################################################
data_loader = BaseDataLoader(batch_size=BATCH_SIZE)

train_dataloader = data_loader.train_loader(train_dataset)
validation_dataloader = data_loader.validation_loader(validation_dataset)
test_dataloader = data_loader.test_loader(test_dataset)

data_loaders = {
    "train_dataloader": train_dataloader,
    "validation_dataloader": validation_dataloader,
    "test_dataloader": test_dataloader,
}

###############################################################################
############################ MODEL INITIALIZATION #############################
###############################################################################
genders_model = ResNet18(num_classes=num_classes, in_channels=1).to(
    device=DEVICE  # type: ignore
)
summary_model = copy.deepcopy(genders_model).to("cpu")

test_genders_model = ResNet18(num_classes=num_classes, in_channels=1).to(
    device=DEVICE  # type: ignore
)

total_trainable_params = sum(
    p.numel() for p in genders_model.parameters() if p.requires_grad
)
print(f"{total_trainable_params:,} trainable parameters.")
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=genders_model.parameters(), lr=LEARNING_RATE)

summary(summary_model, (1, 224, 224))

###############################################################################
##################### MODEL TRAINING AND EVALUATION STEPS #####################
###############################################################################

train_model(
    genders_model,
    data_loaders,
    loss_function,
    optimizer,
    DEVICE,
    EPOCHS,
    vis,
    dataset,
    "genders",
)

test_model(
    test_genders_model,
    os.path.join("checkpoints", "best_model_genders.pth"),
    test_dataloader,
    DEVICE,
    dataset,
)

###############################################################################
################### MODEL EXPLAINABILITY AND VISUALIZATIONS ###################
###############################################################################

checkpoint = torch.load(os.path.join("checkpoints", "best_model_genders.pth"))
test_genders_model.load_state_dict(checkpoint["model_state_dict"])
target_layer = [test_genders_model.layer3[-1]]

vis.plot_correct_incorrect_classifications(
    train_dataset,
    os.path.join("checkpoints", "best_model_genders.pth"),
    test_genders_model,
    DEVICE,
    "train",
    target_layer,
)

vis.plot_correct_incorrect_classifications(
    validation_dataset,
    os.path.join("checkpoints", "best_model_genders.pth"),
    test_genders_model,
    DEVICE,
    "validation",
    target_layer,
)

vis.plot_correct_incorrect_classifications(
    test_dataset,
    os.path.join("checkpoints", "best_model_genders.pth"),
    test_genders_model,
    DEVICE,
    "test",
    target_layer,
)
