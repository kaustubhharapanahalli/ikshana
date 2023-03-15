from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def compute_grad_cam(model, target_layers, image):
    cam = GradCAM(model, target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=image)
    return grayscale_cam
