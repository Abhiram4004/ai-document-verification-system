"""Visual deep learning model and Grad-CAM interpretability for document verification.

Implements transfer learning using a pretrained ResNet-18 backbone with:
- Binary classification head for document manipulation detection
- Class-weighted Cross-Entropy loss for handling ~5:1 imbalance
- Training-only data augmentations (rotation, jitter, scaling)
- Strict exclusion of metadata/annotations (uses ONLY raw pixels and target label)
- Grad-CAM attention map generation for visual model transparency
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DocumentDataset(Dataset):
    """PyTorch Dataset loading raw document images without annotation leakage."""

    def __init__(self, records: list, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)
        return tensor, record.label, record.filename


def get_visual_transforms(image_size: int = 384) -> Tuple[Any, Any]:
    """Return train and validation/inference image transform pipelines.

    Augmentations are applied STRICTLY during training.
    Validation and test sets use deterministic resizing and normalization only.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


class DocumentForgeryResNet(nn.Module):
    """ResNet-18 Transfer Learning model with Grad-CAM gradient hooks."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DocumentForgeryResNet.")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        # Feature extractor layers
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Binary classification head with dropout regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(128, 2),
        )

        # Grad-CAM storage hooks
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Register hook for Grad-CAM during forward pass
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def freeze_backbone(self):
        """Freeze conv1 through layer4, leaving only the classifier head trainable."""
        for name, param in self.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_upper_layers(self):
        """Unfreeze layer4 and classifier head for fine-tuning."""
        for name, param in self.named_parameters():
            if "layer4" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


def compute_gradcam(
    model: DocumentForgeryResNet,
    image_tensor: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """Compute Grad-CAM attention heatmap for a single image tensor.

    Args:
        model: DocumentForgeryResNet instance
        image_tensor: Normalized (1, 3, H, W) tensor
        target_class: Class index for activation (default 1 for Forged)

    Returns:
        2D numpy array of normalized attention intensity [0.0, 1.0]
    """
    model.eval()
    model.zero_grad()

    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    logits = model(image_tensor)

    score = logits[0, target_class]
    score.backward()

    # Pooled gradients across channels
    gradients = model.gradients
    activations = model.activations

    if gradients is None or activations is None:
        return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample to input tensor resolution
    cam = F.interpolate(cam, size=image_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()

    # Min-max normalization
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-6:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


def overlay_gradcam(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a 2D Grad-CAM heatmap array onto a PIL image.

    Args:
        pil_image: Original PIL Image (RGB)
        cam: 2D numpy array of normalized attention intensity [0.0, 1.0]
        alpha: Heatmap blend opacity (0.0 to 1.0)
        colormap: Matplotlib colormap name (default 'jet')

    Returns:
        Blended PIL Image
    """
    import matplotlib.pyplot as plt

    rgb_img = pil_image.convert("RGB")
    # Resize CAM to match original image dimensions
    cam_pil = Image.fromarray((np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)).resize(
        rgb_img.size, resample=Image.Resampling.BILINEAR
    )
    cam_norm = np.array(cam_pil, dtype=np.float32) / 255.0

    cmap = plt.get_cmap(colormap)
    rgba = (cmap(cam_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(rgba, mode="RGB")

    return Image.blend(rgb_img, heatmap_pil, alpha=alpha)
