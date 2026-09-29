"""Differentiable IR-SE50 identity encoder; architecture adapted from InsightFace_Pytorch.

Copyright (c) 2018 TreB1eN. See INSIGHTFACE_PYTORCH_LICENSE.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // 16, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        return inputs * self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(inputs)))))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int):
        super().__init__()
        self.shortcut_layer = nn.MaxPool2d(1, stride) if in_channels == channels else nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, channels, 3, padding=1, bias=False), nn.PReLU(channels), nn.Conv2d(channels, channels, 3, stride, 1, bias=False), nn.BatchNorm2d(channels), SqueezeExcitation(channels))

    def forward(self, inputs):
        return self.shortcut_layer(inputs) + self.res_layer(inputs)


class IRSE50(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        blocks, in_channels = [], 64
        for channels, count in ((64, 3), (128, 4), (256, 14), (512, 3)):
            blocks.append(ResidualBlock(in_channels, channels, 2))
            blocks.extend(ResidualBlock(channels, channels, 1) for _ in range(count - 1))
            in_channels = channels
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(0.6), nn.Flatten(), nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512))

    def forward(self, inputs):
        return F.normalize(self.output_layer(self.body(self.input_layer(inputs))), dim=-1, eps=1e-8)


def preprocess_identity_images(images: torch.Tensor, channel_order: str) -> torch.Tensor:
    """Resize RGB [-1, 1] images without detaching the reconstruction gradient."""
    images = F.interpolate(images.float(), size=(112, 112), mode="bilinear", align_corners=False)
    return images[:, [2, 1, 0]] if channel_order == "bgr" else images


class IdentityEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, channel_order: str):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.channel_order = channel_order
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(preprocess_identity_images(images, self.channel_order)).float(), dim=-1, eps=1e-8)


def load_identity_encoder(checkpoint: Path, channel_order: str) -> IdentityEncoder:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected an IR-SE50 tensor state dict, optionally under 'state_dict'")
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model = IRSE50()
    model.load_state_dict(state, strict=True)
    return IdentityEncoder(model, channel_order)
