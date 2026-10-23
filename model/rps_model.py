# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Load the user's SqueezeNet 1.1 RPS checkpoint and RGB validation images."""
import hashlib
import json
from pathlib import Path

import torch


def load_rps(config_path):
    from PIL import Image
    from torchvision.models import squeezenet1_1
    from torchvision.transforms.functional import pil_to_tensor

    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else config_path.parent / path

    checkpoint = resolve(cfg["checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = squeezenet1_1(weights=None, num_classes=4).eval().to(memory_format=torch.channels_last)
    model.load_state_dict(state, strict=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    inputs, image_paths, rgb_images = [], [], []
    for value in cfg["images"]:
        path = resolve(value)
        with Image.open(path) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            rgb = pil_to_tensor(image)
            rgb_images.append(rgb.permute(1, 2, 0).contiguous())
            x = rgb.unsqueeze(0).float() / 255.0
        inputs.append(((x - mean) / std).contiguous(memory_format=torch.channels_last))
        image_paths.append(str(path))
    if not inputs:
        raise ValueError("RPS export needs representative RGB images")

    # For fixed 224x224 inputs, all three max-pools see odd spatial sizes:
    # 111 -> 55 -> 27 -> 13. Floor and ceil modes are equivalent here.
    # Verify before choosing the CMSIS-NN-supported floor-mode implementation.
    with torch.no_grad():
        original = [model(x).clone() for x in inputs]
    for layer in model.modules():
        if isinstance(layer, torch.nn.MaxPool2d):
            layer.ceil_mode = False
    with torch.no_grad():
        for x, reference in zip(inputs, original):
            if not torch.equal(model(x), reference):
                raise RuntimeError("Pool configuration changed the RPS FP32 reference")

    labels = cfg["labels"]
    if len(labels) != 4:
        raise ValueError("RPS classifier requires four ordered class labels")
    return model, inputs, dict(
        name="SqueezeNet 1.1 RPS", labels=labels, images=image_paths,
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        normalization=dict(mean=mean.flatten().tolist(), std=std.flatten().tolist()),
        resize="224x224 RGB, bilinear", calibration_warning="Only the supplied images are used; this is a deployment smoke test, not dataset accuracy validation.",
    ), rgb_images
