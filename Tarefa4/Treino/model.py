#!/usr/bin/env python3
#shebang line for linux / mac

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def build_detector(num_classes=11):
    m = torchvision.models.mobilenet_v3_small(weights=None).features
    m.out_channels = 576

    anchor_gen = AnchorGenerator(
        sizes=((12, 16, 22, 32),),
        aspect_ratios=((0.75, 1.0, 1.33),),
    )

    model = FasterRCNN(
        backbone=m,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        min_size=128,
        max_size=128,
    )
    return model
