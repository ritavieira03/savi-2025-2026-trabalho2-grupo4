#!/usr/bin/env python3
#shebang line for linux / mac

import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def read_boxes_csv(label_path: Path, img_w: int, img_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    label, xmin, ymin, xmax, ymax
    2, 83, 95, 99, 111
    ...
    Retorna:
      boxes:  FloatTensor [N,4] (x1,y1,x2,y2) em pixels
      labels: Int64Tensor [N] com classes 1..10 (0 é background no torchvision)
    """
    if (not label_path.is_file()) or label_path.stat().st_size == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    boxes: List[List[float]] = []
    labels: List[int] = []

    ## Obter as boxes do ficheiro Labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("label"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))  # 0..9
            x1 = float(parts[1]); y1 = float(parts[2])
            x2 = float(parts[3]); y2 = float(parts[4])

            x1 = _clamp(x1, 0, img_w - 1)
            y1 = _clamp(y1, 0, img_h - 1)
            x2 = _clamp(x2, 0, img_w - 1)
            y2 = _clamp(y2, 0, img_h - 1)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


class MNISTDetectionDataset(Dataset):
    def __init__(self, dataset_folder: str, split: str = "train", keep_empty: bool = False):
        root = Path(dataset_folder)
        self.images_dir = root / split / "images"
        self.labels_dir = root / split / "labels"

        self.image_paths = sorted(self.images_dir.glob("*.png"))

        # Filtra imagens sem labels (em detecção costuma ser melhor para treino)
        if not keep_empty:
            filtered = []
            for ip in self.image_paths:
                lp = self.labels_dir / f"{ip.stem}.txt"
                if lp.is_file() and lp.stat().st_size > 0:
                    filtered.append(ip)
            self.image_paths = filtered

        if len(self.image_paths) == 0:
            raise ValueError(
                f"Dataset vazio! Verifica paths:\n"
                f"images_dir={self.images_dir}\nlabels_dir={self.labels_dir}\n"
                f"(e se tens *.png em images/ e *.txt em labels/)"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")  # FasterRCNN espera 3 canais
        w, h = img.size

        boxes, labels = read_boxes_csv(label_path, w, h)

        # tensor float [0,1] (C,H,W)
        img_t = to_tensor(img)

        # extras úteis
        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return img_t, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
