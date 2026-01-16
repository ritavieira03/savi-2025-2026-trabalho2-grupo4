#!/usr/bin/env python3
import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, sys
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Tarefa1.model import ModelBetterCNN

# ============================================================
# Utilitários de Detecção
# ============================================================

def get_iou(box1, box2, size=28):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[0] + size, box2[0] + size), min(box1[1] + size, box2[1] + size)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box_area = size * size
    union_area = 2 * box_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(detections, iou_thresh=0.2):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if get_iou((best["x"], best["y"]), (d["x"], d["y"])) < iou_thresh
        ]
    return keep

# ============================================================
# Sliding Window
# ============================================================

def detect_digits(image, model, device, stride=2, batch_size=128):
    h, w = image.shape
    model.eval()

    MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
    if image.max() > 1.0:
        image = image / 255.0

    crops, coords = [], []

    for y in range(0, h - 28 + 1, stride):
        for x in range(0, w - 28 + 1, stride):
            crop = image[y:y + 28, x:x + 28]

            # Filtro rápido de fundo
            if crop.mean() < 0.05 or crop.max() < 0.3:
                continue

            crop_norm = (crop - MNIST_MEAN) / MNIST_STD
            crops.append(crop_norm)
            coords.append((x, y))

    if not crops:
        return []

    crops_t = torch.tensor(np.array(crops), dtype=torch.float32).unsqueeze(1).to(device)
    detections = []

    for i in range(0, len(crops_t), batch_size):
        batch = crops_t[i:i + batch_size]
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

            top_probs, top_classes = torch.topk(probs, k=2, dim=1)
            confs = top_probs[:, 0]
            margin = top_probs[:, 0] - top_probs[:, 1]
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        for j in range(len(batch)):
            idx = i + j
            if confs[j] > 0.999 and entropy[j] < 1.0 and margin[j] > 0.8:
                detections.append({
                    "x": coords[idx][0],
                    "y": coords[idx][1],
                    "conf": confs[j].item(),
                    "cls": top_classes[j, 0].item()
                })

    return non_max_suppression(detections)

# ============================================================
# Execução
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Tarefa 3 - Deteção de dígitos por Sliding Window")
    parser.add_argument("--images_dir", type=str, default="../Tarefa2/data/versaoD/test/images")
    parser.add_argument("--checkpoint", type=str, default="../Tarefa1/experiments/best.pkl")
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Número de imagens a analisar (default: 5)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBetterCNN().to(device)

    if not os.path.exists(args.checkpoint):
        print(f"Erro: Checkpoint {args.checkpoint} não encontrado!")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    img_paths = list(pathlib.Path(args.images_dir).glob("*.png"))
    img_paths.sort(key=lambda f: int(re.sub(r'\D', '', f.name) or 0))

    for p in img_paths[:args.num_images]:
        img = plt.imread(str(p))
        if img.ndim == 3:
            img = img.mean(axis=2)

        detections = detect_digits(img, model, device, stride=2)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img, cmap='gray')

        for det in detections:
            rect = patches.Rectangle(
                (det["x"], det["y"]),
                28, 28,
                linewidth=2,
                edgecolor="#00FF00",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                det["x"],
                det["y"] - 5,
                f"{det['cls']} ({det['conf']:.2f})",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="green", alpha=0.6, edgecolor="none")
            )

        ax.axis("off")
        plt.title(f"Imagem: {p.name}")
        plt.show()

if __name__ == "__main__":
    main()
