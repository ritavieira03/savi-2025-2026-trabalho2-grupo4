#!/usr/bin/env python3
# Tarefa 3 – Sliding Window Digit Detection (Completo)

import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Tarefa1.model import ModelBetterCNN

# ============================================================
# Utilitários
# ============================================================

def softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Calcula a entropia da distribuição softmax"""
    eps = 1e-8
    return -torch.sum(probs * torch.log(probs + eps), dim=1)

def iou(box1, box2):
    """Interseção sobre União (IoU) para NMS"""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + 28, box2["x"] + 28)
    y2 = min(box1["y"] + 28, box2["y"] + 28)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = 28*28*2 - inter
    return inter / union

def non_max_suppression(detections, iou_thresh=0.3):
    """Remove boxes sobrepostas mantendo a de maior confiança"""
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if iou(best, d) < iou_thresh]

    return keep

def sliding_window_detection(
    image: np.ndarray,
    model: torch.nn.Module,
    stride: int,
    conf_threshold: float,
    entropy_threshold: float,
    device: str
):
    """Aplica Sliding Window 28x28 sobre a imagem e retorna lista de deteções"""
    detections = []
    h, w = image.shape

    model.eval()

    for y in range(0, h - 28 + 1, stride):
        for x in range(0, w - 28 + 1, stride):
            crop = image[y:y + 28, x:x + 28]
            crop_t = torch.tensor(crop, dtype=torch.float32, device=device)
            crop_t = crop_t.unsqueeze(0).unsqueeze(0) / 255.0

            with torch.no_grad():
                logits = model(crop_t)
                probs = torch.softmax(logits, dim=1)
                entropy = softmax_entropy(probs)
                conf, cls = probs.max(dim=1)
                logit_energy = torch.max(torch.abs(logits))

            # Filtragem combinada
            if (conf.item() > conf_threshold and
                entropy.item() < entropy_threshold and
                logit_energy.item() > 3.0):
                detections.append({
                    "x": x,
                    "y": y,
                    "cls": int(cls.item()),
                    "conf": float(conf.item()),
                    "entropy": float(entropy.item())
                })

    # Remover deteções redundantes
    detections = non_max_suppression(detections)
    return detections

def draw_detections(image, detections, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    for d in detections:
        rect = patches.Rectangle(
            (d["x"], d["y"]), 28, 28,
            fill=False, edgecolor="lime", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            d["x"], max(0, d["y"] - 2),
            f"{d['cls']} ({d['conf']:.2f})",
            fontsize=8, color="lime",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none")
        )

    plt.tight_layout()
    plt.show()

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Tarefa 3 – Sliding Window Digit Detection")

    parser.add_argument("--images_dir", type=str,
                        default="../Tarefa2/data/mnist_detection/test/images",
                        help="Diretoria com imagens da Tarefa 2")
    parser.add_argument("--checkpoint", type=str, default="../Tarefa1/experiments/best.pkl",
                        help="Checkpoint best.pkl da Tarefa 1")
    parser.add_argument("--stride", type=int, default=4,
                        help="Stride da sliding window")
    parser.add_argument("--conf_threshold", type=float, default=0.84,
                        help="Threshold de confiança para considerar deteção")
    parser.add_argument("--entropy_threshold", type=float, default=2.0,
                        help="Threshold máximo de entropia para considerar deteção")
    parser.add_argument("--cpu", action="store_true", help="Forçar execução em CPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # --------------------------------------------------------
    # Carregar modelo
    # --------------------------------------------------------
    model = ModelBetterCNN().to(device)
    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint inválido: 'model_state_dict' não encontrado")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✔ Modelo carregado com sucesso")
    print(f"✔ Dispositivo: {device}")

    # --------------------------------------------------------
    # Processar imagens
    # --------------------------------------------------------
    images_dir = pathlib.Path(args.images_dir)
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"Sem imagens em {images_dir}")

    for img_path in image_paths:
        print(f"\nA processar imagem: {img_path.name}")
        image = plt.imread(str(img_path))
        if image.ndim == 3:
            image = image[..., 0]  # converter para grayscale

        detections = sliding_window_detection(
            image=image,
            model=model,
            stride=args.stride,
            conf_threshold=args.conf_threshold,
            entropy_threshold=args.entropy_threshold,
            device=device
        )

        print(f" - Nº de deteções após filtragem e NMS: {len(detections)}")
        draw_detections(image, detections, title=f"{img_path.name} | deteções: {len(detections)}")

if __name__ == "__main__":
    main()
