#!/usr/bin/env python3
# Tarefa 3 – Sliding Window Digit Detection Final Corrigida

import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, sys
import warnings
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Tarefa1.model import ModelBetterCNN

# ============================================================
# Utilitários
# ============================================================

def iou(box1, box2):
    """Interseção sobre União (IoU) para NMS"""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + 28, box2["x"] + 28)
    y2 = min(box1["y"] + 28, box2["y"] + 28)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = 28*28*2 - inter
    return inter / union

def non_max_suppression(detections, iou_thresh=0.2):
    """Remove boxes sobrepostas mantendo a de maior confiança"""
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if iou(best, d) < iou_thresh]

    return keep

# ============================================================
# Sliding Window Final Corrigida
# ============================================================

def sliding_window_final(
    image: np.ndarray,
    model: torch.nn.Module,
    stride: int = 1,
    device: str = "cpu",
    conf_threshold: float = 0.33,
    map_fraction: float = 0.58,
    gaussian_sigma: float = 2.0
):
    """
    Sliding window 28x28 com threshold adaptativo e pré-processamento MNIST
    """
    h, w = image.shape
    model.eval()
    crops_info = []

    # Normalização MNIST
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    # 1. Guardar crops com confiança mínima
    for y in range(0, h - 28 + 1, stride):
        for x in range(0, w - 28 + 1, stride):
            crop = image[y:y + 28, x:x + 28]
            crop_t = torch.tensor(crop, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            crop_t = (crop_t / 255.0 - MNIST_MEAN) / MNIST_STD

            with torch.no_grad():
                logits = model(crop_t)
                probs = torch.softmax(logits, dim=1)
                conf, cls = probs.max(dim=1)

            if conf.item() >= conf_threshold:
                crops_info.append({
                    "x": x,
                    "y": y,
                    "cls": int(cls.item()),
                    "conf": float(conf.item())
                })

    if not crops_info:
        return []

    # 2. Criar mapa de confiança acumulado
    conf_map = np.zeros((h, w), dtype=np.float32)
    for crop in crops_info:
        x, y = crop["x"], crop["y"]
        conf_map[y:y+28, x:x+28] += crop["conf"]

    # 3. Suavização do mapa para reforçar regiões consistentes
    conf_map = gaussian_filter(conf_map, sigma=gaussian_sigma)

    # 4. Threshold adaptativo baseado em fração do máximo do mapa
    threshold_value = map_fraction * conf_map.max()

    # 5. Filtrar crops pelo mapa suavizado
    detections = []
    for crop in crops_info:
        x, y = crop["x"], crop["y"]
        crop_mean_conf = conf_map[y:y+28, x:x+28].mean()
        if crop_mean_conf >= threshold_value:
            detections.append(crop)

    # 6. Aplicar NMS final
    detections = non_max_suppression(detections)
    return detections

# ============================================================
# Visualização
# ============================================================

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
    parser = argparse.ArgumentParser(description="Tarefa 3 – Sliding Window Digit Detection Final Corrigida")
    parser.add_argument("--images_dir", type=str,
                        default="../Tarefa2/data/mnist_detection/test/images",
                        help="Diretoria com imagens da Tarefa 2")
    parser.add_argument("--checkpoint", type=str, default="../Tarefa1/experiments/best.pkl",
                        help="Checkpoint best.pkl da Tarefa 1")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride da sliding window")
    parser.add_argument("--conf_threshold", type=float, default=0.33,
                        help="Confiança mínima do crop")
    parser.add_argument("--map_fraction", type=float, default=0.58,
                        help="Fração do máximo do mapa de confiança para threshold adaptativo")
    parser.add_argument("--gaussian_sigma", type=float, default=2.0,
                        help="Sigma do filtro gaussiano para suavizar mapa")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carregar modelo
    model = ModelBetterCNN().to(device)
    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # MUITO IMPORTANTE
    print("✔ Modelo carregado com sucesso")
    print(f"✔ Dispositivo: {device}")

    # Processar imagens
    images_dir = pathlib.Path(args.images_dir)
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"Sem imagens em {images_dir}")

    for img_path in image_paths:
        print(f"\nA processar imagem: {img_path.name}")
        image = plt.imread(str(img_path))
        if image.ndim == 3:
            image = image[..., 0]

        detections = sliding_window_final(
            image=image,
            model=model,
            stride=args.stride,
            device=device,
            conf_threshold=args.conf_threshold,
            map_fraction=args.map_fraction,
            gaussian_sigma=args.gaussian_sigma
        )

        print(f" - Nº de deteções após threshold adaptativo e NMS: {len(detections)}")
        draw_detections(image, detections, title=f"{img_path.name} | deteções: {len(detections)}")

if __name__ == "__main__":
    main()
