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
import cv2

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================
# Caminho do projeto
# =========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Tarefa1.model import ModelBetterCNN

# =========================================
# Utilitários
# =========================================

def get_iou(box1, box2):
    x1, y1 = max(box1["x"], box2["x"]), max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["size"], box2["x"] + box2["size"])
    y2 = min(box1["y"] + box1["size"], box2["y"] + box2["size"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1["size"] ** 2
    area2 = box2["size"] ** 2
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

def non_max_suppression(detections, iou_thresh=0.3):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if get_iou(best, d) < iou_thresh
        ]
    return keep

# =========================================
# Supressão de BBs maiores próximas de BBs menores
# =========================================
def suppress_nearby(detections, min_distance=22):
    """
    Remove deteções maiores se houver outra mais pequena próxima.
    """
    if not detections:
        return []

    # Ordenar por tamanho crescente (detecções pequenas primeiro)
    detections = sorted(detections, key=lambda d: d["size"])
    keep = []

    for det in detections:
        x_c = det["x"] + det["size"] / 2
        y_c = det["y"] + det["size"] / 2
        too_close = False

        for k in keep:
            k_xc = k["x"] + k["size"] / 2
            k_yc = k["y"] + k["size"] / 2
            dist = ((x_c - k_xc) ** 2 + (y_c - k_yc) ** 2) ** 0.5
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            keep.append(det)

    return keep

# =========================================
# Margem preta obrigatória
# =========================================
def has_black_margin(crop, margin):
    top = crop[:margin, :]
    bottom = crop[-margin:, :]
    left = crop[:, :margin]
    right = crop[:, -margin:]

    # Exige pixels completamente pretos
    return (
        np.all(top == 0) and
        np.all(bottom == 0) and
        np.all(left == 0) and
        np.all(right == 0)
    )

# =========================================
# Sliding Window Multi-Escala com classificação
# =========================================
def detect_objects(image, model, device, stride=2, batch_size=128):
    h, w = image.shape
    model.eval()

    # Normalizar se necessário
    if image.max() > 1.0:
        image = image / 255.0

    WINDOW_SIZES = [22, 26, 28, 32, 36]
    crops, meta = [], []

    for win in WINDOW_SIZES:
        margin = 1  # margem mínima de 1 pixel

        for y in range(0, h - win + 1, stride):
            for x in range(0, w - win + 1, stride):
                crop = image[y:y + win, x:x + win]

                # descarta fundo óbvio
                if crop.mean() < 0.05 or crop.max() < 0.3:
                    continue

                # margem preta obrigatória
                if not has_black_margin(crop, margin):
                    continue

                # redimensionar para a rede
                crop_resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
                crops.append(crop_resized)
                meta.append((x, y, win))

    if not crops:
        return []

    crops_t = torch.tensor(np.array(crops), dtype=torch.float32).unsqueeze(1).to(device)
    detections = []

    for i in range(0, len(crops_t), batch_size):
        batch = crops_t[i:i + batch_size]
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            scores, preds = probs.max(dim=1)  # score máximo e classe prevista

        for j in range(len(batch)):
            x, y, size = meta[i + j]
            detections.append({
                "x": x,
                "y": y,
                "size": size,
                "score": scores[j].item(),
                "pred": int(preds[j].item())
            })

    # Aplicar supressões
    detections = non_max_suppression(detections)
    detections = suppress_nearby(detections, min_distance=22)
    return detections

# =========================================
# Execução
# =========================================
def main():
    parser = argparse.ArgumentParser(
        description="Tarefa 3 — Deteção e identificação de dígitos por Sliding Window"
    )
    parser.add_argument("--images_dir", type=str, default="../Tarefa2/data/versaoD/test/images")
    parser.add_argument("--checkpoint", type=str, default="../Tarefa1/experiments/best.pkl")
    parser.add_argument("--num_images", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBetterCNN().to(device)

    # Carregar checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Listar imagens
    img_paths = list(pathlib.Path(args.images_dir).glob("*.png"))
    img_paths.sort(key=lambda f: int(re.sub(r'\D', '', f.name) or 0))

    for p in img_paths[:args.num_images]:
        img = plt.imread(str(p))
        if img.ndim == 3:
            img = img.mean(axis=2)  # converter para grayscale

        detections = detect_objects(img, model, device)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")

        for det in detections:
            rect = patches.Rectangle(
                (det["x"], det["y"]),
                det["size"],
                det["size"],
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )
            ax.add_patch(rect)

            # mostrar o número detetado
            ax.text(det["x"], det["y"] - 1, str(det["pred"]),
                    color="red", fontsize=14, fontweight="bold")

        ax.axis("off")
        plt.title(f"Imagem: {p.name}")
        plt.show()

if __name__ == "__main__":
    main()
