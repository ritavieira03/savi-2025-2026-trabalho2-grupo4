#!/usr/bin/env python3
import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, sys
import warnings
import cv2
import random

## Ignorar avisos futuros
warnings.filterwarnings("ignore", category=FutureWarning)

## Definir a raiz do projeto e adicionar ao sys.path para importações locais
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Tarefa1.model import ModelBetterCNN 

## Utilitários de Avaliação

## Carrega as labels de um ficheiro CSV
def load_labels(label_path):
    labels = []
    if not os.path.exists(label_path): return labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            p = [x.strip() for x in line.split(',')]
            if len(p) == 5:
                labels.append({"digit": int(p[0]), "xmin": int(p[1]), "ymin": int(p[2]), "xmax": int(p[3]), "ymax": int(p[4])})
    return labels

## Calcula o IoU entre uma deteção e uma ground truth
def get_iou_eval(det, gt):
    x1, y1 = max(det["x"], gt["xmin"]), max(det["y"], gt["ymin"])
    x2 = min(det["x"] + det["size"], gt["xmax"])
    y2 = min(det["y"] + det["size"], gt["ymax"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_det = det["size"] ** 2
    area_gt = (gt["xmax"] - gt["xmin"]) * (gt["ymax"] - gt["ymin"])
    union = area_det + area_gt - inter
    return inter / union if union > 0 else 0

## Calcula o IoU entre duas boxes quadradas
def get_iou(box1, box2):
    x1, y1 = max(box1["x"], box2["x"]), max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["size"], box2["x"] + box2["size"])
    y2 = min(box1["y"] + box1["size"], box2["y"] + box2["size"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1["size"]**2 + box2["size"]**2 - inter
    return inter / union if union > 0 else 0

## Supressão de múltiplas deteções do mesmo objeto
def non_max_suppression(detections, iou_thresh=0.3):
    if not detections: return []
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if get_iou(best, d) < iou_thresh]
    return keep

## Supressão de deteções próximas umas das outras
def suppress_nearby(detections, min_distance=22):
    if not detections: return []
    detections = sorted(detections, key=lambda d: d["size"])
    keep = []
    for det in detections:
        x_c, y_c = det["x"] + det["size"]/2, det["y"] + det["size"]/2
        too_close = False
        for k in keep:
            k_xc, k_yc = k["x"] + k["size"]/2, k["y"] + k["size"]/2
            if ((x_c - k_xc)**2 + (y_c - k_yc)**2)**0.5 < min_distance:
                too_close = True; break
        if not too_close: keep.append(det)
    return keep

## Verifica se o crop tem margem preta ao redor
def has_black_margin(crop, margin):
    top, bottom = crop[:margin, :], crop[-margin:, :]
    left, right = crop[:, :margin], crop[:, -margin:]
    return np.all(top == 0) and np.all(bottom == 0) and np.all(left == 0) and np.all(right == 0)

## Função principal de deteção de objetos na imagem
def detect_objects(image, model, device, stride=2, batch_size=128):
    h, w = image.shape
    model.eval()
    if image.max() > 1.0: image = image / 255.0
    WINDOW_SIZES = [22, 26, 28, 32, 36]
    crops, meta = [], []
    ## Geração de janelas deslizantes
    for win in WINDOW_SIZES:
        for y in range(0, h - win + 1, stride):
            for x in range(0, w - win + 1, stride):
                crop = image[y:y+win, x:x+win]
                if crop.mean() < 0.05 or crop.max() < 0.3: continue
                if not has_black_margin(crop, 1): continue
                crops.append(cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA))
                meta.append((x, y, win))
    if not crops: return []
    crops_t = torch.tensor(np.array(crops), dtype=torch.float32).unsqueeze(1).to(device)
    detections = []
    ## Classificação em batch
    for i in range(0, len(crops_t), batch_size):
        batch = crops_t[i:i + batch_size]
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            scores, preds = probs.max(dim=1)
        for j in range(len(batch)):
            detections.append({"x": meta[i+j][0], "y": meta[i+j][1], "size": meta[i+j][2], "score": scores[j].item(), "pred": int(preds[j].item())})
    ## Aplicar NMS e supressão de deteções próximas
    detections = non_max_suppression(detections)
    return suppress_nearby(detections, min_distance=22)

## Função principal
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--images_dir", type=str, default="../Tarefa2/data/versaoD/test/images")
    parser.add_argument("-ld", "--labels_dir", type=str, default="../Tarefa2/data/versaoD/test/labels")
    parser.add_argument("-cp", "--checkpoint", type=str, default="../Tarefa1/experiments/best.pkl")
    parser.add_argument("-ns", "--num_show", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBetterCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    all_img_paths = list(pathlib.Path(args.images_dir).glob("*.png"))
    total_in_folder = len(all_img_paths)
    
    ## Selecionar 1% das imagens como amostra
    sample_size = max(1, int(total_in_folder * 0.01))
    img_paths = random.sample(all_img_paths, sample_size)

    print(f"A analisar amostra de {sample_size} imagens (1.0% das imagens da versão D)...")

    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt = 0

    ## Loop pelas imagens da amostra
    for idx, p in enumerate(img_paths):
        img = plt.imread(str(p))
        if img.ndim == 3: img = img.mean(axis=2)
        
        detections = detect_objects(img, model, device)
        gt_boxes = load_labels(os.path.join(args.labels_dir, p.stem + ".txt"))
        total_gt += len(gt_boxes)
        
        matched_gt = set()
        img_tp = 0
        ## Avaliação da deteção
        for det in detections:
            found = False
            for i, gt in enumerate(gt_boxes):
                if i not in matched_gt and get_iou_eval(det, gt) > 0.3 and det["pred"] == gt["digit"]:
                    img_tp += 1
                    matched_gt.add(i)
                    found = True; break
            if not found: total_fp += 1
        
        total_tp += img_tp
        total_fn += (len(gt_boxes) - img_tp)

        ## Visualizar algumas imagens
        if idx < args.num_show:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img, cmap="gray")
            for det in detections:
                rect = patches.Rectangle((det["x"], det["y"]), det["size"], det["size"], linewidth=2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)
                ax.text(det["x"], det["y"]-2, str(det["pred"]), color="red", fontsize=12, fontweight="bold")
            plt.title(f"Visualização: {p.name}")
            plt.show()

    ## Cálculo das métricas
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0

    ## Gráfico de resultado
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = ['Acertos (TP)', 'Falsos Positivos (FP)', 'Não Detetados (FN)']
    values = [total_tp, total_fp, total_fn]
    colors = ['#4CAF50', '#F44336', '#FF9800']

    bars = ax.bar(labels, values, color=colors)
    
    ## Título com métricas incluídas
    title_str = f"Resultado Global (Amostra 1%: {sample_size} imagens)\n"
    title_str += f"Precision: {precision:.2%} | Recall: {recall:.2%}"
    ax.set_title(title_str, fontsize=12, fontweight='bold', pad=20)
    
    ax.set_ylabel("Contagem Absoluta")
    ax.bar_label(bars, padding=3, fontweight='bold')

    plt.tight_layout()
    plt.savefig("analise_resultados.png")
    print(f"Processamento concluído. Gráfico guardado em 'analise_resultados.png'.")

## Executa main
if __name__ == "__main__":
    main()
