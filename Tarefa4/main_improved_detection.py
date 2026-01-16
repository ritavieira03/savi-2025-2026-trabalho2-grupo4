#!/usr/bin/env python3

import os
import sys
import json
import time
import random
import argparse
import pathlib
import importlib.util
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Robustez: se torchinfo não existir, injeta módulo dummy (model.py pode fazer "from torchinfo import summary")
# -----------------------------------------------------------------------------
def _ensure_torchinfo():
    try:
        import torchinfo  # noqa: F401
    except Exception:
        import types
        m = types.ModuleType("torchinfo")

        def summary(*args, **kwargs):  # noqa: ANN001
            return None

        m.summary = summary
        sys.modules["torchinfo"] = m


_ensure_torchinfo()


# -----------------------------------------------------------------------------
# Import do ModelBetterCNN por caminho (sem depender de packages)
# -----------------------------------------------------------------------------
def import_modelbettercnn(model_py: str = ""):
    here = pathlib.Path(__file__).resolve()

    candidates: List[pathlib.Path] = []
    if model_py:
        candidates.append(pathlib.Path(model_py).expanduser().resolve())

    candidates += [
        here.parent / "model.py",
        here.parent.parent / "model.py",
        here.parent.parent / "Tarefa1" / "model.py",
        here.parent.parent / "tarefa1" / "model.py",
    ]

    rels = ["model.py", "Tarefa1/model.py", "tarefa1/model.py"]
    for p in list(here.parents)[:6]:
        for r in rels:
            candidates.append((p / r).resolve())

    model_path = None
    for c in candidates:
        if c.is_file() and c.name == "model.py":
            model_path = c
            break

    if model_path is None:
        msg = "Não encontrei model.py. Tentei, por ex.:\n" + "\n".join(str(c) for c in candidates[:12])
        raise FileNotFoundError(msg)

    spec = importlib.util.spec_from_file_location("external_model_module", str(model_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Falha ao criar spec para: {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "ModelBetterCNN"):
        raise ImportError(f"O ficheiro {model_path} não tem ModelBetterCNN.")

    print(f"[i] Model carregado de: {model_path}")
    return module.ModelBetterCNN


# -----------------------------------------------------------------------------
# Utils dataset
# -----------------------------------------------------------------------------
def resolve_split_dirs(base_path: str, split: str):
    """
    Resolve diretórios do split, tolerando variações:
      train/training/Train...
      test/testing/Test...
      images/imgs/img...
      labels/annotations/ann...
    """
    base_root = pathlib.Path(base_path).expanduser().resolve()

    split_candidates = [split]
    if split == "train":
        split_candidates += ["training", "Train", "TRAIN"]
    if split == "test":
        split_candidates += ["testing", "Test", "TEST"]

    chosen_split = None
    for s in split_candidates:
        if (base_root / s).is_dir():
            chosen_split = s
            break
    if chosen_split is None:
        chosen_split = split

    base = base_root / chosen_split

    images_candidates = ["images", "Images", "imgs", "img"]
    labels_candidates = ["labels", "Labels", "label", "annotations", "ann", "annotation"]

    images_dir = None
    for d in images_candidates:
        p = base / d
        if p.is_dir():
            images_dir = p
            break

    labels_dir = None
    for d in labels_candidates:
        p = base / d
        if p.is_dir():
            labels_dir = p
            break

    return base, images_dir, labels_dir


def list_images(images_dir: pathlib.Path) -> List[pathlib.Path]:
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    paths: List[pathlib.Path] = []
    for e in exts:
        paths += list(images_dir.glob(e))
    paths = sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return paths


# -----------------------------------------------------------------------------
# Leitura labels do generate_data.py:  label, xmin, ymin, xmax, ymax
# -----------------------------------------------------------------------------
def read_labels_txt(label_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    labels: List[int] = []
    boxes: List[List[int]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:  # salta cabeçalho
        line = line.strip()
        if not line:
            continue
        parts = line.split(", ")
        if len(parts) != 5:
            parts = [p.strip() for p in line.split(",")]
        lab, xmin, ymin, xmax, ymax = map(int, parts)
        labels.append(lab)
        boxes.append([xmin, ymin, xmax, ymax])

    return np.asarray(labels, np.int64), np.asarray(boxes, np.int64)


def iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


# -----------------------------------------------------------------------------
# Dataset: crops 28x28 (dígitos) + crops de fundo (bg)
#   - suporta subset explícito de imagens via img_paths_list
# -----------------------------------------------------------------------------
class SceneCropsDataset(Dataset):
    """
    Produz amostras 28x28 para treino de classificador:
      - classes 0..9 : dígitos
      - classe 10    : fundo (background)
    """

    def __init__(
        self,
        base_path: str,
        split: str = "train",
        imsize: int = 128,
        out_size: int = 28,
        bg_class: int = 10,
        bg_per_digit: float = 1.0,
        bg_minmax: Tuple[int, int] = (22, 36),
        pad: int = 2,
        seed: int = 0,
        img_paths_list: Optional[List[pathlib.Path]] = None,
    ):
        self.imsize = imsize
        self.out_size = out_size
        self.bg_class = bg_class
        self.bg_per_digit = bg_per_digit
        self.bg_minmax = bg_minmax
        self.pad = pad
        self.rng = random.Random(seed)

        self.base, self.images_dir, self.labels_dir = resolve_split_dirs(base_path, split)

        self.img_paths: List[str] = []
        self.samples: List[Tuple[int, Tuple[int, int, int, int], int]] = []

        if self.images_dir is None or self.labels_dir is None:
            return

        if img_paths_list is None:
            img_paths = list_images(self.images_dir)
        else:
            img_paths = img_paths_list

        self.img_paths = [str(p) for p in img_paths]

        for idx, ip_str in enumerate(self.img_paths):
            ip = pathlib.Path(ip_str)

            # label com o mesmo stem
            lp = self.labels_dir / f"{ip.stem}.txt"
            # fallback: 00012.png -> 12.txt
            if not lp.is_file() and ip.stem.isdigit():
                lp2 = self.labels_dir / f"{int(ip.stem)}.txt"
                if lp2.is_file():
                    lp = lp2

            if not lp.is_file():
                continue

            labs, boxes = read_labels_txt(lp)

            # dígitos
            for lab, box in zip(labs.tolist(), boxes.tolist()):
                self.samples.append((idx, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), int(lab)))

            # fundo: crops sem overlap com bboxes
            n_digits = int(len(boxes))
            n_bg = int(round(self.bg_per_digit * max(1, n_digits)))

            for _ in range(n_bg):
                for _attempt in range(200):
                    s = self.rng.randint(self.bg_minmax[0], self.bg_minmax[1])
                    x1 = self.rng.randint(0, self.imsize - s)
                    y1 = self.rng.randint(0, self.imsize - s)
                    cand = [x1, y1, x1 + s, y1 + s]

                    if boxes.size == 0:
                        self.samples.append((idx, (cand[0], cand[1], cand[2], cand[3]), self.bg_class))
                        break

                    max_i = 0.0
                    for b in boxes.tolist():
                        max_i = max(
                            max_i,
                            iou_xyxy([cand[0], cand[1], cand[2], cand[3]], [b[0], b[1], b[2], b[3]]),
                        )
                    if max_i == 0.0:
                        self.samples.append((idx, (cand[0], cand[1], cand[2], cand[3]), self.bg_class))
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx, box, cls = self.samples[i]
        ip = self.img_paths[img_idx]

        img = Image.open(ip).convert("L")
        W, H = img.size

        x1, y1, x2, y2 = box
        if cls != self.bg_class and self.pad > 0:
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(W, x2 + self.pad)
            y2 = min(H, y2 + self.pad)

        crop = img.crop((x1, y1, x2, y2)).resize((self.out_size, self.out_size), resample=Image.BILINEAR)
        x = torch.from_numpy(np.array(crop, dtype=np.float32) / 255.0).unsqueeze(0)  # [1,28,28]
        y = torch.tensor(cls, dtype=torch.long)
        return x, y


# -----------------------------------------------------------------------------
# Patch do ModelBetterCNN para N classes (sem mexer no model.py)
# -----------------------------------------------------------------------------
def patch_model_to_n_classes(model: nn.Module, n_classes: int) -> nn.Module:
    if not hasattr(model, "classifier"):
        raise ValueError("Modelo não tem atributo 'classifier'.")

    clf = getattr(model, "classifier")
    if not isinstance(clf, nn.Sequential):
        raise ValueError("'classifier' não é nn.Sequential (não suportado neste script).")

    last_linear_idx = None
    last_linear = None
    for idx in reversed(range(len(clf))):
        if isinstance(clf[idx], nn.Linear):
            last_linear_idx = idx
            last_linear = clf[idx]
            break

    if last_linear_idx is None or last_linear is None:
        raise ValueError("Não encontrei nn.Linear no 'classifier'.")

    clf[last_linear_idx] = nn.Linear(last_linear.in_features, n_classes)
    return model


# -----------------------------------------------------------------------------
# FCN: conversão robusta do classificador para convolucional
# -----------------------------------------------------------------------------
def _find_linear(clf: nn.Sequential, in_f: int, out_f: int) -> nn.Linear:
    for m in clf:
        if isinstance(m, nn.Linear) and m.in_features == in_f and m.out_features == out_f:
            return m
    raise ValueError(f"Não encontrei Linear({in_f}->{out_f}) no classifier.")


def _find_bn1d(clf: nn.Sequential, n: int) -> nn.BatchNorm1d:
    for m in clf:
        if isinstance(m, nn.BatchNorm1d) and m.num_features == n:
            return m
    raise ValueError(f"Não encontrei BatchNorm1d({n}) no classifier.")


def _find_last_linear(clf: nn.Sequential, in_f: int, out_f: int) -> nn.Linear:
    for m in reversed(clf):
        if isinstance(m, nn.Linear) and m.in_features == in_f and m.out_features == out_f:
            return m
    raise ValueError(f"Não encontrei última Linear({in_f}->{out_f}) no classifier.")


class BetterCNN_FCN(nn.Module):
    def __init__(self, trained_model: nn.Module, num_classes: int = 11):
        super().__init__()
        if not hasattr(trained_model, "features") or not hasattr(trained_model, "classifier"):
            raise ValueError("Modelo não parece ser ModelBetterCNN (faltam 'features'/'classifier').")

        self.features = trained_model.features  # type: ignore
        clf: nn.Sequential = trained_model.classifier  # type: ignore

        fc1 = _find_linear(clf, in_f=64 * 7 * 7, out_f=256)
        bn1 = _find_bn1d(clf, n=256)
        fc2 = _find_last_linear(clf, in_f=256, out_f=num_classes)

        self.conv_fc1 = nn.Conv2d(64, 256, kernel_size=7, bias=True)
        self.bn2d = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.drop2d = nn.Dropout2d(p=0.5)
        self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)

        with torch.no_grad():
            self.conv_fc1.weight.copy_(fc1.weight.view(256, 64, 7, 7))
            self.conv_fc1.bias.copy_(fc1.bias)

            self.bn2d.weight.copy_(bn1.weight)
            self.bn2d.bias.copy_(bn1.bias)
            self.bn2d.running_mean.copy_(bn1.running_mean)
            self.bn2d.running_var.copy_(bn1.running_var)

            self.conv_out.weight[:, :, 0, 0].copy_(fc2.weight)
            self.conv_out.bias.copy_(fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.drop2d(self.relu(self.bn2d(self.conv_fc1(x))))
        x = self.conv_out(x)
        return x


# -----------------------------------------------------------------------------
# Métricas e gráficos
# -----------------------------------------------------------------------------
def compute_confusion_matrix(y_true: List[int], y_pred: List[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray):
    n = cm.shape[0]
    prec = np.zeros(n, dtype=np.float64)
    rec = np.zeros(n, dtype=np.float64)
    f1 = np.zeros(n, dtype=np.float64)

    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec[c] = tp / (tp + fp + 1e-9)
        rec[c] = tp / (tp + fn + 1e-9)
        f1[c] = 2 * prec[c] * rec[c] / (prec[c] + rec[c] + 1e-9)

    macro = {"precision": float(np.mean(prec)), "recall": float(np.mean(rec)), "f1": float(np.mean(f1))}
    return prec, rec, f1, macro


def plot_cm(cm: np.ndarray, class_names: List[str], out_path: str, title: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predito")
    plt.ylabel("GT")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------------------------------------------------------
# NMS (xyxy)
# -----------------------------------------------------------------------------
def nms(dets: List[Dict], iou_thr: float = 0.3) -> List[Dict]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep: List[Dict] = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        bx = [best["x1"], best["y1"], best["x2"], best["y2"]]
        remaining = []
        for d in dets:
            dx = [d["x1"], d["y1"], d["x2"], d["y2"]]
            if iou_xyxy(bx, dx) < iou_thr:
                remaining.append(d)
        dets = remaining

    return keep


# -----------------------------------------------------------------------------
# Deteção por FCN + multi-escala
# -----------------------------------------------------------------------------
@torch.no_grad()
def detect_fcn_multiscale(
    img01: np.ndarray,
    fcn_model: nn.Module,
    device: torch.device,
    bg_class: int = 10,
    scales: List[float] = None,
    score_thr: float = 0.6,
    nms_iou: float = 0.3,
) -> Tuple[List[Dict], float]:
    if scales is None:
        scales = [1.0]

    H0, W0 = img01.shape
    dets: List[Dict] = []

    t0 = time.perf_counter()

    for s in scales:
        if abs(s - 1.0) < 1e-9:
            img_s = img01
        else:
            newW = max(16, int(round(W0 * s)))
            newH = max(16, int(round(H0 * s)))
            img_s = np.array(
                Image.fromarray((img01 * 255).astype(np.uint8)).resize((newW, newH), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0

        x = torch.from_numpy(img_s).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        logits = fcn_model(x)[0]  # [C,Hout,Wout]
        probs = torch.softmax(logits, dim=0)  # [C,Hout,Wout]

        scores, preds = probs.max(dim=0)  # [Hout,Wout]
        scores_np = scores.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        # stride efetivo do ModelBetterCNN: 4 (2 maxpools stride=2)
        stride = 4
        win = 28  # patch equivalente ao treino

        Hout, Wout = preds_np.shape
        for i in range(Hout):
            for j in range(Wout):
                pred = int(preds_np[i, j])
                if pred == bg_class:
                    continue
                sc = float(scores_np[i, j])
                if sc < score_thr:
                    continue

                x1_s = j * stride
                y1_s = i * stride
                x2_s = x1_s + win
                y2_s = y1_s + win

                # coords na imagem original
                x1 = x1_s / s
                y1 = y1_s / s
                x2 = x2_s / s
                y2 = y2_s / s

                x1 = max(0.0, min(float(W0), x1))
                y1 = max(0.0, min(float(H0), y1))
                x2 = max(0.0, min(float(W0), x2))
                y2 = max(0.0, min(float(H0), y2))

                dets.append(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": sc, "pred": pred, "scale": float(s)}
                )

    dets = nms(dets, iou_thr=nms_iou)
    t1 = time.perf_counter()
    return dets, (t1 - t0)


def default_scales_for_minmax(min_size: int = 22, max_size: int = 36, ref: int = 28) -> List[float]:
    sizes = [min_size, 26, ref, 32, max_size]
    scales = [ref / s for s in sizes]
    scales = sorted(list({round(v, 3) for v in scales}))
    return scales


# -----------------------------------------------------------------------------
# Matching deteções vs GT (greedy por score)
# -----------------------------------------------------------------------------
def match_detections_to_gt(
    dets: List[Dict],
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.3,
) -> Tuple[int, int, int, List[float]]:
    used_gt = set()
    matched_ious: List[float] = []

    dets_sorted = sorted(dets, key=lambda d: d["score"], reverse=True)
    tp = fp = 0

    for d in dets_sorted:
        db = [d["x1"], d["y1"], d["x2"], d["y2"]]
        best_iou = 0.0
        best_j = -1

        for j, gb in enumerate(gt_boxes.tolist()):
            if j in used_gt:
                continue
            iouv = iou_xyxy(db, [gb[0], gb[1], gb[2], gb[3]])
            if iouv > best_iou:
                best_iou = iouv
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            if int(d["pred"]) == int(gt_labels[best_j]):
                tp += 1
                used_gt.add(best_j)
                matched_ious.append(best_iou)
            else:
                fp += 1
        else:
            fp += 1

    fn = int(len(gt_boxes) - len(used_gt))
    return tp, fp, fn, matched_ious


# -----------------------------------------------------------------------------
# Qualitativo
# -----------------------------------------------------------------------------
def save_qualitative_image(
    out_path: pathlib.Path,
    img01: np.ndarray,
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    dets: List[Dict],
    title: str,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img01, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    # GT branco
    for lab, b in zip(gt_labels.tolist(), gt_boxes.tolist()):
        x1, y1, x2, y2 = b
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.0, edgecolor="white"))
        ax.text(x1, max(0, y1 - 2), str(lab), color="white", fontsize=12, fontweight="bold")

    # Pred verde
    for d in dets:
        ax.add_patch(
            plt.Rectangle((d["x1"], d["y1"]), d["x2"] - d["x1"], d["y2"] - d["y1"],
                          fill=False, linewidth=2.0, edgecolor="lime")
        )
        ax.text(d["x1"], max(0, d["y1"] - 2), f"{d['pred']}:{d['score']:.2f}", color="lime", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="T4: Re-treino (bg) + FCN + deteção multi-escala")
    ap.add_argument("--data_base", type=str, default="../Tarefa2/data/versaoD", help="caminho para dados (generate_data.py)")
    ap.add_argument("--out_dir", type=str, default="experiments_improved_detection")
    ap.add_argument("--model_py", type=str, default="", help="caminho para model.py (ex: ../Tarefa1/model.py)")

    ap.add_argument("--imsize", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--bg_per_digit", type=float, default=1.0, help="crops bg por dígito (aprox.)")

    ap.add_argument("--num_qual", type=int, default=8)
    ap.add_argument("--score_thr", type=float, default=0.60)
    ap.add_argument("--nms_iou", type=float, default=0.30)
    ap.add_argument("--match_iou", type=float, default=0.30)

    ap.add_argument("--use_multiscale", action="store_true", help="ativa pirâmide de escalas (recomendado p/ 22..36)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] device: {device}")

    # -------------------------------------------------------------------------
    # ALTERAÇÃO PEDIDA: treino com 1% das imagens de TESTE
    # -------------------------------------------------------------------------
    base_test, test_images_dir, test_labels_dir = resolve_split_dirs(args.data_base, "test")
    if test_images_dir is None or test_labels_dir is None:
        print("\n[ERRO] Não encontrei split de teste com subpastas images/labels (ou equivalentes).")
        print(f"  base_test = {base_test}")
        print(f"  images_dir = {test_images_dir}")
        print(f"  labels_dir = {test_labels_dir}")
        return

    all_test_imgs = list_images(test_images_dir)
    if len(all_test_imgs) == 0:
        print("\n[ERRO] Não existem imagens no split de teste.")
        print(f"  test_images_dir = {test_images_dir}")
        return

    rng = random.Random(args.seed)
    rng.shuffle(all_test_imgs)

    n_train_imgs = max(1, int(round(0.01 * len(all_test_imgs))))
    train_imgs = all_test_imgs[:n_train_imgs]
    eval_imgs = all_test_imgs[n_train_imgs:] if len(all_test_imgs) > n_train_imgs else all_test_imgs[:]

    print(f"[i] total imagens TESTE: {len(all_test_imgs)}")
    print(f"[i] treino usa 1% do TESTE: {len(train_imgs)} imagens | avaliação: {len(eval_imgs)} imagens")

    # datasets a partir do split "test", mas com subsets diferentes
    train_ds = SceneCropsDataset(
        args.data_base,
        split="test",
        imsize=args.imsize,
        bg_per_digit=args.bg_per_digit,
        seed=args.seed,
        img_paths_list=train_imgs,
    )
    test_ds = SceneCropsDataset(
        args.data_base,
        split="test",
        imsize=args.imsize,
        bg_per_digit=args.bg_per_digit,
        seed=args.seed + 1,
        img_paths_list=eval_imgs,
    )

    if len(train_ds) == 0:
        print("\n[ERRO] train_ds tem 0 amostras (nenhum crop).")
        print("  Confirma que existem labels .txt a corresponder às imagens selecionadas.")
        print(f"  labels_dir = {train_ds.labels_dir}")
        return
    if len(test_ds) == 0:
        print("\n[ERRO] test_ds tem 0 amostras (nenhum crop).")
        print("  Confirma que existem labels .txt a corresponder às imagens selecionadas.")
        print(f"  labels_dir = {test_ds.labels_dir}")
        return

    print(f"[i] train crops: {len(train_ds)} | test crops: {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # modelo 11 classes (classificador)
    ModelBetterCNN = import_modelbettercnn(args.model_py)
    model = ModelBetterCNN()
    model = patch_model_to_n_classes(model, n_classes=11).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_acc": []}
    best_acc = -1.0

    y_true: List[int] = []
    y_pred: List[int] = []

    for ep in range(args.epochs):
        # -------------------------
        # treino (tqdm por época)
        # -------------------------
        model.train()
        losses = []
        t_ep0 = time.perf_counter()

        pbar = tqdm(train_dl, desc=f"Treino ep {ep+1}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        t_ep1 = time.perf_counter()

        # -------------------------
        # avaliação (tqdm por época)
        # -------------------------
        model.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        pbar_eval = tqdm(test_dl, desc=f"Eval   ep {ep+1}/{args.epochs}", unit="batch", dynamic_ncols=True)
        with torch.no_grad():
            for x, y in pbar_eval:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)

                correct += int((pred == y).sum().item())
                total += int(y.numel())

                y_true.extend(y.detach().cpu().tolist())
                y_pred.extend(pred.detach().cpu().tolist())

                acc_now = correct / max(1, total)
                pbar_eval.set_postfix(acc=f"{acc_now*100:.2f}%")

        train_loss = float(np.mean(losses)) if losses else 0.0
        acc = correct / max(1, total)

        history["train_loss"].append(train_loss)
        history["test_acc"].append(acc)

        print(f"\n[ep {ep+1}/{args.epochs}] train_loss={train_loss:.4f} | test_acc={acc*100:.2f}% | tempo_ep={(t_ep1-t_ep0):.1f}s")

        ckpt = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "history": history,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "checkpoint.pkl")
        if acc > best_acc:
            best_acc = acc
            torch.save(ckpt, out_dir / "best.pkl")

    print(f"\n[i] best test acc (classificação crops): {best_acc*100:.2f}%")

    # métricas classificação 11c
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=11)
    prec, rec, f1, macro = precision_recall_f1_from_cm(cm)

    class_names = [str(i) for i in range(10)] + ["bg"]
    plot_cm(cm, class_names, str(out_dir / "confusion_matrix_11c.png"), "Matriz de Confusão (11 classes)")

    results = {
        "accuracy": float(best_acc),
        "macro": macro,
        "per_class": {class_names[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])} for i in range(11)},
    }
    with open(out_dir / "metrics_11c.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_acc"], label="test_acc")
    plt.xlabel("época")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=160)
    plt.close()

    print("[i] guardado: confusion_matrix_11c.png, metrics_11c.json, training_curve.png")

    # -------------------------------------------------------------------------
    # FCN + deteção em cenas (qualitativo)
    # -------------------------------------------------------------------------
    fcn = BetterCNN_FCN(model, num_classes=11).to(device)
    fcn.eval()

    scales = default_scales_for_minmax(22, 36, 28) if args.use_multiscale else [1.0]
    print(f"[i] escalas deteção: {scales}")

    # para qualitativo de deteção, usa amostra das imagens de avaliação (eval_imgs)
    pick = rng.sample(eval_imgs, k=min(args.num_qual, len(eval_imgs)))

    total_tp = total_fp = total_fn = 0
    matched_ious_all: List[float] = []
    det_times: List[float] = []
    total_dets = 0

    for p in pick:
        img01 = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0

        lp = test_labels_dir / f"{p.stem}.txt"
        if not lp.is_file() and p.stem.isdigit():
            lp2 = test_labels_dir / f"{int(p.stem)}.txt"
            if lp2.is_file():
                lp = lp2
        if not lp.is_file():
            continue

        gt_labels, gt_boxes = read_labels_txt(lp)

        dets, dt = detect_fcn_multiscale(
            img01=img01,
            fcn_model=fcn,
            device=device,
            bg_class=10,
            scales=scales,
            score_thr=args.score_thr,
            nms_iou=args.nms_iou,
        )
        det_times.append(dt)
        total_dets += len(dets)

        tp, fp, fn, matched_ious = match_detections_to_gt(
            dets=dets, gt_labels=gt_labels, gt_boxes=gt_boxes, iou_thr=args.match_iou
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        matched_ious_all.extend(matched_ious)

        title = f"{p.name} | TP={tp} FP={fp} FN={fn} | dt={dt*1000:.1f}ms"
        save_qualitative_image(out_dir / f"qual_{p.stem}.png", img01, gt_labels, gt_boxes, dets, title)

    prec_det = total_tp / max(1, (total_tp + total_fp))
    rec_det = total_tp / max(1, (total_tp + total_fn))
    miou = float(np.mean(matched_ious_all)) if matched_ious_all else 0.0
    mean_dt = float(np.mean(det_times)) if det_times else 0.0

    det_summary = {
        "num_images_eval": len(pick),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": float(prec_det),
        "recall": float(rec_det),
        "mean_iou_matched": float(miou),
        "score_thr": float(args.score_thr),
        "nms_iou": float(args.nms_iou),
        "match_iou": float(args.match_iou),
        "scales": scales,
        "mean_detection_time_s": float(mean_dt),
        "mean_detection_time_ms": float(mean_dt * 1000.0),
        "avg_dets_per_image": float(total_dets / max(1, len(pick))),
        "device": str(device),
    }
    with open(out_dir / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(det_summary, f, indent=2, ensure_ascii=False)

    print("\n[i] Deteção (amostra qualitativa):")
    print(f"    precision={prec_det:.4f} | recall={rec_det:.4f} | mean IoU={miou:.4f}")
    print(f"    tempo médio por imagem: {mean_dt*1000:.1f} ms | dets/imagem: {det_summary['avg_dets_per_image']:.2f}")
    print(f"[i] guardado: qual_*.png e detection_summary.json em {out_dir}")


if __name__ == "__main__":
    main()
