#!/usr/bin/env python3
# shebang line for linux / mac

import argparse
import pathlib
import time
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch

import sys
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SCENE_DIR = PROJECT_ROOT / "Tarefa2" / "data" / "mnist_detection" / "test"
DEFAULT_CKPT_PATH = PROJECT_ROOT / "Tarefa1" / "experiments" / "best.pkl"

from Tarefa1.model import ModelBetterCNN


# -------------------------
# Utils: IOU + NMS
# -------------------------
def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(dets: List[Dict], iou_thr: float) -> List[Dict]:
    # det: {"bbox":(x1,y1,x2,y2), "score":float, "cls":int}
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            if iou_xyxy(d["bbox"], k["bbox"]) >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


# -------------------------
# Sliding window + batching
# -------------------------
def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def make_windows(H: int, W: int, win: int, stride: int):
    for y in range(0, H - win + 1, stride):
        for x in range(0, W - win + 1, stride):
            yield x, y, x + win, y + win


def infer_windows(
    model: torch.nn.Module,
    device: torch.device,
    img01: np.ndarray,          # float32 in [0,1], shape (H,W)
    win_sizes: List[int],
    stride: int,
    batch_size: int,
    prob_thr: float,
    ent_thr: float,
    fg_mean_thr: float,
) -> List[Dict]:
    H, W = img01.shape
    dets: List[Dict] = []

    model.eval()

    for win in win_sizes:
        coords = []
        crops = []

        for (x1, y1, x2, y2) in make_windows(H, W, win, stride):
            patch = img01[y1:y2, x1:x2]

            # filtro "fundo": se for quase tudo preto, ignora
            if float(patch.mean()) < fg_mean_thr:
                continue

            # redimensiona para 28x28 porque o classificador foi treinado em MNIST
            patch28 = cv2.resize(patch, (28, 28), interpolation=cv2.INTER_LINEAR)

            coords.append((x1, y1, x2, y2))
            crops.append(patch28)

        if not crops:
            continue

        crops = np.stack(crops, axis=0).astype(np.float32)  # (N,28,28)
        # -> (N,1,28,28)
        crops_t = torch.from_numpy(crops[:, None, :, :]).to(device)

        with torch.no_grad():
            for i in range(0, len(coords), batch_size):
                bt = crops_t[i:i+batch_size]
                logits = model(bt)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

                for j, p in enumerate(probs):
                    cls = int(np.argmax(p))
                    score = float(np.max(p))
                    ent = entropy(p)

                    if score >= prob_thr and ent <= ent_thr:
                        bbox = coords[i + j]
                        dets.append({"bbox": bbox, "score": score, "cls": cls, "win": win, "ent": ent})

    return dets


def read_gt_labels(label_path: pathlib.Path):
    # formato do teu generate_data.py: primeira linha header; depois "label, xmin, ymin, xmax, ymax"
    labels, bboxes = [], []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
    for ln in lines:
        l, x1, y1, x2, y2 = [int(v) for v in ln.strip().split(", ")]
        labels.append(l)
        bboxes.append((x1, y1, x2, y2))
    return labels, bboxes


def draw_dets(
    img_gray_u8: np.ndarray,
    dets: List[Dict],
    out_path: pathlib.Path,
    draw_text: bool = True,
    gt: Tuple[List[int], List[Tuple[int,int,int,int]]] | None = None,
):
    # desenhar em BGR
    vis = cv2.cvtColor(img_gray_u8, cv2.COLOR_GRAY2BGR)

    # GT (opcional) em cinza claro
    if gt is not None:
        gt_labels, gt_bboxes = gt
        for l, (x1,y1,x2,y2) in zip(gt_labels, gt_bboxes):
            cv2.rectangle(vis, (x1,y1), (x2,y2), (200,200,200), 1)
            if draw_text:
                cv2.putText(vis, f"GT:{l}", (x1, max(10, y1-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)

    # deteções em branco
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 2)
        if draw_text:
            cv2.putText(vis, f"{d['cls']} {d['score']:.2f}", (x1, min(vis.shape[0]-5, y2+12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main():
    ap = argparse.ArgumentParser(description="Tarefa 3 - Sliding Window MNIST Detection")
    ap.add_argument("--scene-dir", type=str, default=str(DEFAULT_SCENE_DIR),
                    help="Pasta base com images/ e labels/")
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT_PATH),
                    help="Checkpoint do classificador (best.pkl)")
    ap.add_argument("--outdir", type=str, default="./sliding_out")
    ap.add_argument("--max-images", type=int, default=25)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--win-sizes", type=int, nargs="+", default=[22, 28, 36],
                    help="Tamanhos de janela (multi-escala). Ex: 22 28 36")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--prob-thr", type=float, default=0.99)
    ap.add_argument("--ent-thr", type=float, default=0.70)
    ap.add_argument("--fg-mean-thr", type=float, default=0.03,
                    help="Média mínima do patch (em [0,1]) para não ser 'fundo'")
    ap.add_argument("--nms-iou", type=float, default=0.30)
    ap.add_argument("--per-class-nms", action="store_true", help="Faz NMS por classe")
    ap.add_argument("--draw-gt", action="store_true", help="Desenha também GT (cinza)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo (tem de ser o mesmo que treinaste)
    model = ModelBetterCNN().to(device)

    ckpt_path = pathlib.Path(args.ckpt).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado em {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    base = pathlib.Path(args.scene_dir).expanduser()
    if not base.is_dir():
        raise FileNotFoundError(f"Pasta da cena não encontrada: {base}")

    img_dir = base / "images"
    lab_dir = base / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {img_dir}")
    outdir = pathlib.Path(args.outdir)

    img_paths = sorted(img_dir.glob("*.png"))
    if args.max_images is not None:
        img_paths = img_paths[:args.max_images]

    t0 = time.time()
    total_dets = 0

    for p in img_paths:
        img_u8 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
            continue

        img01 = (img_u8.astype(np.float32) / 255.0)

        dets = infer_windows(
            model=model,
            device=device,
            img01=img01,
            win_sizes=args.win_sizes,
            stride=args.stride,
            batch_size=args.batch,
            prob_thr=args.prob_thr,
            ent_thr=args.ent_thr,
            fg_mean_thr=args.fg_mean_thr,
        )

        # NMS (global ou por classe)
        if args.per_class_nms:
            byc: Dict[int, List[Dict]] = {}
            for d in dets:
                byc.setdefault(d["cls"], []).append(d)
            dets_nms = []
            for c, lst in byc.items():
                dets_nms.extend(nms(lst, args.nms_iou))
            dets = sorted(dets_nms, key=lambda d: d["score"], reverse=True)
        else:
            dets = nms(dets, args.nms_iou)

        total_dets += len(dets)

        gt = None
        if args.draw_gt:
            lp = lab_dir / f"{p.stem}.txt"
            if lp.exists():
                gt = read_gt_labels(lp)

        out_path = outdir / f"{p.stem}_detections.png"
        draw_dets(img_u8, dets, out_path, draw_text=True, gt=gt)

        print(f"{p.name}: {len(dets)} dets (depois NMS)")

    dt = time.time() - t0
    n = max(1, len(img_paths))
    print(f"\nDone. {n} imagens | {total_dets} dets | {dt:.2f}s total | {dt/n:.3f}s/imagem")
    print(f"Outputs em: {outdir.resolve()}")


if __name__ == "__main__":
    main()
