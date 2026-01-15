#!/usr/bin/env python3
# shebang line for linux / mac

"""
Tarefa 2 (ponto 5) — Análise e Visualização do dataset MNIST-Detection.

Este script:
  1) Cria mosaicos de imagens com as respetivas bounding boxes (ground-truth).
  2) Calcula e guarda estatísticas do dataset:
       - distribuição de classes
       - histograma do nº de dígitos por imagem
       - estatísticas de tamanho (w/h/área) das bboxes
       - tamanho médio dos dígitos (global e por classe)

Estrutura esperada do dataset:
  <dataset_dir>/images/<id>.png
  <dataset_dir>/labels/<id>.txt   (formato: "label, xmin, ymin, xmax, ymax")

Exemplos:
  python main_dataset_stats.py ./data/mnist_detection/train --outdir ./out_train
  python main_dataset_stats.py ./data/mnist_detection/test  --mosaic-num 25 --mosaic-cols 5
  python main_dataset_stats.py ./data/mnist_detection_A/train ./data/mnist_detection_B/train
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import typing as T

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import re


BOX_COLOR_OK = "tab:green"
BOX_COLOR_BAD = "tab:red"



def read_labels(label_path: pathlib.Path) -> T.Tuple[np.ndarray, np.ndarray, bool]:
    """
    Lê labels e bboxes no formato XYXY (xmin, ymin, xmax, ymax).

    Retorna:
      labels: (N,)
      bboxes_xyxy: (N,4)
      ok_strict: True se TODAS as linhas foram lidas no formato esperado ", "
                 False se foi preciso fallback (logo vamos pintar BB a vermelho)
    """
    assert label_path.is_file(), f"Label file not found: {label_path}"

    labels: T.List[int] = []
    bboxes_xyxy: T.List[T.List[int]] = []
    ok_strict = True

    with open(label_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    for line in lines[1:]:  # ignora cabeçalho
        line = line.strip()
        if not line:
            continue

        # 1) Tenta formato esperado: ", "
        parts = line.split(", ")
        if len(parts) != 5:
            # 2) fallback: aceita "," sem espaços / espaços estranhos
            parts = [p.strip() for p in line.split(",")]
            ok_strict = False

        if len(parts) != 5:
            raise ValueError(f"Linha inválida em {label_path}: {line}")

        label, xmin, ymin, xmax, ymax = [int(p) for p in parts]
        labels.append(label)
        bboxes_xyxy.append([xmin, ymin, xmax, ymax])

    return np.asarray(labels, dtype=np.int64), np.asarray(bboxes_xyxy, dtype=np.int64), ok_strict


def list_image_paths(dataset_dir: pathlib.Path) -> T.List[pathlib.Path]:
    image_dir = dataset_dir / "images"
    assert image_dir.is_dir(), f"Missing images dir: {image_dir}"

    paths = list(image_dir.glob("*.png"))
    try:
        paths.sort(key=lambda p: int(p.stem))  # ordena por id numérico
    except ValueError:
        paths.sort()
    return paths


def bbox_wh_area(bboxes_xyxy: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ## Devolver (w, h, area) de bboxes XYXY."""
    w = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]).astype(np.float32)
    h = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]).astype(np.float32)
    area = (w * h).astype(np.float32)
    return w, h, area


def collect_dataset_stats(dataset_dir: pathlib.Path, n_classes: int = 10) -> dict:
    """Varre o dataset e devolve estatísticas agregadas."""
    image_paths = list_image_paths(dataset_dir)
    label_dir = dataset_dir / "labels"
    assert label_dir.is_dir(), f"Missing labels dir: {label_dir}"

    class_counts = np.zeros((n_classes,), dtype=np.int64)
    digits_per_image: T.List[int] = []

    all_w: T.List[float] = []
    all_h: T.List[float] = []
    all_area: T.List[float] = []

    # por classe
    w_per_class: T.List[T.List[float]] = [[] for _ in range(n_classes)]
    h_per_class: T.List[T.List[float]] = [[] for _ in range(n_classes)]
    area_per_class: T.List[T.List[float]] = [[] for _ in range(n_classes)]

    missing_labels = 0
    for impath in image_paths:
        label_path = label_dir / f"{impath.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            continue

        labels, bboxes, _ = read_labels(label_path)
        digits_per_image.append(int(labels.shape[0]))

        for lab in labels.tolist():
            if 0 <= lab < n_classes:
                class_counts[lab] += 1

        if bboxes.size > 0:
            w, h, area = bbox_wh_area(bboxes)
            all_w.extend(w.tolist())
            all_h.extend(h.tolist())
            all_area.extend(area.tolist())

            for lab, wi, hi, ai in zip(labels.tolist(), w.tolist(), h.tolist(), area.tolist()):
                if 0 <= lab < n_classes:
                    w_per_class[lab].append(float(wi))
                    h_per_class[lab].append(float(hi))
                    area_per_class[lab].append(float(ai))

    digits_per_image_arr = np.asarray(digits_per_image, dtype=np.int64)
    all_w_arr = np.asarray(all_w, dtype=np.float32)
    all_h_arr = np.asarray(all_h, dtype=np.float32)
    all_area_arr = np.asarray(all_area, dtype=np.float32)

    def safe_mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

    def safe_median(x: np.ndarray) -> float:
        return float(np.median(x)) if x.size else float("nan")

    per_class_mean_area = [float(np.mean(area_per_class[c])) if area_per_class[c] else float("nan") for c in range(n_classes)]
    per_class_mean_w = [float(np.mean(w_per_class[c])) if w_per_class[c] else float("nan") for c in range(n_classes)]
    per_class_mean_h = [float(np.mean(h_per_class[c])) if h_per_class[c] else float("nan") for c in range(n_classes)]

    stats = {
        "dataset_dir": str(dataset_dir),
        "num_images": int(len(image_paths)),
        "missing_label_files": int(missing_labels),
        "num_digits_total": int(class_counts.sum()),
        "class_counts": class_counts.tolist(),
        "digits_per_image": {
            "mean": safe_mean(digits_per_image_arr.astype(np.float32)),
            "median": safe_median(digits_per_image_arr.astype(np.float32)),
            "min": int(digits_per_image_arr.min()) if digits_per_image_arr.size else 0,
            "max": int(digits_per_image_arr.max()) if digits_per_image_arr.size else 0,
        },
        "digits_per_image_hist": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(digits_per_image_arr, return_counts=True))
        }
        if digits_per_image_arr.size
        else {},
        "bbox": {
            "mean_w": safe_mean(all_w_arr),
            "mean_h": safe_mean(all_h_arr),
            "mean_area": safe_mean(all_area_arr),
            "median_w": safe_median(all_w_arr),
            "median_h": safe_median(all_h_arr),
            "median_area": safe_median(all_area_arr),
        },
        "bbox_mean_per_class": {
            "mean_w": per_class_mean_w,
            "mean_h": per_class_mean_h,
            "mean_area": per_class_mean_area,
        },
    }
    return stats


def draw_bboxes(
    ax: plt.Axes,
    bboxes_xyxy: np.ndarray,
    labels: np.ndarray,
    color: str,
) -> None:
    """Desenha bboxes + etiqueta no matplotlib Axes, todas com a mesma cor."""
    for (xmin, ymin, xmax, ymax), lab in zip(bboxes_xyxy.tolist(), labels.tolist()):
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=1.7,
            edgecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            max(0, ymin - 2),
            str(int(lab)),
            fontsize=8,
            color=color,
            bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none"),
        )



def save_mosaic(
    dataset_dir: pathlib.Path,
    out_path: pathlib.Path,
    n: int = 16,
    cols: int = 4,
    seed: int = 0,
    max_images: int | None = None,
    ):
    ## Guarda um mosaico com bounding boxes
    image_paths = list_image_paths(dataset_dir)
    if max_images is not None:
        image_paths = image_paths[: max_images]

    if not image_paths:
        raise RuntimeError(f"Não foram encontradas imagens em {dataset_dir}/images")

    rng = np.random.default_rng(seed)
    n = min(n, len(image_paths))
    chosen = rng.choice(image_paths, size=n, replace=False)

    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    label_dir = dataset_dir / "labels"
    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            ax.axis("off")
            continue

        impath = pathlib.Path(chosen[i])
        label_path = label_dir / f"{impath.stem}.txt"

        im = plt.imread(str(impath))
        ax.imshow(im, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{impath.stem}", fontsize=9)

        if label_path.is_file():
            try:
                labels, bboxes, ok_strict = read_labels(label_path)
                bb_color = BOX_COLOR_OK if ok_strict else BOX_COLOR_BAD
                draw_bboxes(ax, bboxes, labels, color=bb_color)
                ax.set_title(f"{impath.stem} ({len(labels)})", fontsize=9)
            except Exception:
                # Se houver erro a ler, marca a imagem com moldura vermelha
                h_img, w_img = im.shape[0], im.shape[1]
                ax.add_patch(
                    patches.Rectangle(
                        (0, 0),
                        w_img - 1,
                        h_img - 1,
                        fill=False,
                        linewidth=2.2,
                        edgecolor=BOX_COLOR_BAD,
                    )
                )
                ax.set_title(f"{impath.stem} (label ERROR)", fontsize=9, color=BOX_COLOR_BAD)
        else:
            ax.set_title(f"{impath.stem} (no labels)", fontsize=9, color=BOX_COLOR_BAD)


    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_class_distribution(stats_list: T.List[dict], out_path: pathlib.Path):
    ## Gráfico de barras: distribuição de classes
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(10)

    width = 0.8 / max(1, len(stats_list))
    for i, st in enumerate(stats_list):
        counts = np.asarray(st["class_counts"], dtype=np.int64)
        name = st.get("dataset_id") or pathlib.Path(st["dataset_dir"]).name
        ax.bar(x + (i - (len(stats_list) - 1) / 2) * width, counts, width=width, label=name)

    ax.set_xticks(x)
    ax.set_xlabel("Classe (dígito)")
    ax.set_ylabel("Contagem")
    ax.set_title("Distribuição de classes")
    if len(stats_list) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_digits_per_image_hist(stats: dict, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    hist = stats.get("digits_per_image_hist", {})
    if hist:
        ks = np.asarray([int(k) for k in hist.keys()], dtype=np.int64)
        vs = np.asarray([int(v) for v in hist.values()], dtype=np.int64)
        order = np.argsort(ks)
        ks, vs = ks[order], vs[order]
        ax.bar(ks, vs)
        ax.set_xlabel("Nº de dígitos na imagem")
        ax.set_ylabel("Nº de imagens")
        dpi = stats["digits_per_image"]
        ax.set_title("Histograma do nº de dígitos por imagem")
    else:
        ax.text(0.5, 0.5, "Sem dados para histograma.", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bbox_size_hist(dataset_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    """Histograma do tamanho (área) das bboxes (precisa de varrer o dataset)."""
    image_paths = list_image_paths(dataset_dir)
    label_dir = dataset_dir / "labels"
    areas: T.List[float] = []

    for impath in image_paths:
        lp = label_dir / f"{impath.stem}.txt"
        if not lp.is_file():
            continue
        _, bboxes, _ = read_labels(lp)
        if bboxes.size == 0:
            continue
        _, _, area = bbox_wh_area(bboxes)
        areas.extend(area.tolist())

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if areas:
        ax.hist(np.asarray(areas, dtype=np.float32), bins=30)
        ax.set_xlabel("Área da bbox (px²)")
        ax.set_ylabel("Frequência")
        ax.set_title("Histograma do tamanho dos dígitos (área das bboxes)")
    else:
        ax.text(0.5, 0.5, "Sem bboxes para calcular histograma.", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mean_bbox_area_per_class(stats: dict, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(10)
    mean_area = np.asarray(stats["bbox_mean_per_class"]["mean_area"], dtype=np.float32)
    ax.bar(x, mean_area)
    ax.set_xticks(x)
    ax.set_xlabel("Classe (dígito)")
    ax.set_ylabel("Área média (px²)")
    ax.set_title("Tamanho médio por classe (área da bbox)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _pretty_title_from_filename(p: pathlib.Path) -> str:
    name = p.stem.lower()

    mapping = {
        "digits_per_image_hist": "Nº de dígitos por imagem",
        "bbox_area_hist": "Área das bounding boxes",
        "mean_bbox_area_per_class": "Área média por classe",
        "class_distribution": "Distribuição de classes",
    }
    if name in mapping:
        return mapping[name]

    # fallback: converter snake_case -> Título
    name = re.sub(r"[_\-]+", " ", name).strip()
    return name[:1].upper() + name[1:]


def save_statistics_mosaic(
    png_paths: T.List[pathlib.Path],
    out_path: pathlib.Path,
    suptitle: str = "Estatísticas",
    ncols: int = 2,
) -> None:
    """
    Cria um mosaico (1 imagem) a partir de vários PNGs.
    Cada tile mostra o PNG e um título por cima (derivado do nome do ficheiro).
    """
    png_paths = [p for p in png_paths if p.is_file()]
    if not png_paths:
        return

    n = len(png_paths)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows))
    axes = np.asarray(axes).reshape(nrows, ncols)

    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            ax.axis("off")
            continue

        p = png_paths[i]
        img = plt.imread(str(p))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(_pretty_title_from_filename(p), fontsize=12)

    fig.suptitle(suptitle, fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Análise e visualização do dataset MNIST-Detection")
    parser.add_argument(
        "directories",
        nargs="*",
        default=["./data/mnist_detection/test/"],
        help="Escreva o caminho da pasta das imagens que deseja analisar (images/ e labels/).",
    )
    parser.add_argument("-od", "--outdir", default="./dataset_stats_out", help="Caminho da pasta onde guardar os resultados")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Seed para amostragem do mosaico")
    parser.add_argument("-mn", "--mosaic-num", type=int, default=16, help="Nº de imagens no mosaico")
    parser.add_argument("-mc", "--mosaic-cols", type=int, default=4, help="Nº de colunas no mosaico")
    parser.add_argument("-mp", "--mosaic-pages", type=int, default=1, help="(IGNORADO para não criar mais ficheiros)")
    parser.add_argument("-mi", "--max-images", type=int, default=None, help="Limita o nº de imagens lidas (útil para testes rápidos)")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Dataset "principal": o 1º (usado para mosaic_with_bboxes + stats plots do estatisticas.png)
    primary_dataset_dir = pathlib.Path(args.directories[0])
    primary_dataset_id = (
        f"{primary_dataset_dir.parent.name}_{primary_dataset_dir.name}"
        if primary_dataset_dir.parent.name
        else primary_dataset_dir.name
    )

    stats_list: T.List[dict] = []
    primary_stats: dict | None = None

    # --- calcular stats de todos os datasets (se vierem vários) ---
    for d in args.directories:
        dataset_dir = pathlib.Path(d)
        dataset_id = f"{dataset_dir.parent.name}_{dataset_dir.name}" if dataset_dir.parent.name else dataset_dir.name

        stats = collect_dataset_stats(dataset_dir)
        stats["dataset_id"] = dataset_id
        stats_list.append(stats)

        if dataset_dir.resolve() == primary_dataset_dir.resolve():
            primary_stats = stats

        print(f"\nDataset: {dataset_dir}")
        print(f" - Nº total de imagens analisadas: {stats['num_images']}  |  Labels em falta: {stats['missing_label_files']}")
        print(f" - Nº total de digitos criados: {stats['num_digits_total']}")
        print(f"  Outputs: {outdir}")

    # --- guardar JSON global (um único ficheiro) ---
    summary = {
        "primary_dataset": str(primary_dataset_dir),
        "datasets": stats_list,
    }
    with open(outdir / "stats_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    # --- 1) mosaic_with_bboxes.png (apenas do dataset principal) ---
    save_mosaic(
        primary_dataset_dir,
        outdir / "mosaic_with_bboxes.png",
        n=args.mosaic_num,
        cols=args.mosaic_cols,
        seed=args.seed,
        max_images=args.max_images,
    )

    # --- 2) class_distribution.png (GLOBAL) ---
    plot_class_distribution(stats_list, outdir / "class_distribution.png")

    # --- 3) estatisticas.png (GLOBAL), SEM tempfile ---
    if primary_stats is None:
        primary_stats = stats_list[0]  # fallback seguro

    # Pasta fixa para guardar os gráficos intermédios (não são apagados)
    tmpdir = outdir / "estatisticas_graficos"
    tmpdir.mkdir(parents=True, exist_ok=True)

    p1 = tmpdir / "digits_per_image_hist.png"
    p2 = tmpdir / "bbox_area_hist.png"
    p3 = tmpdir / "mean_bbox_area_per_class.png"

    plot_digits_per_image_hist(primary_stats, p1)
    plot_bbox_size_hist(primary_dataset_dir, p2)
    plot_mean_bbox_area_per_class(primary_stats, p3)

    stat_pngs = [
        p1,
        p2,
        p3,
        tmpdir / "class_distribution.png",
    ]
    save_statistics_mosaic(
        stat_pngs,
        outdir / "estatisticas.png",
        suptitle="Estatísticas",
        ncols=2,
    )




if __name__ == "__main__":
    main()
