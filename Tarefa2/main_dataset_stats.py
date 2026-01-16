#!/usr/bin/env python3
# shebang line for linux / mac


import argparse, json, math, pathlib, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


## Ir buscar os labels e bounding boxes do ficheiro das labels
def read_labels(caminho: pathlib.Path):
    labels, boxes, ok = [], [], True
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f.readlines()[1:]:
            linha = linha.strip()
            if not linha:
                continue
            valores = linha.split(", ")
            if len(valores) != 5:
                valores = [x.strip() for x in linha.split(",")]
                ok = False
                raise ValueError(f"Linha inválida em {caminho}: {linha}")
            lab, xmin, ymin, xmax, ymax = map(int, valores)
            labels.append(lab)
            boxes.append([xmin, ymin, xmax, ymax])
    return np.asarray(labels, np.int64), np.asarray(boxes, np.int64), ok


## Criar uma lista com os caminhos das imagens do dataset
def list_image_paths(dataset_path: pathlib.Path):
    img_path = dataset_path / "images"
    paths = list(img_path.glob("*.png"))
    try:
        paths.sort(key=lambda x: int(x.stem))
    except ValueError:
        paths.sort()
    return paths


## Calcular a área de uma bounding box
def bbox_area(b):
    w = (b[:, 2] - b[:, 0]).astype(np.float32)
    h = (b[:, 3] - b[:, 1]).astype(np.float32)
    return w, h, (w * h).astype(np.float32)


## Recolher estatísticas do dataset
def collect_dataset_stats(dataset_path: pathlib.Path, n_classes=10):
    imgs = list_image_paths(dataset_path)
    labels_path = dataset_path / "labels"
    class_counts = np.zeros(n_classes, np.int64)
    digits_per_image, all_w, all_h, all_area = [], [], [], []
    wpc, hpc, apc = [[] for _ in range(n_classes)], [[] for _ in range(n_classes)], [[] for _ in range(n_classes)]

    missing = 0
    for impath in imgs:
        lp = labels_path / f"{impath.stem}.txt"
        if not lp.is_file():
            missing += 1
            continue
        labels, bboxes, _ = read_labels(lp)
        digits_per_image.append(int(labels.size))
        for lab in labels.tolist():
            if 0 <= lab < n_classes:
                class_counts[lab] += 1
        if bboxes.size:
            w, h, area = bbox_area(bboxes)
            all_w += w.tolist()
            all_h += h.tolist()
            all_area += area.tolist()
            for lab, wi, hi, ai in zip(labels.tolist(), w.tolist(), h.tolist(), area.tolist()):
                if 0 <= lab < n_classes:
                    wpc[lab].append(float(wi))
                    hpc[lab].append(float(hi))
                    apc[lab].append(float(ai))

    dpi = np.asarray(digits_per_image, np.int64)
    all_w, all_h, all_area = map(lambda x: np.asarray(x, np.float32), (all_w, all_h, all_area))

    def m(x): return float(np.mean(x)) if x.size else float("nan")
    def md(x): return float(np.median(x)) if x.size else float("nan")

    return {
        "dataset_path": str(dataset_path),
        "num_images": int(len(imgs)),
        "missing_label_files": int(missing),
        "num_digits_total": int(class_counts.sum()),
        "class_counts": class_counts.tolist(),
        "digits_per_image_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(dpi, return_counts=True))} if dpi.size else {},
        "bbox": {
            "mean_w": m(all_w), "mean_h": m(all_h), "mean_area": m(all_area),
            "median_w": md(all_w), "median_h": md(all_h), "median_area": md(all_area),
        },
        "bbox_mean_per_class": {
            "mean_w": [float(np.mean(wpc[c])) if wpc[c] else float("nan") for c in range(n_classes)],
            "mean_h": [float(np.mean(hpc[c])) if hpc[c] else float("nan") for c in range(n_classes)],
            "mean_area": [float(np.mean(apc[c])) if apc[c] else float("nan") for c in range(n_classes)],
        },
    }


## Desenhar as bounding boxes numa imagem
def draw_bboxes(ax, bboxes, labels, color="#A8DCAB"):
    for (xmin, ymin, xmax, ymax), lab in zip(bboxes.tolist(), labels.tolist()):
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=1.7, edgecolor=color))
        ax.text(xmin, max(0, ymin - 2), str(int(lab)), fontsize=8, color=color,
                bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none"))


## Juntar várias imagens numa só imagem com as respetivas bounding boxes
def save_mosaic(dataset_path: pathlib.Path, out_path: pathlib.Path, n=16, cols=4, seed=0, max_images=None):
    imgs = list_image_paths(dataset_path)
    if max_images is not None:
        imgs = imgs[:max_images]
    if not imgs:
        raise RuntimeError(f"Sem imagens em {dataset_path}/images")

    rng = np.random.default_rng(seed)
    n = min(n, len(imgs))
    chosen = rng.choice(imgs, size=n, replace=False)

    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    labels_path = dataset_path / "labels"

    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            ax.axis("off"); continue
        impath = pathlib.Path(chosen[i])
        lp = labels_path / f"{impath.stem}.txt"
        im = plt.imread(str(impath))
        ax.imshow(im, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])

        if lp.is_file():
            try:
                labels, bboxes, _ = read_labels(lp)
                draw_bboxes(ax, bboxes, labels, "#A8DCAB")
                ax.set_title(f"{impath.stem} ({len(labels)})", fontsize=9)
            except Exception:
                h, w = im.shape[0], im.shape[1]
                ax.add_patch(patches.Rectangle((0, 0), w - 1, h - 1, fill=False, linewidth=2.2, edgecolor="#A8DCAB"))
                ax.set_title(f"{impath.stem} (label ERROR)", fontsize=9, color="red")
        else:
            ax.set_title(f"{impath.stem} (no labels)", fontsize=9, color="red")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

## Criar gráficos das várias estatísticas recolhidas
def plot_class_distribution(stats_list, out_path: pathlib.Path, n_classes=10):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(n_classes)
    width = 0.8 / max(1, len(stats_list))
    for i, st in enumerate(stats_list):
        counts = np.asarray(st["class_counts"], np.int64)
        name = st.get("dataset_id") or pathlib.Path(st["dataset_path"]).name
        ax.bar(x + (i - (len(stats_list) - 1) / 2) * width, counts, width=width, label=name)
    ax.set_xticks(x); ax.set_xlabel("Classe (dígito)"); ax.set_ylabel("Contagem"); ax.set_title("Distribuição de classes", fontsize=18)
    if len(stats_list) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_digits_per_image(stats, out_path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    hist = stats.get("digits_per_image_hist", {})
    if hist:
        ks = np.asarray([int(k) for k in hist.keys()], np.int64)
        vs = np.asarray([int(v) for v in hist.values()], np.int64)
        o = np.argsort(ks); ks, vs = ks[o], vs[o]
        ax.bar(ks, vs)
        ax.set_xlabel("Nº de dígitos na imagem"); ax.set_ylabel("Nº de imagens"); ax.set_title("Histograma do nº de dígitos por imagem", fontsize=16)
    else:
        ax.text(0.5, 0.5, "Sem dados para histograma.", ha="center", va="center"); ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_bbox_size(dataset_path: pathlib.Path, out_path: pathlib.Path):
    areas = []
    labels_path = dataset_path / "labels"
    for impath in list_image_paths(dataset_path):
        lp = labels_path / f"{impath.stem}.txt"
        if not lp.is_file():
            continue
        _, b, _ = read_labels(lp)
        if b.size:
            areas += bbox_area(b)[2].tolist()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if areas:
        ax.hist(np.asarray(areas, np.float32), bins=30)
        ax.set_xlabel("Área da bbox (px²)"); ax.set_ylabel("Frequência"); ax.set_title("Histograma do tamanho dos dígitos", fontsize=16)
    else:
        ax.text(0.5, 0.5, "Sem bboxes para calcular histograma.", ha="center", va="center"); ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_mean_bbox_area_per_class(stats, out_path: pathlib.Path, n_classes=10):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(n_classes)
    mean_area = np.asarray(stats["bbox_mean_per_class"]["mean_area"], np.float32)
    ax.bar(x, mean_area)
    ax.set_xticks(x); ax.set_xlabel("Classe (dígito)"); ax.set_ylabel("Área média (px²)"); ax.set_title("Tamanho médio por classe", fontsize=18)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


## Definir os títulos dos gráficos com base no nome do ficheiro
def define_title(caminho: pathlib.Path):
    m = {
        "digits_per_image_hist": "Nº de dígitos por imagem",
        "bbox_area_hist": "Área das bounding boxes",
        "mean_bbox_area_per_class": "Área média por classe",
        "class_distribution": "Distribuição de classes",
    }
    s = caminho.stem.lower()
    if s in m:
        return m[s]
    s = re.sub(r"[_\-]+", " ", s).strip()
    return s[:1].upper() + s[1:]


## Guardar as várias imagens PNG numa única imagem com subplots
def save_all_statistics(pngs, out_path: pathlib.Path, suptitle="Estatísticas", ncols=2):
    pngs = [caminho for caminho in pngs if caminho.is_file()]
    if not pngs:
        return
    n = len(pngs)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows))
    axes = np.asarray(axes).reshape(nrows, ncols)
    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            ax.axis("off"); continue
        img = plt.imread(str(pngs[i]))
        ax.imshow(img); ax.axis("off")
    fig.suptitle(suptitle, fontsize=25, y=0.90)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Análise e visualização do dataset MNIST-Detection")
    ap.add_argument("directories", nargs="*", default=["./data/mnist_detection/test/"])
    ap.add_argument("-op", "--out-path", default="./estatisticas_gerais/")
    ap.add_argument("-s", "--seed", type=int, default=0)
    ap.add_argument("-mn", "--mosaic-num", type=int, default=16)
    ap.add_argument("-mc", "--mosaic-cols", type=int, default=4)
    ap.add_argument("-mi", "--max-images", type=int, default=None)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out_path); out_path.mkdir(parents=True, exist_ok=True)
    primary = pathlib.Path(args.directories[0])

    stats_list, primary_stats = [], None
    for d in args.directories:
        dd = pathlib.Path(d)
        st = collect_dataset_stats(dd)
        st["dataset_id"] = f"{dd.parent.name}_{dd.name}" if dd.parent.name else dd.name
        stats_list.append(st)
        if dd.resolve() == primary.resolve():
            primary_stats = st
        print(f"\nDataset utilizado: {dd}")
        print(f"\n - Nº total de imagens analisadas: {st['num_images']}  |  Labels em falta: {st['missing_label_files']}")
        print(f" - Nº total de digitos criados: {st['num_digits_total']}")
        print(f"\nDocumentos gerados em: {out_path}\n")

    with open(out_path / "estatisticas.json", "w", encoding="utf-8") as f:
        json.dump({"primary_dataset": str(primary), "datasets": stats_list}, f, indent=2, ensure_ascii=False)

    save_mosaic(primary, out_path / "bboxes.png", n=args.mosaic_num, cols=args.mosaic_cols, seed=args.seed, max_images=args.max_images)

    class_png = out_path / "class_distribution.png"
    plot_class_distribution(stats_list, class_png)

    if primary_stats is None:
        primary_stats = stats_list[0]

    g = out_path / "estatisticas_graficos"; g.mkdir(parents=True, exist_ok=True)
    p1, p2, p3 = g / "digits_per_image.png", g / "bbox_area.png", g / "mean_bbox_area_per_class.png"
    plot_digits_per_image(primary_stats, p1)
    plot_bbox_size(primary, p2)
    plot_mean_bbox_area_per_class(primary_stats, p3)

    save_all_statistics([p1, p2, p3, class_png], out_path / "estatisticas.png", suptitle="Estatísticas Gerais", ncols=2)



if __name__ == "__main__":
    main()
