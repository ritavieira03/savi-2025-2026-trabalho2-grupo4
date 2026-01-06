#!/usr/bin/env python3

import argparse
import pathlib
import cv2
import numpy as np
import tqdm

from torchvision.datasets import MNIST

def load_mnist_numpy(root: str, train: bool):
    """Carrega o MNIST garantindo o mirror correto via torchvision."""
    # Fix do URL para evitar erros de download dos servidores originais
    MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
    
    ds = MNIST(root=root, train=train, download=True)
    X = ds.data.numpy().astype(np.uint8)
    Y = ds.targets.numpy().astype(np.uint8)
    return X, Y

def calculate_iou(prediction_box, gt_box):
    """Calcula a Interseção sobre União (IoU)."""
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box

    # Sem sobreposição
    if (x2_t <= x1_p) or (x2_p <= x1_t) or (y2_t <= y1_p) or (y2_p <= y1_t):
        return 0.0

    x1i, y1i = max(x1_t, x1_p), max(y1_t, y1_p)
    x2i, y2i = min(x2_t, x2_p), min(y2_t, y2_p)
    
    intersection = max(0.0, x2i - x1i) * max(0.0, y2i - y1i)
    pred_area = max(0.0, x2_p - x1_p) * max(0.0, y2_p - y1_p)
    gt_area = max(0.0, x2_t - x1_t) * max(0.0, y2_t - y1_t)
    
    union = pred_area + gt_area - intersection
    return float(intersection / union) if union > 0 else 0.0

def compute_iou_all(bbox, all_bboxes):
    """Calcula IoU contra todos os bboxes existentes."""
    if not all_bboxes:
        return [0.0]
    return [calculate_iou(bbox, b) for b in all_bboxes]

def tight_bbox(digit, orig_bbox):
    """
    Ajusta a bbox para os limites reais dos píxeis não-vazios.
    Usa NumPy para maior eficiência (evita loops for em Python).
    """
    xmin_old, ymin_old, xmax_old, ymax_old = orig_bbox
    
    # Encontra coordenadas onde o dígito não é zero
    coords = np.argwhere(digit > 0)
    if coords.size == 0:
        return [int(c) for c in orig_bbox]

    # argwhere devolve [row, col] -> [y, x]
    y_min_rel, x_min_rel = coords.min(axis=0)
    y_max_rel, x_max_rel = coords.max(axis=0)

    return [
        int(xmin_old + x_min_rel),
        int(ymin_old + y_min_rel),
        int(xmin_old + x_max_rel + 1), # +1 para ser exclusivo no estilo OpenCV/NumPy
        int(ymin_old + y_max_rel + 1)
    ]

def check_dataset_health(dirpath: pathlib.Path, num_images: int):
    """Verifica se o dataset já existe sem interromper o programa abruptamente."""
    if not dirpath.is_dir():
        return False
    
    for image_id in range(num_images):
        impath = dirpath / "images" / f"{image_id}.png"
        label_path = dirpath / "labels" / f"{image_id}.txt"
        if not impath.is_file() or not label_path.is_file():
            print(f"(!) Dataset incompleto em {dirpath}. A regenerar...")
            return False
    return True

def generate_dataset(
    dirpath: pathlib.Path,
    num_images: int,
    max_digit_size: int,
    min_digit_size: int,
    imsize: int,
    max_digits_per_image: int,
    mnist_images: np.ndarray,
    mnist_labels: np.ndarray,
):
    if check_dataset_health(dirpath, num_images):
        print(f"[*] Dataset em {dirpath} já existe. Ignorar.")
        return

    image_dir = dirpath / "images"
    label_dir = dirpath / "labels"
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)

    for image_id in tqdm.trange(num_images, desc=f"Gerando {dirpath.name}"):
        # Canvas em float32 para cálculos, depois convertido
        im = np.zeros((imsize, imsize), dtype=np.float32)
        labels = []
        bboxes = []

        # Garante pelo menos 1 dígito
        n_digits = np.random.randint(1, max_digits_per_image + 1)

        for _ in range(n_digits):
            for _attempt in range(100): # Evita loop infinito se a imagem estiver cheia
                width = np.random.randint(min_digit_size, max_digit_size + 1)
                x0 = np.random.randint(0, imsize - width + 1)
                y0 = np.random.randint(0, imsize - width + 1)

                candidate = [x0, y0, x0 + width, y0 + width]
                ious = compute_iou_all(candidate, bboxes)

                if max(ious) == 0.0:
                    digit_idx = np.random.randint(0, len(mnist_images))
                    digit = mnist_images[digit_idx].astype(np.float32)
                    digit = cv2.resize(digit, (width, width), interpolation=cv2.INTER_LINEAR)

                    # Ajuste de bbox e inserção na imagem
                    bbox = tight_bbox(digit, [x0, y0, x0 + width, y0 + width])
                    
                    # Usamos np.maximum para evitar saturação excessiva em sobreposições mínimas
                    roi = im[y0:y0+width, x0:x0+width]
                    im[y0:y0+width, x0:x0+width] = np.maximum(roi, digit)

                    bboxes.append(bbox)
                    labels.append(int(mnist_labels[digit_idx]))
                    break

        # Finalização da imagem
        im = np.clip(im, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(image_dir / f"{image_id}.png"), im)
        
        with open(label_dir / f"{image_id}.txt", "w", encoding="utf-8") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for l, b in zip(labels, bboxes):
                fp.write(f"{l},{b[0]},{b[1]},{b[2]},{b[3]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Dataset MNIST Detection")
    parser.add_argument("--base-path", default="data/mnist_detection")
    parser.add_argument("--mnist-root", default="data/mnist_raw")
    parser.add_argument("--imsize", default=128, type=int)
    parser.add_argument("--max-digit-size", default=100, type=int)
    parser.add_argument("--min-digit-size", default=15, type=int)
    parser.add_argument("--num-train-images", default=10000, type=int)
    parser.add_argument("--num-test-images", default=1000, type=int)
    parser.add_argument("--max-digits-per-image", default=5, type=int)

    args = parser.parse_args()

    X_train, Y_train = load_mnist_numpy(args.mnist_root, train=True)
    X_test, Y_test = load_mnist_numpy(args.mnist_root, train=False)

    datasets = [
        ("train", X_train, Y_train, args.num_train_images),
        ("test", X_test, Y_test, args.num_test_images)
    ]

    for name, X, Y, n in datasets:
        generate_dataset(
            pathlib.Path(args.base_path) / name,
            n, args.max_digit_size, args.min_digit_size,
            args.imsize, args.max_digits_per_image, X, Y
        )