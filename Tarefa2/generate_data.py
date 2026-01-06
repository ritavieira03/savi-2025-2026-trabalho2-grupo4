#!/usr/bin/env python3
# shebang line for linux / mac

import argparse
import pathlib

import cv2
import numpy as np
import tqdm

from torchvision.datasets import MNIST as TorchMNIST
from torchvision.datasets.mnist import MNIST as TorchMNISTClass


def load_mnist_numpy(root: str, train: bool):
    
    # Fix do URL (enunciado)
    TorchMNISTClass.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

    ds = TorchMNIST(root=root, train=train, download=True)
    X = ds.data.numpy().astype(np.uint8)
    Y = ds.targets.numpy().astype(np.uint8)
    return X, Y


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): [xmin, ymin, xmax, ymax]
    Returns:
        float: IoU
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box

    # no overlap
    if (x2_t <= x1_p) or (x2_p <= x1_t) or (y2_t <= y1_p) or (y2_p <= y1_t):
        return 0.0

    # intersection
    x1i = max(x1_t, x1_p)
    y1i = max(y1_t, y1_p)
    x2i = min(x2_t, x2_p)
    y2i = min(y2_t, y2_p)
    intersection = max(0.0, x2i - x1i) * max(0.0, y2i - y1i)

    # union
    pred_area = max(0.0, x2_p - x1_p) * max(0.0, y2_p - y1_p)
    gt_area = max(0.0, x2_t - x1_t) * max(0.0, y2_t - y1_t)
    union = pred_area + gt_area - intersection

    if union <= 0:
        return 0.0

    iou = intersection / union
    iou = float(np.clip(iou, 0.0, 1.0))
    return iou


def compute_iou_all(bbox, all_bboxes):
    ious = [0.0]
    for other_bbox in all_bboxes:
        ious.append(calculate_iou(bbox, other_bbox))
    return ious


def tight_bbox(digit, orig_bbox):
    """
    Ajusta a bbox para ficar "apertada" ao conteúdo não-zero do dígito (após resize).
    """
    xmin, ymin, xmax, ymax = orig_bbox

    # xmin
    shift = 0
    for i in range(digit.shape[1]):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmin += shift

    # xmax
    shift = 0
    for i in range(-1, -digit.shape[1], -1):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmax -= shift

    # ymin
    shift = 0
    for i in range(digit.shape[0]):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymin += shift

    # ymax
    shift = 0
    for i in range(-1, -digit.shape[0], -1):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymax -= shift

    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def dataset_exists(dirpath: pathlib.Path, num_images: int):
    if not dirpath.is_dir():
        return False
    for image_id in range(num_images):
        error_msg = f"MNIST dataset already generated in {dirpath}, \n\tbut did not find filepath:"
        error_msg2 = f"You can delete the directory by running: rm -r {dirpath.parent}"

        impath = dirpath.joinpath("images", f"{image_id}.png")
        assert impath.is_file(), f"{error_msg} {impath} \n\t{error_msg2}"

        label_path = dirpath.joinpath("labels", f"{image_id}.txt")
        assert label_path.is_file(), f"{error_msg} {label_path} \n\t{error_msg2}"

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
    """
    Gera imagens (imsize x imsize) com até max_digits_per_image dígitos (>=1),
    guardando:
      - images/{id}.png
      - labels/{id}.txt  (csv: label,xmin,ymin,xmax,ymax)
    """
    if dataset_exists(dirpath, num_images):
        return

    if max_digits_per_image < 1:
        raise ValueError("--max-digits-per-image tem de ser >= 1")

    max_image_value = 255
    assert mnist_images.dtype == np.uint8

    image_dir = dirpath.joinpath("images")
    label_dir = dirpath.joinpath("labels")
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)

    for image_id in tqdm.trange(num_images, desc=f"Generating dataset, saving to: {dirpath}"):
        im = np.zeros((imsize, imsize), dtype=np.float32)
        labels = []
        bboxes = []

        # número de dígitos nesta imagem (garante pelo menos 1)
        n_digits = np.random.randint(1, max_digits_per_image + 1)

        for _ in range(n_digits):
            # sample posição/tamanho até não haver sobreposição
            while True:
                width = np.random.randint(min_digit_size, max_digit_size + 1)
                x0 = np.random.randint(0, imsize - width + 1)
                y0 = np.random.randint(0, imsize - width + 1)

                candidate = [x0, y0, x0 + width, y0 + width]
                ious = compute_iou_all(candidate, bboxes)

                # Sem sobreposição (IoU==0)
                if max(ious) == 0.0:
                    break

            digit_idx = np.random.randint(0, len(mnist_images))
            digit = mnist_images[digit_idx].astype(np.float32)  # (28,28)
            digit = cv2.resize(digit, (width, width), interpolation=cv2.INTER_LINEAR)

            label = int(mnist_labels[digit_idx])
            labels.append(label)

            assert im[y0 : y0 + width, x0 : x0 + width].shape == digit.shape, (
                f"imshape: {im[y0:y0+width, x0:x0+width].shape}, digit shape: {digit.shape}"
            )

            bbox = tight_bbox(digit, [x0, y0, x0 + width, y0 + width])
            bboxes.append(bbox)

            im[y0 : y0 + width, x0 : x0 + width] += digit
            im[im > max_image_value] = max_image_value

        image_target_path = image_dir.joinpath(f"{image_id}.png")
        label_target_path = label_dir.joinpath(f"{image_id}.txt")

        im_u8 = im.astype(np.uint8)
        cv2.imwrite(str(image_target_path), im_u8)

        with open(label_target_path, "w", encoding="utf-8") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for l, bbox in zip(labels, bboxes):
                fp.write(f"{l},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-path", default="data/mnist_detection")
    parser.add_argument("--mnist-root", default="data/mnist_raw")

    parser.add_argument("--imsize", default=300, type=int)
    parser.add_argument("--max-digit-size", default=100, type=int)
    parser.add_argument("--min-digit-size", default=15, type=int)

    parser.add_argument("--num-train-images", default=10000, type=int)
    parser.add_argument("--num-test-images", default=1000, type=int)

    parser.add_argument("--max-digits-per-image", default=5, type=int)

    args = parser.parse_args()

    X_train, Y_train = load_mnist_numpy(args.mnist_root, train=True)
    X_test, Y_test = load_mnist_numpy(args.mnist_root, train=False)

    for split, (X, Y) in zip(["train", "test"], [(X_train, Y_train), (X_test, Y_test)]):
        n = args.num_train_images if split == "train" else args.num_test_images
        generate_dataset(
            pathlib.Path(args.base_path, split),
            n,
            args.max_digit_size,
            args.min_digit_size,
            args.imsize,
            args.max_digits_per_image,
            X,
            Y,
        )
