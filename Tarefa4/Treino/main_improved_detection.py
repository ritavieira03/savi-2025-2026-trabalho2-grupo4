#!/usr/bin/env python3
#shebang line for linux / mac

import os
import argparse
import torch

from torch.utils.data import Subset
from dataset import MNISTDetectionDataset
from model import build_detector
from trainer import TrainerDetection


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-df", "--dataset_folder", type=str, default="../../Tarefa2/data/versaoA")
    parser.add_argument("-ep", "--experiment_path", type=str, default="./experiments_det")
    parser.add_argument("-ne", "--num_epochs", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    args = vars(parser.parse_args())
    os.makedirs(args["experiment_path"], exist_ok=True)

    print("dataset_folder =", args["dataset_folder"])
    print("experiment_path =", args["experiment_path"])

    train_ds = MNISTDetectionDataset(args["dataset_folder"], split="train", keep_empty=False)
    test_ds = MNISTDetectionDataset(args["dataset_folder"], split="test", keep_empty=False)

    g = torch.Generator().manual_seed(42)

    n_train = max(1, int(len(train_ds) * 0.1))
    n_test  = max(1, int(len(test_ds)  * 0.1))

    train_idx = torch.randperm(len(train_ds), generator=g)[:n_train].tolist()
    test_idx  = torch.randperm(len(test_ds),  generator=g)[:n_test].tolist()

    train_ds = Subset(train_ds, train_idx)
    test_ds  = Subset(test_ds,  test_idx)

    print("train len =", len(train_ds))
    print("test  len =", len(test_ds))

    model = build_detector(num_classes=11)

    trainer = TrainerDetection(model, train_ds, test_ds, args)
    trainer.train()


if __name__ == "__main__":
    main()
