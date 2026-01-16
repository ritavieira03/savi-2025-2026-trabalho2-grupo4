#!/usr/bin/env python3
#shebang line for linux / mac

import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import detection_collate_fn


class TrainerDetection:
    def __init__(self, model, train_ds, test_ds, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args.get("num_workers", 2),
            collate_fn=detection_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=args.get("num_workers", 2),
            collate_fn=detection_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        self.optim = AdamW(
            self.model.parameters(),
            lr=args["lr"],
            weight_decay=args.get("weight_decay", 1e-4),
        )

        os.makedirs(args["experiment_path"], exist_ok=True)
        self.ckpt_path = os.path.join(args["experiment_path"], "detector.pt")

    def train(self):
        best = float("inf")

        for epoch in range(1, self.args["num_epochs"] + 1):
            self.model.train()
            running = 0.0

            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args['num_epochs']}"):
                images = [im.to(self.device) for im in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

                running += float(loss.item())

            train_loss = running / max(1, len(self.train_loader))
            test_loss = self.evaluate_loss()

            print(f"[epoch {epoch}] train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")

            if test_loss < best:
                best = test_loss
                torch.save({"model": self.model.state_dict(), "args": self.args}, self.ckpt_path)
                print(f"  -> guardado: {self.ckpt_path}")

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        # Para obter losses em detecção: modo train()
        self.model.train()
        running = 0.0

        for images, targets in self.test_loader:
            images = [im.to(self.device) for im in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())
            running += float(loss.item())

        return running / max(1, len(self.test_loader))
