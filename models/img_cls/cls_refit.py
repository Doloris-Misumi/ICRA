#!/usr/bin/env python3
"""Refit stage 1 on the complete official 3D-LRF train split for the selected epoch count."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.img_cls.cls_model import ImageClsBackbone
from models.img_cls.cls_train import (
    ImgDataset,
    WEATHER_NAMES,
    cfg_from_yaml_file,
    set_seed,
)
from utils.split_contract import format_split_summary, verify_split_contract


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=202206)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    args.config = args.config.resolve()
    args.selection_dir = args.selection_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    output_is_nonempty = args.output_dir.exists() and any(args.output_dir.iterdir())
    if output_is_nonempty and not args.resume:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_state_path = args.output_dir / "last_state.pth"
    if args.resume and not last_state_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {last_state_path}")

    best_record = json.loads((args.selection_dir / "best_metrics.json").read_text(encoding="utf-8"))
    selected_epoch = int(best_record["epoch"])
    num_epochs = selected_epoch + 1
    if num_epochs <= 0:
        raise ValueError(f"invalid selected epoch count: {num_epochs}")

    set_seed(args.seed)
    config = cfg_from_yaml_file(args.config)
    split_summary = verify_split_contract(config, PROJECT_ROOT)
    print(format_split_summary(split_summary), flush=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    full_train = ImgDataset(config, split="official_train", transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        full_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )

    device = torch.device(args.device)
    model = ImageClsBackbone().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    source_paths = {
        "cls_refit.py": Path(__file__).resolve(),
        "cls_train.py": (PROJECT_ROOT / "models/img_cls/cls_train.py").resolve(),
        "cls_model.py": (PROJECT_ROOT / "models/img_cls/cls_model.py").resolve(),
        "split_contract.py": (PROJECT_ROOT / "utils/split_contract.py").resolve(),
        "config": args.config,
        "split_manifest": (
            PROJECT_ROOT / "resources/split/split_manifest_3dlrf_icra.json"
        ).resolve(),
        "selection_best_metrics": (args.selection_dir / "best_metrics.json").resolve(),
    }
    manifest = {
        "protocol": "stage-1 refit on complete official 3D-LRF train split",
        "selection_dir": str(args.selection_dir),
        "selection_metric": "internal_val_macro_f1",
        "selected_epoch_zero_based": selected_epoch,
        "refit_epochs": num_epochs,
        "official_train_samples": len(full_train),
        "official_train_class_counts": dict(
            Counter(WEATHER_NAMES[label] for _, label, _ in full_train.samples)
        ),
        "official_test_evaluations": 0,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "lr": args.lr,
        "momentum": args.momentum,
        "resumable_epoch_state": str(last_state_path),
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
    }
    manifest_path = args.output_dir / "refit_manifest.json"
    if args.resume:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "selection_dir", "selected_epoch_zero_based", "refit_epochs", "seed",
            "batch_size", "lr", "momentum",
        )
        mismatches = {
            key: {"existing": existing_manifest.get(key), "requested": manifest.get(key)}
            for key in invariant_keys
            if existing_manifest.get(key) != manifest.get(key)
        }
        if existing_manifest.get("source_sha256") != manifest.get("source_sha256"):
            mismatches["source_sha256"] = {
                "existing": existing_manifest.get("source_sha256"),
                "requested": manifest.get("source_sha256"),
            }
        if mismatches:
            raise ValueError(f"refit resume invariant mismatch: {mismatches}")
    else:
        shutil.copy2(args.config, args.output_dir / "config.yml")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    start_epoch = 0
    if args.resume:
        resume_state = torch.load(last_state_path, map_location=device)
        if int(resume_state["refit_epochs"]) != num_epochs:
            raise ValueError(
                f"resume refit_epochs={resume_state['refit_epochs']} != selected {num_epochs}"
            )
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        generator.set_state(resume_state["loader_generator_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        np.random.set_state(resume_state["numpy_rng_state"])
        random.setstate(resume_state["python_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        start_epoch = int(resume_state["next_epoch"])
        if start_epoch < 0 or start_epoch > num_epochs:
            raise ValueError(f"invalid resume next_epoch={start_epoch} for {num_epochs} epochs")
        print(f"resuming stage1 refit at epoch={start_epoch:03d}/{num_epochs - 1:03d}", flush=True)

    metrics_path = args.output_dir / "train_metrics.jsonl"
    if args.resume and metrics_path.is_file():
        metric_records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        retained_records = [
            record for record in metric_records if int(record["epoch"]) < start_epoch
        ]
        retained_epochs = [int(record["epoch"]) for record in retained_records]
        if retained_epochs != list(range(start_epoch)):
            raise ValueError(
                f"resume metric epochs {retained_epochs} do not match checkpoint next_epoch={start_epoch}"
            )
        metrics_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained_records),
            encoding="utf-8",
        )
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, labels, _, _ in tqdm(loader, desc=f"stage1 refit {epoch:03d}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model({"cam_front_img": images})["img_cls_output"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            sample_count += labels.size(0)
        record = {"epoch": epoch, "train_loss": running_loss / max(sample_count, 1)}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"stage1 refit epoch={epoch:03d}/{num_epochs - 1:03d} "
            f"train_loss={record['train_loss']:.6f}",
            flush=True,
        )
        atomic_torch_save(
            {
                "next_epoch": epoch + 1,
                "refit_epochs": num_epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loader_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            last_state_path,
        )

    atomic_torch_save(model.state_dict(), args.output_dir / "final.pth")
    print(f"stage1 refit complete checkpoint={args.output_dir / 'final.pth'}", flush=True)


if __name__ == "__main__":
    main()
