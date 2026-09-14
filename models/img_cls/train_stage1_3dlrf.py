#!/usr/bin/env python3
"""Train a 3D-LRF-compatible weather encoder on disjoint frame splits.

This trains a new frontend; it does not reconstruct the shared paper checkpoint.
Validation sequences may overlap training sequences in the supplied split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.img_cls.cls_model_3dlrf import ImageClsBackbone3DLRF as ImageClsBackbone
from utils.split_contract import format_split_summary, read_split, verify_split_contract


WEATHER_NAMES = ("normal", "overcast", "fog", "rain", "sleet", "lightsnow", "heavysnow")


def merge_new_config(config, new_config):
    for key, value in new_config.items():
        if not isinstance(value, dict):
            config[key] = value
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], value)
    return config


def cfg_from_yaml_file(cfg_file: Path):
    with cfg_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return merge_new_config(EasyDict(), raw)


def resolve_project_path(value: str) -> Path:
    path = Path(os.path.expanduser(str(value)))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class ImgDataset(Dataset):
    def __init__(self, cfg, split: str, transform=None):
        super().__init__()
        if split not in cfg.DATASET.PATH_SPLIT:
            raise KeyError(f"unknown split {split!r}; available={list(cfg.DATASET.PATH_SPLIT.keys())}")
        self.split = split
        self.transform = transform
        split_path = resolve_project_path(cfg.DATASET.PATH_SPLIT[split])
        rows = read_split(split_path)
        rows_by_sequence = {}
        for row in rows:
            sequence, label_file = row.split(",", 1)
            rows_by_sequence.setdefault(sequence, set()).add(label_file[:-4])

        samples = []
        missing_images = []
        for dataset_root_raw in cfg.DATASET.DIR.LIST_DIR:
            dataset_root = Path(dataset_root_raw).resolve()
            for sequence in sorted(rows_by_sequence, key=int):
                sequence_root = dataset_root / sequence
                if not sequence_root.is_dir():
                    continue
                desc_fields = (sequence_root / "description.txt").read_text(
                    encoding="utf-8"
                ).splitlines()[0].strip().split(",")
                weather = desc_fields[-1]
                if weather not in WEATHER_NAMES:
                    raise ValueError(f"unknown weather {weather!r} for sequence {sequence}")
                weather_label = WEATHER_NAMES.index(weather)
                for label_path in sorted((sequence_root / "info_label").glob("*.txt")):
                    if label_path.stem not in rows_by_sequence[sequence]:
                        continue
                    header = label_path.read_text(encoding="utf-8").splitlines()[0]
                    indices = header.split(",", 1)[0].split("=", 1)[1].split("_")
                    camera_index = indices[2]
                    image_path = sequence_root / "cam-front" / f"cam-front_{camera_index}.png"
                    if not image_path.is_file():
                        missing_images.append(str(image_path))
                        continue
                    samples.append((image_path, weather_label, sequence))

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} camera images referenced by {split} are missing; "
                f"first={missing_images[0]}"
            )
        if len(samples) != len(rows):
            raise RuntimeError(
                f"stage-1 {split} resolved {len(samples)} samples from {len(rows)} split rows"
            )
        self.samples = samples

    def __getitem__(self, index):
        image_path, label, sequence = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        image = image[:, :1280]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(image_path), sequence

    def __len__(self):
        return len(self.samples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=202206)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics_from_confusion(confusion: torch.Tensor) -> dict:
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = confusion.sum().item()
    result = {
        "accuracy": true_positive.sum().item() / max(total, 1.0),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "support": int(total),
        "confusion_matrix": confusion.to(torch.int64).tolist(),
        "per_class": {},
    }
    for index, name in enumerate(WEATHER_NAMES):
        result["per_class"][name] = {
            "precision": precision[index].item(),
            "recall": recall[index].item(),
            "f1": f1[index].item(),
            "support": int(support[index].item()),
        }
    return result


def evaluate(model, loader, device) -> dict:
    model.eval()
    confusion = torch.zeros((len(WEATHER_NAMES), len(WEATHER_NAMES)), dtype=torch.int64)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels, _, _ in tqdm(loader, desc="stage1 val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model({"cam_front_img": images})["img_cls_output"]
            total_loss += criterion(logits, labels).item()
            predictions = logits.argmax(dim=1)
            indices = labels.cpu() * len(WEATHER_NAMES) + predictions.cpu()
            confusion += torch.bincount(indices, minlength=len(WEATHER_NAMES) ** 2).reshape(
                len(WEATHER_NAMES), len(WEATHER_NAMES)
            )
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / max(metrics["support"], 1)
    return metrics


def main():
    args = parse_args()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    set_seed(args.seed)
    cfg = cfg_from_yaml_file(args.config)
    split_summary = verify_split_contract(cfg, PROJECT_ROOT)
    print(format_split_summary(split_summary), flush=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = ImgDataset(cfg, split=args.train_split, transform=transform)
    valset = ImgDataset(cfg, split=args.val_split, transform=transform)
    train_sequences = {sample[2] for sample in trainset.samples}
    val_sequences = {sample[2] for sample in valset.samples}
    train_images = {str(sample[0]) for sample in trainset.samples}
    val_images = {str(sample[0]) for sample in valset.samples}
    if train_images & val_images:
        raise RuntimeError("stage-1 train/val camera frame overlap")

    generator = torch.Generator().manual_seed(args.seed)
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    network = ImageClsBackbone().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum)

    shutil.copy2(args.config, args.output_dir / "config.yml")
    manifest_source = PROJECT_ROOT / "resources/split/split_manifest_3dlrf_icra.json"
    shutil.copy2(manifest_source, args.output_dir / manifest_source.name)
    run_manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_samples": len(trainset),
        "val_samples": len(valset),
        "train_sequences": sorted(train_sequences, key=int),
        "val_sequences": sorted(val_sequences, key=int),
        "train_class_counts": dict(Counter(WEATHER_NAMES[label] for _, label, _ in trainset.samples)),
        "val_class_counts": dict(Counter(WEATHER_NAMES[label] for _, label, _ in valset.samples)),
        "selection_metric": "macro_f1",
        "official_test_evaluations_during_selection": 0,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    best_macro_f1 = float("-inf")
    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(args.epochs):
        network.train()
        running_loss = 0.0
        sample_count = 0
        for images, labels, _, _ in tqdm(trainloader, desc=f"stage1 train {epoch:03d}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = network({"cam_front_img": images})["img_cls_output"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            sample_count += labels.size(0)

        val_metrics = evaluate(network, valloader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(sample_count, 1),
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"stage1 epoch={epoch:03d} train_loss={record['train_loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f}",
            flush=True,
        )

        torch.save(network.state_dict(), args.output_dir / "last.pth")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(network.state_dict(), args.output_dir / "best.pth")
            best_record = dict(record)
            best_record["selection_metric"] = "macro_f1"
            (args.output_dir / "best_metrics.json").write_text(
                json.dumps(best_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"stage1 new_best epoch={epoch:03d} macro_f1={best_macro_f1:.6f}", flush=True)

    import hashlib
    best_path = args.output_dir / "best.pth"
    provenance = {
        "checkpoint": str(best_path),
        "checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "architecture": "ImageClsBackbone3DLRF",
        "lineage_status": "recorded_local_training",
        "train_split": str(resolve_project_path(cfg.DATASET.PATH_SPLIT[args.train_split])),
        "selection_split": str(resolve_project_path(cfg.DATASET.PATH_SPLIT[args.val_split])),
        "limitation": "Frame-disjoint selection; sequence independence is not claimed. Reusing this validation split for detector selection is not an independent outer validation.",
    }
    provenance_path = args.output_dir / "stage1_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    stage2_cfg = yaml.safe_load(args.config.read_text())
    stage2_cfg["MODEL"]["IMG_CLS"].update(
        NAME="ImageClsBackbone3DLRF", MODEL_PATH=str(best_path),
        PROVENANCE_PATH=str(provenance_path), LINEAGE_POLICY="shared_legacy")
    stage2_cfg["VAL"]["RUN_CONDITIONAL_AT_END"] = False
    (args.output_dir / "stage2_config.yml").write_text(yaml.safe_dump(stage2_cfg, sort_keys=False))

    best_state = torch.load(args.output_dir / "best.pth", map_location=device)
    network.load_state_dict(best_state, strict=True)
    final_metrics = evaluate(network, valloader, device)
    (args.output_dir / "best_recheck_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"stage1 complete best_macro_f1={best_macro_f1:.6f} "
        f"recheck_macro_f1={final_metrics['macro_f1']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
