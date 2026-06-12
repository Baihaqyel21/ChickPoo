import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chickpoo.config import CLASS_NAMES, PROJECT_ROOT


class ChickPooDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None, max_per_class: int | None = None, path_column: str = "path"):
        df = pd.read_csv(csv_path)
        if max_per_class is not None:
            df = (
                df.groupby("label_id", group_keys=False)
                .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
                .reset_index(drop=True)
            )
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_column = path_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row[self.path_column]) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label_id"])
        return image, label


def parse_args():
    parser = argparse.ArgumentParser(description="Train ChickPoo MobileNetV2 baseline and fine-tuned model.")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "mobilenetv2_baseline")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional quick debug limit per class.")
    parser.add_argument("--path-column", type=str, default="path", help="CSV column containing the image path.")
    parser.add_argument("--csv-suffix", type=str, default="", help="Suffix before .csv, e.g. _cached_128.")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def make_loaders(args):
    train_tf, eval_tf = make_transforms(args.image_size)
    train_ds = ChickPooDataset(
        args.splits_dir / f"train{args.csv_suffix}.csv",
        transform=train_tf,
        max_per_class=args.max_per_class,
        path_column=args.path_column,
    )
    val_ds = ChickPooDataset(
        args.splits_dir / f"val{args.csv_suffix}.csv",
        transform=eval_tf,
        max_per_class=args.max_per_class,
        path_column=args.path_column,
    )
    test_ds = ChickPooDataset(
        args.splits_dir / f"test{args.csv_suffix}.csv",
        transform=eval_tf,
        max_per_class=args.max_per_class,
        path_column=args.path_column,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(num_classes: int):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(dataset: ChickPooDataset, device):
    counts = dataset.df["label_id"].value_counts().sort_index()
    weights = len(dataset.df) / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights.values, dtype=torch.float32, device=device)


def set_trainable_for_baseline(model: nn.Module):
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def set_trainable_for_finetune(model: nn.Module, trainable_blocks: int = 4):
    for param in model.features.parameters():
        param.requires_grad = False
    for block in model.features[-trainable_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds = []
    all_labels = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        progress = tqdm(loader, leave=False, desc="train" if train else "eval")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / len(loader.dataset)
    metrics = summarize_predictions(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


def summarize_predictions(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def predict(model, loader, device):
    model.eval()
    labels = []
    preds = []
    probs = []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, leave=False, desc="predict"):
            logits = model(images.to(device))
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels.extend(batch_labels.numpy().tolist())
            preds.extend(batch_probs.argmax(axis=1).tolist())
            probs.extend(batch_probs.tolist())
    return np.array(labels), np.array(preds), np.array(probs)


def save_checkpoint(path: Path, model, args, stage: str, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": args.image_size,
            "stage": stage,
            "metrics": metrics,
        },
        path,
    )


def save_metrics(output_dir: Path, name: str, labels, preds, probs):
    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
        output_dict=True,
    )
    summary = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }
    with open(output_dir / f"{name}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    np.save(output_dir / f"{name}_probabilities.npy", probs)
    plot_confusion(summary["confusion_matrix"], output_dir / f"{name}_confusion_matrix.png", title=name)
    return summary


def plot_confusion(matrix, output_path: Path, title: str):
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_stage(model, train_loader, val_loader, criterion, optimizer, device, args, stage: str, epochs: int, start_epoch: int):
    best_macro_f1 = -1.0
    best_path = args.output_dir / f"best_{stage}.pt"
    history = []
    for epoch in range(1, epochs + 1):
        epoch_index = start_epoch + epoch
        started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, train=False)
        row = {
            "stage": stage,
            "epoch": epoch_index,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "seconds": round(time.time() - started, 2),
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(best_path, model, args, stage=stage, metrics=val_metrics)
    return history, best_path


def load_checkpoint_into_model(model, path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(args)
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    class_weights = compute_class_weights(train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    config = vars(args).copy()
    config["device"] = str(device)
    config["class_names"] = CLASS_NAMES
    config["train_count"] = len(train_ds)
    config["val_count"] = len(val_ds)
    config["test_count"] = len(test_ds)
    config["class_weights"] = class_weights.detach().cpu().numpy().tolist()
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, default=str)

    set_trainable_for_baseline(model)
    baseline_optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    baseline_history, baseline_path = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        baseline_optimizer,
        device,
        args,
        stage="baseline",
        epochs=args.epochs,
        start_epoch=0,
    )

    load_checkpoint_into_model(model, baseline_path, device)
    baseline_test_labels, baseline_test_preds, baseline_test_probs = predict(model, test_loader, device)
    baseline_test_summary = save_metrics(args.output_dir, "baseline_test", baseline_test_labels, baseline_test_preds, baseline_test_probs)

    set_trainable_for_finetune(model, trainable_blocks=4)
    fine_tune_optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.fine_tune_lr,
        weight_decay=args.weight_decay,
    )
    fine_tune_history, fine_tune_path = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        fine_tune_optimizer,
        device,
        args,
        stage="fine_tune",
        epochs=args.fine_tune_epochs,
        start_epoch=args.epochs,
    )

    load_checkpoint_into_model(model, fine_tune_path, device)
    tuned_test_labels, tuned_test_preds, tuned_test_probs = predict(model, test_loader, device)
    tuned_test_summary = save_metrics(args.output_dir, "fine_tuned_test", tuned_test_labels, tuned_test_preds, tuned_test_probs)

    history = pd.DataFrame(baseline_history + fine_tune_history)
    history.to_csv(args.output_dir / "history.csv", index=False)

    comparison = {
        "baseline_test": {
            "accuracy": baseline_test_summary["accuracy"],
            "macro_f1": baseline_test_summary["macro_f1"],
        },
        "fine_tuned_test": {
            "accuracy": tuned_test_summary["accuracy"],
            "macro_f1": tuned_test_summary["macro_f1"],
        },
    }
    with open(args.output_dir / "comparison.json", "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
