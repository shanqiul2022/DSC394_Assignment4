"""
Usage:
    python3 -u -m homework.train_planner \
        --model mlp_planner \
        --epochs 20 --batch_size 64 --lr 1e-3 --data_root ./drive_data
"""

import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from homework.models import MODEL_FACTORY, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import RoadDataset


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_l1_loss(preds, targets, mask):
    """
    Computes the masked L1 loss.
    Args:
        preds:   (B, n_waypoints, 2)
        targets: (B, n_waypoints, 2)
        mask:    (B, n_waypoints)  (bool or 0/1)
    """
    l1 = torch.abs(preds - targets)        # (B, n_waypoints, 2)
    mask = mask.unsqueeze(-1).float()      # (B, n_waypoints, 1)
    l1 = l1 * mask                         # zero-out invalid waypoints
    valid = mask.sum()
    loss = l1.sum() / (valid + 1e-8)
    return loss


def evaluate(model, loader, device, model_name: str):
    """Evaluate model on the validation set using PlannerMetric."""
    model.eval()
    metric = PlannerMetric()

    with torch.no_grad():
        for batch in loader:
            waypoints = batch["waypoints"].to(device)        # (B, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)

            if model_name == "cnn_planner":
                # CNNPlanner uses only the image as input
                images = batch["image"].to(device)           # (B, 3, 96, 128)
                preds = model(image=images)
            else:
                # MLP and Transformer planners use lane boundaries
                track_left = batch["track_left"].to(device)   # (B, n_track, 2)
                track_right = batch["track_right"].to(device) # (B, n_track, 2)
                preds = model(track_left=track_left, track_right=track_right)

            metric.add(preds, waypoints, waypoints_mask)

    results = metric.compute()
    return results


def _episodes_under(split_dir: Path):
    """
    Return a list of episode directories under split_dir (those that contain info.npz).
    Also supports the case where split_dir itself is a single episode dir with info.npz.
    """
    if (split_dir / "info.npz").exists():
        return [split_dir]

    eps = sorted([p for p in split_dir.iterdir() if (p / "info.npz").exists()])
    return eps


def _load_split_as_dataset(split_dir: Path):
    """
    Build a dataset for a split by concatenating per-episode RoadDataset instances.
    If only a single episode exists, just return that dataset directly.
    """
    episodes = _episodes_under(split_dir)
    if len(episodes) == 0:
        raise FileNotFoundError(
            f"No episodes found in {split_dir}. "
            f"Expected subfolders like {split_dir}/000001/info.npz"
        )
    if len(episodes) == 1:
        return RoadDataset(str(episodes[0])), 1

    ds_list = [RoadDataset(str(ep)) for ep in episodes]
    return ConcatDataset(ds_list), len(ds_list)


def train(args):
    set_seed(1337)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Resolve project root ( .../DSC394_Assignment4 )
    ROOT = Path(__file__).resolve().parents[1]
    data_root = (ROOT / args.data_root).resolve()
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    # Build datasets
    train_ds, n_train_eps = _load_split_as_dataset(train_dir)
    val_ds, n_val_eps = _load_split_as_dataset(val_dir)
    print(f"Train episodes: {n_train_eps}, Val episodes: {n_val_eps}")
    print(f"Train samples:  {len(train_ds)} | Val samples: {len(val_ds)}")

    # DataLoader (num_workers=0 is safest in Colab for custom datasets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model + optimizer
    ModelCls = MODEL_FACTORY[args.model]

    # For all planners we keep n_track=10, n_waypoints=3 so grader can load defaults
    if args.model == "cnn_planner":
        model = ModelCls(n_waypoints=3).to(device)
    else:
        model = ModelCls(n_track=10, n_waypoints=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_l1 = float("inf")
    best_path = None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            waypoints = batch["waypoints"].to(device)          # (B, 3, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # (B, 3)

            if args.model == "cnn_planner":
                images = batch["image"].to(device)              # (B, 3, 96, 128)
                preds = model(image=images)
            else:
                track_left = batch["track_left"].to(device)     # (B, 10, 2)
                track_right = batch["track_right"].to(device)   # (B, 10, 2)
                preds = model(track_left=track_left, track_right=track_right)

            loss = masked_l1_loss(preds, waypoints, waypoints_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate for epoch average
            total_loss += loss.item() * waypoints_mask.sum().item()
            total_count += waypoints_mask.sum().item()

            pbar.set_postfix({"train_L1": loss.item()})

        train_loss = total_loss / max(total_count, 1)
        print(f"Epoch {epoch}: avg train L1 = {train_loss:.4f}")

        # Validation
        val_metrics = evaluate(model, val_loader, device, args.model)
        val_l1 = val_metrics["l1_error"]
        print(
            f"  [VAL] L1={val_l1:.4f}, "
            f"Longitudinal={val_metrics['longitudinal_error']:.4f}, "
            f"Lateral={val_metrics['lateral_error']:.4f}"
        )

        # Save best model so grader can load it later
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            best_path = save_model(model)
            print(f"  Model saved to: {best_path}")

    print(f"Best val L1={best_val_l1:.4f}, model at: {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Planner Models")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--data_root", type=str, default="drive_data",
        help="relative path (under project root) that contains train/ and val/"
    )
    parser.add_argument(
        "--model", type=str,
        default="mlp_planner",
        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
        help="which planner model to train"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()


