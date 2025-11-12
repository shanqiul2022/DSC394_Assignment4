"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from homework.models import MLPPlanner, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import RoadDataset  

def masked_l1_loss(preds, targets, mask):
    """
    Computes the masked L1 loss.
    Args:
        preds: (B, n_waypoints, 2)
        targets: (B, n_waypoints, 2)
        mask: (B, n_waypoints)
    """
    l1 = torch.abs(preds - targets)  # (B, n_waypoints, 2)
    mask = mask.unsqueeze(-1)        # (B, n_waypoints, 1)
    l1 = l1 * mask                   # zero-out invalid waypoints
    valid = mask.sum()
    loss = l1.sum() / (valid + 1e-8)
    return loss


def evaluate(model, loader, device):
    """Evaluate model on the validation set using PlannerMetric."""
    model.eval()
    metric = PlannerMetric()

    with torch.no_grad():
        for batch in loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            preds = model(track_left=track_left, track_right=track_right)
            metric.add(preds, waypoints, waypoints_mask)

    results = metric.compute()
    return results


def train(args):
    """Train the MLPPlanner model."""
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    train_ds = RoadDataset(split="train", transform="default")
    val_ds = RoadDataset(split="val", transform="default")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model + optimizer
    model = MLPPlanner(n_track=10, n_waypoints=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            preds = model(track_left=track_left, track_right=track_right)
            loss = masked_l1_loss(preds, waypoints, waypoints_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * waypoints_mask.sum().item()
            total_count += waypoints_mask.sum().item()

            pbar.set_postfix({"train_L1": loss.item()})

        train_loss = total_loss / max(total_count, 1)
        print(f"Epoch {epoch}: avg train L1 = {train_loss:.4f}")

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"  [VAL] L1={val_metrics['l1_error']:.4f}, "
            f"Longitudinal={val_metrics['longitudinal_error']:.4f}, "
            f"Lateral={val_metrics['lateral_error']:.4f}"
        )

    # Save model weights
    save_path = save_model(model)
    print(f"✅ Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MLP Planner (Assignment 4 Part 1a)")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use (cuda or cpu)"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
