"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from homework.models import MLPPlanner, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import RoadDataset  # adjust import to match repo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mlp_planner(
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    # 1. Dataset & DataLoader (adjust split/path arguments to match your code)
    train_ds = RoadDataset(split="train")
    val_ds = RoadDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model
    model = MLPPlanner(n_track=10, n_waypoints=3).to(device)

    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Loss: masked L1 loss
    l1 = nn.L1Loss(reduction="none")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_train = 0

        for batch in train_loader:
            track_left = batch["track_left"].to(device)      # (B, 10, 2)
            track_right = batch["track_right"].to(device)    # (B, 10, 2)
            waypoints = batch["waypoints"].to(device)        # (B, 3, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # (B, 3) bool

            preds = model(track_left=track_left, track_right=track_right)

            # L1 per element: (B, 3, 2)
            loss_raw = l1(preds, waypoints)

            # Apply mask: expand to (B, 3, 2)
            mask = waypoints_mask[..., None]  # (B, 3, 1)
            loss_masked = loss_raw * mask

            # Average only over valid entries
            valid_count = mask.sum()
            loss = loss_masked.sum() / (valid_count + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * valid_count.item()
            num_train += valid_count.item()

        train_loss /= max(num_train, 1)
        print(f"Epoch {epoch+1}: train L1 loss (masked) = {train_loss:.4f}")

        # Optional: evaluate using PlannerMetric
        model.eval()
        metric = PlannerMetric()
        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                preds = model(track_left=track_left, track_right=track_right)
                metric.add(preds, waypoints, waypoints_mask)

        results = metric.compute()
        print(
            f"  Val L1={results['l1_error']:.4f}, "
            f"Longitudinal={results['longitudinal_error']:.4f}, "
            f"Lateral={results['lateral_error']:.4f}"
        )

    # Save model weights so the grader can load them
    path = save_model(model)
    print("Saved model to:", path)


