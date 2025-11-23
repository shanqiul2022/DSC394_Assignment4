from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden: int = 128,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden (int): hidden width of the MLP
        """
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        in_dim = 4 * n_track          # left (n_track,2) + right (n_track,2) => 4*n_track scalars
        out_dim = 2 * n_waypoints     # (x,y) for each waypoint

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0]
        x_left = track_left.reshape(B, -1)    # (B, 2*n_track)
        x_right = track_right.reshape(B, -1)  # (B, 2*n_track)
        x = torch.cat([x_left, x_right], dim=-1)  # (B, 4*n_track)

        out = self.net(x)                          # (B, 2*n_waypoints)
        out = out.view(B, self.n_waypoints, 2)     # (B, n_waypoints, 2)
        return out


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
    ):
        """
        A Perceiver-style transformer that uses learned waypoint queries to attend
        over the lane boundary points.

        Args:
            n_track (int): number of points per side of the track
            n_waypoints (int): number of waypoints to predict
            d_model (int): transformer embedding dimension
            n_heads (int): number of attention heads
            num_layers (int): number of cross-attention blocks
            dim_ff (int): hidden size of the feed-forward networks
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # normalize coordinates slightly
        self.input_norm = nn.LayerNorm(2)
        self.norm_in = self.input_norm

        # Embed each (x, y) lane point into d_model
        self.track_proj = nn.Linear(2, d_model)
        self.lane_proj = self.track_proj
        # Positional embeddings for lane points (2 * n_track positions)
        self.lane_pos_embed = nn.Embedding(2 * n_track, d_model)
        # Learned query embeddings for each waypoint (the "latent array")
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, dim_ff),
            *[
                TransformerBlock(d_model, n_heads, dim_ff)
                for _ in range(num_layers - 1)
            ]
        ])

        # Final head: project each query embedding to (x, y) waypoint
        self.output_head = nn.Linear(d_model, 2)

    def forward(self, track_left, track_right, **kwargs):
        B, n_track, _ = track_left.shape
        assert n_track == self.n_track, f"Expected n_track={self.n_track}, got {n_track}"

        # (1) Combine lane boundaries
        lane = torch.cat([track_left, track_right], dim=1)
        lane = self.track_proj(lane)

        # add positional encodings
        pos_idx = torch.arange(lane.size(1), device=lane.device)
        lane = lane + self.lane_pos_embed(pos_idx)[None, :, :]

        # (2) Build queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # (4) First layer: cross-attention between queries and lane features
        x = self.layers[0].cross(queries, lane)

        # (5) Remaining layers: self-attention over the waypoint latents
        for layer in self.layers[1:]:
            x = layer(x)

        # (6) Project to (x, y) waypoints
        out = self.output_head(x)  # (B, n_waypoints, 2)
        return out


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def cross(self, query, lane):
        attn_out, _ = self.cross_attn(query=query, key=lane, value=lane)
        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerBlock(nn.Module):
    """Self-attention block used after the cross-attention layer."""
    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
