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
        Perceiver-style planner:
        - lane boundary points are encoded into a sequence of tokens;
        - a small set of learned waypoint queries attend to them via cross-attention;
        - optional self-attention layers refine the waypoint latents.

        Args:
            n_track (int): number of points per side of the track
            n_waypoints (int): number of waypoints to predict
            d_model (int): transformer embedding dimension
            n_heads (int): number of attention heads
            num_layers (int): total number of blocks
            dim_ff (int): hidden size of the feed-forward networks
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # normalize (x, y)
        self.input_norm = nn.LayerNorm(2)

        # project each (x, y) lane point into d_model
        self.track_proj = nn.Linear(2, d_model)

        # positional embeddings for lane points: indices 0..(2*n_track-1)
        self.lane_pos_embed = nn.Embedding(2 * n_track, d_model)
       
        # learned base queries for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)
       
        # deterministic sinusoidal positions for the waypoint queries
        # registered as a buffer so it moves with to(device) but has no gradients
        query_pos = self._build_sinusoidal_embeddings(n_waypoints, d_model)


        self.register_buffer("query_pos_embed", query_pos, persistent=False)

        # first block: cross-attention between queries and lane features
        # remaining blocks: self-attention among waypoint latents
        self.layers = nn.ModuleList(
            [CrossAttnBlock(d_model, n_heads, dim_ff)]
            + [
                TransformerBlock(d_model, n_heads, dim_ff)
                for _ in range(num_layers - 1)
            ]
        )

        # final head: map each latent to (x, y)
        self.output_head = nn.Linear(d_model, 2)

    @staticmethod
    def _build_sinusoidal_embeddings(n: int, d: int) -> torch.Tensor:
        """
        Standard transformer-style sinusoidal positional embeddings.
        Shape: (n, d)
        """
        pe = torch.zeros(n, d)
        position = torch.arange(0, n, dtype=torch.float32).unsqueeze(1)    # (n, 1)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) *
            (-torch.log(torch.tensor(10000.0)) / d)
        )  # (d/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        return pe  # (n, d)

    def forward(self, track_left, track_right, **kwargs):
        """
        Args:
            track_left:  (B, n_track, 2)
            track_right: (B, n_track, 2)

        Returns:
            waypoints:   (B, n_waypoints, 2)
        """
        B, n_track, _ = track_left.shape
        assert n_track == self.n_track, f"Expected n_track={self.n_track}, got {n_track}"

        # (1) build lane token sequence: concat left/right
        # lane: (B, 2*n_track, 2)
        lane = torch.cat([track_left, track_right], dim=1)
        lane = self.track_proj(lane)           # (B, 2*n_track, d_model)

        # (2) add lane positional embeddings
        pos_idx = torch.arange(2 * self.n_track, device=lane.device)  # (2*n_track,)
        lane_pos = self.lane_pos_embed(pos_idx)                       # (2*n_track, d_model)
        lane = lane + lane_pos.unsqueeze(0)                           # (B, 2*n_track, d_model)

        # (3) build query latents: learned base + fixed sinusoidal offsets
        # query_embed.weight: (n_waypoints, d_model)
        # query_pos_embed:    (n_waypoints, d_model)
        queries = self.query_embed.weight + self.query_pos_embed
        queries = queries.unsqueeze(0).expand(B, -1, -1)                    # (B, n_waypoints, d_model)            # (B, n_waypoints, d_model)

        # (4) first layer: cross-attention to pull info from lane tokens
        x = self.layers[0].cross(queries, lane)                       # (B, n_waypoints, d_model)

        # (5) subsequent layers: self-attention over waypoint latents
        for layer in self.layers[1:]:
            x = layer(x)                                              # (B, n_waypoints, d_model)

        # (6) predict (x, y) for each waypoint latent
        out = self.output_head(x)                                     # (B, n_waypoints, 2)
        return out


class CrossAttnBlock(nn.Module):
    """
    One cross-attention + feed-forward block:

      queries ----> Q
      lane_feats -> K, V

    Output has the same shape as queries.
    """

    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def cross(self, query, lane):
        """
        Args:
            query: (B, n_waypoints, d_model)
            lane:  (B, 2*n_track, d_model)
        """
        attn_out, _ = self.cross_attn(query=query, key=lane, value=lane)
        x = self.norm1(query + attn_out)   # residual + norm

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)         # residual + norm
        return x


class TransformerBlock(nn.Module):
    """
    Standard transformer block with self-attention over the waypoint latents.
    """

    def __init__(self, d_model, n_heads, dim_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, n_waypoints, d_model)
        """
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
