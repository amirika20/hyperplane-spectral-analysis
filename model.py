import torch
import torch.nn as nn

from config import Config

ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "sigmoid":    nn.Sigmoid,
    "gelu":       nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "elu":        nn.ELU,
}


class Net(nn.Module):
    """
    Multi-layer network with `depth` hidden layers of width H.

    Architecture:  x → [Linear(d,H) → Activation] × depth → Linear(H,K)

    The first hidden layer defines H hyperplanes
        H_i = {x ∈ R^d : w_i^T x + b_i = 0}
    whose geometry is the subject of spectral analysis, regardless of the
    choice of activation function.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.activation not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unknown activation {cfg.activation!r}. "
                f"Available: {list(ACTIVATION_REGISTRY)}"
            )
        act_cls = ACTIVATION_REGISTRY[cfg.activation]
        layers: list[nn.Module] = []
        in_dim = cfg.d
        for _ in range(cfg.depth):
            layers.append(nn.Linear(in_dim, cfg.H))
            layers.append(act_cls())
            in_dim = cfg.H
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.H, cfg.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))

    def first_layer(self) -> nn.Linear:
        """Returns the first Linear layer (the one whose hyperplanes are analysed)."""
        return next(m for m in self.hidden if isinstance(m, nn.Linear))

    def hyperplane_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(W, b) of the first hidden layer: W[i]=w_i, b[i]=b_i."""
        layer = self.first_layer()
        return layer.weight.detach(), layer.bias.detach()

    def all_layer_params(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """[(W_l, b_l)] for every hidden Linear layer, in order."""
        return [
            (m.weight.detach(), m.bias.detach())
            for m in self.hidden
            if isinstance(m, nn.Linear)
        ]
