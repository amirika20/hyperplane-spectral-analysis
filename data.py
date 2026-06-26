"""
Dataset factory. Add new datasets by implementing a generator function with signature
    fn(cfg: Config, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]
and registering it in DATASET_REGISTRY.
"""
import math
from typing import Callable

import torch
from torch.utils.data import TensorDataset, DataLoader

from config import Config

# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------

def _gaussians(cfg: Config, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K Gaussian clusters with low-rank + isotropic covariance.
    Means are random unit vectors on S^{d-1}.
    Covariance: (1/rank) U U^T + noise_std^2 I, U ~ N(0,I).
    """
    X_list, y_list = [], []
    for k in range(cfg.K):
        mean = torch.randn(cfg.d, generator=rng)
        mean = mean / mean.norm()

        U = torch.randn(cfg.d, cfg.cov_rank, generator=rng) / (cfg.cov_rank ** 0.5)
        z = torch.randn(cfg.n_per_class, cfg.cov_rank, generator=rng)
        eps = torch.randn(cfg.n_per_class, cfg.d, generator=rng)
        X_k = mean.unsqueeze(0) + z @ U.T + cfg.noise_std * eps

        X_list.append(X_k)
        y_list.append(torch.full((cfg.n_per_class,), k, dtype=torch.long))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


def _cone(cfg: Config, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Double cone in R^d.

    A hidden unit axis a defines a double cone centered at the origin.
    The score of a point x is the absolute cosine of its angle to the axis:

        score(x) = |aᵀx| / ‖x‖  ∈ [0, 1]

    score ≈ 1  →  x nearly parallel to a  (near the axis / tip)
    score ≈ 0  →  x nearly orthogonal to a (near the equatorial belt)

    The K classes are K nested cone shells, boundaries set at empirical
    quantiles so every class has exactly n_per_class points.

    Decision boundary between adjacent classes k and k+1:
        |aᵀx| / ‖x‖ = c_k   (a cone surface, degree-2 algebraic variety)

    This is strictly non-linear: it cannot be written as a single hyperplane
    because the boundary depends on both the direction and scale of x.
    """
    a = torch.randn(cfg.d, generator=rng)
    a = a / a.norm()

    N = cfg.n_per_class * cfg.K
    X = torch.randn(N, cfg.d, generator=rng)

    norms = X.norm(dim=1).clamp(min=1e-8)
    score = (X @ a).abs() / norms          # |cos θ|, shape (N,)

    # K balanced classes via empirical quantiles
    cuts = torch.quantile(score, torch.linspace(0.0, 1.0, cfg.K + 1)[1:-1])
    y = torch.bucketize(score, cuts)        # 0 = near equator, K-1 = near axis

    return X, y


def _cylinder(cfg: Config, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2D cylinder embedded in R^d.

    Two orthonormal vectors u1, u2 span the signal subspace.
    Each point is sampled uniformly on the unit circle in that subspace, then
    corrupted with isotropic Gaussian noise in all d dimensions:
        x = cos(θ) u1 + sin(θ) u2 + noise_std * ε,  ε ~ N(0, I_d)

    The K classes correspond to K equal angular sectors of [0, 2π).
    The decision boundaries are K hyperplanes through the origin in R^d
    (intersections of the subspace with the sector boundaries), making
    the task non-trivially hard: the network must discover u1, u2.
    """
    # Two orthonormal signal directions (Gram-Schmidt)
    u1 = torch.randn(cfg.d, generator=rng)
    u1 = u1 / u1.norm()
    v = torch.randn(cfg.d, generator=rng)
    v = v - (v @ u1) * u1
    u2 = v / v.norm()

    N = cfg.n_per_class * cfg.K
    theta = torch.rand(N, generator=rng) * 2 * math.pi

    # Project onto signal subspace + noise
    eps = torch.randn(N, cfg.d, generator=rng)
    X = (theta.cos().unsqueeze(1) * u1 +
         theta.sin().unsqueeze(1) * u2 +
         cfg.noise_std * eps)

    # Label by angular sector
    sector = (2 * math.pi / cfg.K)
    y = (theta / sector).long().clamp(0, cfg.K - 1)

    return X, y


def _load_torchvision_flat(dataset_cls, cfg: Config, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared loader for MNIST-style torchvision datasets (28×28, 10 classes)."""
    import os
    os.makedirs(cfg.data_dir, exist_ok=True)

    train_ds = dataset_cls(cfg.data_dir, train=True,  download=True)
    test_ds  = dataset_cls(cfg.data_dir, train=False, download=True)

    all_images = torch.cat([train_ds.data, test_ds.data], dim=0).float() / 255.0
    all_labels = torch.cat([train_ds.targets, test_ds.targets], dim=0)

    X_list, y_list = [], []
    for new_label, cls in enumerate(cfg.mnist_classes):
        mask = all_labels == cls
        X_cls = all_images[mask]
        n = min(cfg.n_per_class, X_cls.shape[0])
        idx = torch.randperm(X_cls.shape[0], generator=rng)[:n]
        X_cls = X_cls[idx].view(n, -1)
        X_list.append(X_cls)
        y_list.append(torch.full((n,), new_label, dtype=torch.long))

    return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)


def _mnist(cfg: Config, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flattened MNIST subset.

    Downloads MNIST via torchvision (train + test splits combined).
    For each digit in cfg.mnist_classes, samples up to cfg.n_per_class examples.
    Images are flattened to 784-d vectors and normalized to [0, 1].

    cfg.K and cfg.d are ignored here — the caller (main.py) must update
    cfg.d = 784 and cfg.K = len(cfg.mnist_classes) after loading.
    """
    try:
        from torchvision.datasets import MNIST
    except ImportError:
        raise ImportError("torchvision is required for the MNIST dataset: pip install torchvision")
    return _load_torchvision_flat(MNIST, cfg, rng)


def _fashion_mnist(cfg: Config, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flattened Fashion-MNIST subset.

    Downloads Fashion-MNIST via torchvision (train + test splits combined).
    Classes 0-9 correspond to: T-shirt/top, Trouser, Pullover, Dress, Coat,
    Sandal, Shirt, Sneaker, Bag, Ankle boot.
    cfg.mnist_classes selects which classes to include (default: all 10).
    Images are flattened to 784-d vectors and normalized to [0, 1].
    """
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError:
        raise ImportError("torchvision is required for Fashion-MNIST: pip install torchvision")
    return _load_torchvision_flat(FashionMNIST, cfg, rng)


# ---------------------------------------------------------------------------
# Registry — add new datasets here
# ---------------------------------------------------------------------------

DatasetFn = Callable[[Config, torch.Generator], tuple[torch.Tensor, torch.Tensor]]

DATASET_REGISTRY: dict[str, DatasetFn] = {
    "gaussians":     _gaussians,
    "cylinder":      _cylinder,
    "cone":          _cone,
    "mnist":         _mnist,
    "fashion_mnist": _fashion_mnist,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {cfg.dataset!r}. "
            f"Available: {list(DATASET_REGISTRY)}"
        )
    rng = torch.Generator()
    rng.manual_seed(cfg.data_seed)

    X, y = DATASET_REGISTRY[cfg.dataset](cfg, rng)

    # Shuffle
    perm = torch.randperm(X.shape[0], generator=rng)
    return X[perm], y[perm]


def make_dataloaders(
    cfg: Config,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Returns (train_loader, val_loader, X_all, y_all).
    X_all / y_all are the full (shuffled) dataset — used by main.py to set cfg.d / cfg.K.
    val_frac fraction of data is held out for validation; never seen during training.
    """
    X, y = generate_dataset(cfg)
    N = X.shape[0]
    n_val = max(1, int(N * cfg.val_frac))
    n_train = N - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False
    )
    return train_loader, val_loader, X, y
