import argparse
import torch

from config import Config
from data import make_dataloaders, DATASET_REGISTRY
from model import Net, ACTIVATION_REGISTRY
from train import train
from spectral import compute_all_velocities
from plotting import save_all_plots


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Spectral analysis of hyperplanes in a shallow ReLU network"
    )
    # Data
    p.add_argument("--dataset", choices=list(DATASET_REGISTRY), default=cfg.dataset)
    p.add_argument("--d", type=int, default=cfg.d, help="Input dimension")
    p.add_argument("--K", type=int, default=cfg.K, help="Number of classes")
    p.add_argument("--n_per_class", type=int, default=cfg.n_per_class)
    p.add_argument("--val_frac", type=float, default=cfg.val_frac,
                   help="Fraction of data held out for validation (default 0.2)")
    p.add_argument("--cov_rank", type=int, default=cfg.cov_rank,
                   help="Rank of low-rank covariance component")
    p.add_argument("--noise_std", type=float, default=cfg.noise_std,
                   help="Isotropic noise std in covariance")
    p.add_argument("--data_seed", type=int, default=cfg.data_seed,
                   help="Seed for dataset generation and DataLoader shuffle (default 42)")
    p.add_argument("--model_seed", type=int, default=cfg.model_seed,
                   help="Seed for model weight initialisation (default 0)")
    # MNIST-specific
    p.add_argument("--mnist_classes", nargs="+", type=int, default=None,
                   help="Digits to include, e.g. --mnist_classes 0 1 (default: all 10)")
    p.add_argument("--data_dir", type=str, default=cfg.data_dir)
    # Model
    p.add_argument("--H", type=int, default=cfg.H, help="Width of each hidden layer")
    p.add_argument("--depth", type=int, default=cfg.depth, help="Number of hidden layers")
    p.add_argument("--activation", choices=list(ACTIVATION_REGISTRY), default=cfg.activation)
    # Training
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default=cfg.optimizer)
    p.add_argument("--momentum", type=float, default=cfg.momentum)
    p.add_argument("--snapshot_every", type=int, default=cfg.snapshot_every,
                   help="Take spectral snapshot every N epochs")
    # Output
    p.add_argument("--output_dir", type=str, default=cfg.output_dir)

    args = p.parse_args()
    if args.mnist_classes is None:
        args.mnist_classes = list(range(10))
    return Config(**vars(args))


def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: dataset={cfg.dataset}, d={cfg.d}, H={cfg.H}, K={cfg.K}, "
          f"n_per_class={cfg.n_per_class}, noise_std={cfg.noise_std}, "
          f"epochs={cfg.epochs}, lr={cfg.lr}, optimizer={cfg.optimizer}, "
          f"activation={cfg.activation}, "
          f"data_seed={cfg.data_seed}, model_seed={cfg.model_seed}, "
          f"snapshot_every={cfg.snapshot_every}")

    torch.manual_seed(cfg.data_seed)   # seeds DataLoader shuffle
    train_loader, val_loader, X, y = make_dataloaders(cfg)
    # MNIST overrides d and K from actual data; safe to do for all datasets
    cfg.d = X.shape[1]
    cfg.K = int(y.max().item()) + 1
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"Dataset: {X.shape[0]} samples ({n_train} train / {n_val} val), "
          f"{cfg.K} classes, d={cfg.d}")

    X_train = X[:n_train]
    X_c = X_train - X_train.mean(dim=0)
    Sigma_data = (X_c.T @ X_c) / n_train
    data_eigenvalues = torch.linalg.eigvalsh(Sigma_data).flip(0).clamp(min=0.0)
    _tr = data_eigenvalues.sum().item()
    _sq = (data_eigenvalues ** 2).sum().item()
    data_r_eff = _tr ** 2 / _sq if _sq > 0 else float("nan")

    torch.manual_seed(cfg.model_seed)  # seeds weight initialisation
    model = Net(cfg)
    print(f"Model: depth={cfg.depth}, {sum(p.numel() for p in model.parameters())} parameters")

    snapshots, grad_snapshots = train(model, train_loader, val_loader, cfg, device)

    vel_snapshots = compute_all_velocities(snapshots, grad_snapshots)
    save_all_plots(snapshots, vel_snapshots, cfg.output_dir,
                   n_classes=cfg.K,
                   data_eigenvalues=data_eigenvalues,
                   data_r_eff=data_r_eff)
    print("Done.")


if __name__ == "__main__":
    main()
