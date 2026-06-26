import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

from config import Config
from model import Net
from spectral import GradSnapshot, LayerGrad, SpectralSnapshot, compute_spectral_snapshot


def build_optimizer(model: Net, cfg: Config) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


@torch.no_grad()
def evaluate(model: Net, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Returns (avg_loss, accuracy) on the given loader."""
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    model.eval()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * X_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += X_batch.size(0)
    return total_loss / total, correct / total


def _capture_grad_snapshot(
    model: Net,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    cfg: Config,
    device: torch.device,
) -> GradSnapshot:
    """
    Compute the full-batch effective gradient for every hidden layer.

    For SGD: eff_grad = dL/dW (full-batch mean gradient).
    For Adam: eff_grad = P ⊙ dL/dW where P = 1/(sqrt(v_hat)+eps) is the
              bias-corrected coordinatewise Adam preconditioner.
    """
    model.eval()
    optimizer.zero_grad()

    N = len(train_loader.dataset)
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        # Scale by batch_size/N so accumulated gradient equals full-batch mean
        loss = criterion(logits, y_batch) * (X_batch.size(0) / N)
        loss.backward()

    hidden_layers = [m for m in model.hidden if isinstance(m, nn.Linear)]
    layer_grads: List[LayerGrad] = []

    for idx, layer in enumerate(hidden_layers):
        grad_W = layer.weight.grad.float().detach().clone()
        grad_b = layer.bias.grad.float().detach().clone()

        if (
            cfg.optimizer == "adam"
            and layer.weight in optimizer.state
            and "exp_avg_sq" in optimizer.state[layer.weight]
        ):
            state_W = optimizer.state[layer.weight]
            state_b = optimizer.state[layer.bias]

            step = state_W["step"]
            if isinstance(step, torch.Tensor):
                step = step.item()

            beta2      = optimizer.defaults["betas"][1]
            eps_adam   = optimizer.defaults["eps"]
            bias_corr2 = 1.0 - beta2 ** step

            v_hat_W = state_W["exp_avg_sq"] / bias_corr2
            v_hat_b = state_b["exp_avg_sq"] / bias_corr2

            P_W = 1.0 / (v_hat_W.float().sqrt() + eps_adam)   # (H, d)
            P_b = 1.0 / (v_hat_b.float().sqrt() + eps_adam)   # (H,)

            eff_grad_W = (P_W * grad_W).cpu()
            eff_grad_b = (P_b * grad_b).cpu()
        else:
            eff_grad_W = grad_W.cpu()
            eff_grad_b = grad_b.cpu()

        layer_grads.append(LayerGrad(
            layer_idx=idx,
            eff_grad_W=eff_grad_W,
            eff_grad_b=eff_grad_b,
        ))

    optimizer.zero_grad()  # don't leak into the next training step
    model.train()
    return GradSnapshot(epoch=epoch, layers=layer_grads)


def train(
    model: Net,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> tuple[List[SpectralSnapshot], List[GradSnapshot]]:
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    snapshots:      List[SpectralSnapshot] = []
    grad_snapshots: List[GradSnapshot]     = []

    # Snapshot at epoch 0 (before any training)
    model.eval()
    snapshots.append(compute_spectral_snapshot(model, epoch=0))
    grad_snapshots.append(
        _capture_grad_snapshot(model, train_loader, optimizer, criterion, 0, cfg, device)
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        if epoch % cfg.snapshot_every == 0 or epoch == cfg.epochs:
            train_acc = train_correct / train_total
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:>5}/{cfg.epochs}  "
                f"train_loss={train_loss / train_total:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )
            model.eval()
            snapshots.append(compute_spectral_snapshot(model, epoch=epoch))
            grad_snapshots.append(
                _capture_grad_snapshot(model, train_loader, optimizer, criterion, epoch, cfg, device)
            )

    return snapshots, grad_snapshots
