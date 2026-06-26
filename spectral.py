from dataclasses import dataclass
from typing import List

import torch

from model import Net


@dataclass
class LayerSpectral:
    """Spectral analysis of the activation hyperplanes in one hidden layer."""
    layer_idx: int
    # Raw parameters stored for inter-snapshot velocity computation
    W: torch.Tensor             # (H, d_l) weight matrix
    b: torch.Tensor             # (H,) bias vector
    # --- orientation / covariance analysis ---
    eigenvalues: torch.Tensor   # (d_l,) sorted descending, d_l = input dim of this layer
    r_eff: float
    trace: float
    anchors: torch.Tensor       # (H, d_l)  x_{*,i} = rho_i * n_i
    norms: torch.Tensor         # (H,)  ||w_i||  (weight vector norms)
    distances: torch.Tensor     # (H,)  rho_i = -b_i / ||w_i||  (signed distance to origin)
    # --- distance (radial) analysis ---
    # ||x_{*,i}|| = |rho_i|: how far each hyperplane sits from the origin.
    # High variance means hyperplanes are spread across many radial shells;
    # low variance means they cluster at a similar distance from the origin.
    anchor_norm_mean: float     # mean_i  ||x_{*,i}||
    anchor_norm_std:  float     # std_i   ||x_{*,i}||
    # --- weight-matrix effective rank ---
    # r_eff of the singular-value spectrum of W (H × d_l).
    # Measures how many directions in weight space are meaningfully used,
    # independent of where the hyperplanes sit (biases play no role here).
    weight_r_eff: float


@dataclass
class LayerGrad:
    """Effective gradient for one layer at a snapshot epoch.

    For GD: eff_grad = dL/dW (raw full-batch gradient).
    For Adam: eff_grad = P ⊙ dL/dW where P = 1 / (sqrt(v_hat) + eps) is the
    bias-corrected coordinatewise preconditioner.
    """
    layer_idx: int
    eff_grad_W: torch.Tensor  # (H, d)
    eff_grad_b: torch.Tensor  # (H,)


@dataclass
class GradSnapshot:
    """Effective gradients captured at a specific epoch for all hidden layers."""
    epoch: int
    layers: List[LayerGrad]


@dataclass
class LayerVelocity:
    """Velocity of hyperplane geometry between two consecutive spectral snapshots."""
    layer_idx: int
    epoch_from: int
    epoch_to: int
    # Per-neuron velocity scalars (H,)
    direction_velocity: torch.Tensor  # v_i^(u) = ||u_dot_i||  (angular speed)
    distance_velocity: torch.Tensor   # v_i^(r) = |r_dot_i|    (translation speed)
    anchor_velocity: torch.Tensor     # v_i^(a) = sqrt((v^r)^2 + r^2*(v^u)^2)
    # Direction velocity vectors used for population analysis (H, d_l)
    u_dot: torch.Tensor
    # Population-level direction velocity covariance C_u = (1/H) U_vel^T U_vel
    Cu_eigenvalues: torch.Tensor      # sorted descending, length = min(H, d_l)
    Cu_lambda_max: float              # largest eigenvalue of C_u
    Cu_r_eff: float                   # effective rank of C_u


@dataclass
class VelocitySnapshot:
    """Velocities computed between two consecutive SpectralSnapshots."""
    epoch_from: int
    epoch_to: int
    layers: List[LayerVelocity]


@dataclass
class SpectralSnapshot:
    epoch: int
    layers: List[LayerSpectral]   # one entry per hidden layer, in order


def _layer_spectral(W: torch.Tensor, b: torch.Tensor, layer_idx: int) -> LayerSpectral:
    """
    Compute hyperplane-anchor covariance analysis for one linear layer.

    W: (H, d_l)  — weight matrix of the layer
    b: (H,)      — bias vector

    The hyperplane for neuron i in this layer is
        H_i = { z ∈ R^{d_l} : w_i^T z + b_i = 0 }
    defined in the space of that layer's inputs (R^{d_l}).
    """
    norms    = W.norm(dim=1)              # (H,)
    norms_sq = norms ** 2

    distances = -b / norms                              # rho_i = -b_i / ||w_i||
    anchors   = -(b / norms_sq).unsqueeze(1) * W        # (H, d_l)

    H = W.shape[0]
    mean    = anchors.mean(dim=0, keepdim=True)
    X_tilde = anchors - mean                            # (H, d_l)
    Sigma   = (X_tilde.T @ X_tilde) / H                # (d_l, d_l)

    eigenvalues = torch.linalg.eigvalsh(Sigma).flip(0).clamp(min=0.0)

    trace     = eigenvalues.sum().item()
    lam_sq    = (eigenvalues ** 2).sum().item()
    r_eff     = (trace ** 2) / lam_sq if lam_sq > 0 else float("nan")

    # Radial analysis: anchor norm = |rho_i| = ||x_{*,i}||
    anchor_norms = anchors.norm(dim=1)   # |rho_i|, shape (H,)

    # Weight effective rank via singular values of W
    sv      = torch.linalg.svdvals(W)    # (min(H, d_l),), descending
    sv_sq   = sv ** 2
    sv_sq_s = sv_sq.sum().item()
    weight_r_eff = sv_sq_s ** 2 / (sv_sq ** 2).sum().item() if sv_sq_s > 0 else float("nan")

    return LayerSpectral(
        layer_idx=layer_idx,
        W=W.cpu().clone(),
        b=b.cpu().clone(),
        eigenvalues=eigenvalues.cpu(),
        r_eff=r_eff,
        trace=trace,
        anchors=anchors.cpu(),
        norms=norms.cpu(),
        distances=distances.cpu(),
        anchor_norm_mean=anchor_norms.mean().item(),
        anchor_norm_std=anchor_norms.std().item(),
        weight_r_eff=weight_r_eff,
    )


def compute_spectral_snapshot(model: Net, epoch: int) -> SpectralSnapshot:
    layers = [
        _layer_spectral(W, b, idx)
        for idx, (W, b) in enumerate(model.all_layer_params())
    ]
    return SpectralSnapshot(epoch=epoch, layers=layers)


def _compute_layer_velocity(
    snap: LayerSpectral,
    eff_grad_W: torch.Tensor,
    eff_grad_b: torch.Tensor,
    epoch_from: int,
    epoch_to: int,
) -> LayerVelocity:
    """
    Compute analytical hyperplane velocity from the effective gradient.

    For GD (gradient flow), the dynamics are:
      u_dot_i = -(1/rho_i)(I - u_i u_i^T) G_w_i
      r_dot_i =  (1/rho_i)[G_b_i + r_i * u_i^T G_w_i]

    where G_w = dL/dW is the full-batch gradient (for GD) or the
    preconditioned gradient P ⊙ dL/dW (for Adam).

    Geometric quantities are evaluated at the 'from' snapshot.
    """
    W  = snap.W.float()
    b  = snap.b.float()
    Gw = eff_grad_W.float()
    Gb = eff_grad_b.float()

    eps = 1e-8
    rho = W.norm(dim=1).clamp(min=eps)   # (H,)
    u   = W / rho.unsqueeze(1)            # (H, d) unit normals
    r   = -b / rho                        # (H,) signed distances

    # Direction velocity: perpendicular component of effective gradient
    uT_Gw  = (u * Gw).sum(dim=1)                                        # (H,)
    u_dot  = -(Gw - uT_Gw.unsqueeze(1) * u) / rho.unsqueeze(1)         # (H, d)
    dir_vel = u_dot.norm(dim=1)                                          # (H,)

    # Signed-distance velocity
    r_dot   = (Gb + r * uT_Gw) / rho                                    # (H,)
    dist_vel = r_dot.abs()                                               # (H,)

    # Anchor velocity: orthogonal decomposition => Pythagorean sum
    anch_vel = (dist_vel**2 + r**2 * dir_vel**2).sqrt()                 # (H,)

    # Population-level C_u = (1/H) U_vel^T U_vel
    # Eigenvalues of C_u equal sv(U_vel)^2 / H (using the smaller Gram matrix)
    H  = W.shape[0]
    sv = torch.linalg.svdvals(u_dot)           # min(H,d), descending
    Cu_eig = sv ** 2 / H                       # eigenvalues of C_u, descending

    ev_sum    = Cu_eig.sum().item()
    ev_sq_sum = (Cu_eig ** 2).sum().item()
    Cu_r_eff  = ev_sum ** 2 / ev_sq_sum if ev_sq_sum > 0 else float("nan")

    return LayerVelocity(
        layer_idx=snap.layer_idx,
        epoch_from=epoch_from,
        epoch_to=epoch_to,
        direction_velocity=dir_vel.cpu(),
        distance_velocity=dist_vel.cpu(),
        anchor_velocity=anch_vel.cpu(),
        u_dot=u_dot.cpu(),
        Cu_eigenvalues=Cu_eig.cpu(),
        Cu_lambda_max=Cu_eig[0].item() if len(Cu_eig) > 0 else 0.0,
        Cu_r_eff=Cu_r_eff,
    )


def compute_all_velocities(
    snapshots: List[SpectralSnapshot],
    grad_snapshots: List[GradSnapshot],
) -> List[VelocitySnapshot]:
    """
    Compute analytical inter-snapshot velocities for every consecutive pair.

    grad_snapshots[i] must align with snapshots[i] (same epoch).
    velocity[i] uses the geometry at snapshots[i] and the effective gradient
    at grad_snapshots[i], representing the instantaneous velocity at the start
    of the interval [epoch_i, epoch_{i+1}].
    """
    result: List[VelocitySnapshot] = []
    for i in range(len(snapshots) - 1):
        snap      = snapshots[i]
        grad_snap = grad_snapshots[i]
        layers = [
            _compute_layer_velocity(
                snap.layers[l],
                grad_snap.layers[l].eff_grad_W,
                grad_snap.layers[l].eff_grad_b,
                epoch_from=snapshots[i].epoch,
                epoch_to=snapshots[i + 1].epoch,
            )
            for l in range(len(snap.layers))
        ]
        result.append(VelocitySnapshot(
            epoch_from=snapshots[i].epoch,
            epoch_to=snapshots[i + 1].epoch,
            layers=layers,
        ))
    return result
