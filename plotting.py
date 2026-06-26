import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde

from spectral import SpectralSnapshot, VelocitySnapshot


def _cmap_colors(n: int):
    return cm.viridis(np.linspace(0.1, 0.9, n))


def plot_spectrum_and_rank(snapshots: List[SpectralSnapshot], output_dir: str,
                           n_classes: int = None,
                           data_eigenvalues=None, data_r_eff: float = None):
    """
    One row per hidden layer, two columns:
      Left:  eigenvalue spectrum λ_k vs rank k (one curve per snapshot epoch).
      Right: effective rank r_eff over training epochs.
    """
    depth   = len(snapshots[0].layers)
    trained = [s for s in snapshots if s.epoch > 0]
    colors  = _cmap_colors(len(trained))
    epochs  = [s.epoch for s in snapshots]

    def _nonzero(lam: np.ndarray) -> np.ndarray:
        return lam[lam > 1e-10]

    fig, axes = plt.subplots(depth, 3, figsize=(18, 5 * depth), squeeze=False)

    for l in range(depth):
        ax_spec = axes[l, 0]
        ax_rank = axes[l, 1]
        ax_dist = axes[l, 2]
        label   = f"Layer {l + 1}"

        # --- Eigenvalue spectrum ---------------------------------------------
        for snap, color in zip(trained, colors):
            lam = _nonzero(snap.layers[l].eigenvalues.numpy())
            ax_spec.plot(np.arange(1, len(lam) + 1), lam, color=color,
                         label=f"epoch {snap.epoch}", alpha=0.85, linewidth=1.2)

        init = next((s for s in snapshots if s.epoch == 0), None)
        if init is not None:
            lam = _nonzero(init.layers[l].eigenvalues.numpy())
            ax_spec.plot(np.arange(1, len(lam) + 1), lam,
                         color="crimson", linewidth=2.0, linestyle="--",
                         label="epoch 0 (init)", zorder=5)

        if data_eigenvalues is not None:
            data_lam = _nonzero(data_eigenvalues.numpy())
            ax_spec.plot(np.arange(1, len(data_lam) + 1), data_lam,
                         color="black", linewidth=2.0, linestyle=":",
                         label="data $\\Sigma_X$", zorder=6)

        ax_spec.set_xlabel("Rank k")
        ax_spec.set_ylabel(r"$\lambda_k$")
        ax_spec.set_title(f"{label} — eigenvalue spectrum")
        ax_spec.legend(fontsize=7, ncol=2)
        ax_spec.set_yscale("log")

        # --- Effective rank: anchors vs weights ------------------------------
        r_effs_anchor = [s.layers[l].r_eff        for s in snapshots]
        r_effs_weight = [s.layers[l].weight_r_eff for s in snapshots]
        ax_rank.plot(epochs, r_effs_anchor, marker="o", markersize=4,
                     linewidth=1.5, label="anchors $x_{*,i}$")
        ax_rank.plot(epochs, r_effs_weight, marker="s", markersize=4,
                     linewidth=1.5, linestyle="--", label="weights $W$")
        if n_classes is not None:
            ax_rank.axhline(n_classes, color="red", linestyle="--", linewidth=1.2,
                            label=f"# classes ({n_classes})")
        if data_r_eff is not None:
            ax_rank.axhline(data_r_eff, color="black", linestyle=":", linewidth=1.2,
                            label=f"data $r_{{\\mathrm{{eff}}}}$ ({data_r_eff:.1f})")
        ax_rank.set_xlabel("Epoch")
        ax_rank.set_ylabel(r"$r_{\mathrm{eff}}$")
        ax_rank.set_title(f"{label} — effective rank over training")
        ax_rank.legend(fontsize=8)

        # --- Anchor-norm mean ± std ------------------------------------------
        stds = np.array([s.layers[l].anchor_norm_std for s in snapshots])
        ax_dist.plot(epochs, stds, marker="o", markersize=4, linewidth=1.5)
        ax_dist.set_xlabel("Epoch")
        ax_dist.set_ylabel(r"std $\|\mathbf{x}_{*,i}\|$")
        ax_dist.set_title(f"{label} — anchor distance std")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrum_and_rank.png"), dpi=150)
    plt.close(fig)



def _plot_kde_one_layer(
    snapshots: List[SpectralSnapshot],
    layer_idx: int,
    output_dir: str,
):
    """KDE figure for a single hidden layer across all snapshots."""
    n     = len(snapshots)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig      = plt.figure(figsize=(5.5 * ncols, 3.5 * nrows))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.6, wspace=0.45)

    def _log_kde(vals, ax, color):
        if len(vals) < 2:
            return 0.0
        log_v  = np.log10(vals[vals > 0])
        kde    = gaussian_kde(log_v, bw_method="scott")
        log_xs = np.linspace(log_v.min(), log_v.max(), 400)
        xs     = 10.0 ** log_xs
        pdf    = kde(log_xs) / (xs * np.log(10.0))
        ax.plot(xs, pdf, linewidth=1.5, color=color)
        ax.fill_between(xs, pdf, alpha=0.25, color=color)
        ax.set_xscale("log")
        return float(pdf.max())

    for idx, snap in enumerate(snapshots):
        row, col = divmod(idx, ncols)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[row, col],
            width_ratios=[4, 1], wspace=0.06,
        )
        ax_b = fig.add_subplot(inner_gs[0])
        ax_s = fig.add_subplot(inner_gs[1])

        lam     = snap.layers[layer_idx].eigenvalues.numpy()
        lam_pos = lam[lam > 1e-10]

        if len(lam_pos) < 2:
            ax_b.set_visible(False)
            ax_s.set_visible(False)
            continue

        lam_norm = lam_pos / lam_pos.sum()
        cutoff   = np.percentile(lam_norm, 70)
        bulk     = lam_norm[lam_norm <= cutoff]
        top      = lam_norm[lam_norm  > cutoff]

        y_max_b = _log_kde(bulk, ax_b, "steelblue")
        ax_b.set_ylabel("Density", fontsize=8)
        ax_b.tick_params(axis="both", labelsize=7)
        ax_b.spines["right"].set_visible(False)

        y_max_s = _log_kde(top, ax_s, "crimson")
        ax_s.tick_params(axis="both", labelsize=7)
        ax_s.spines["left"].set_visible(False)
        ax_s.set_xlabel("top 30%", fontsize=7, color="crimson")

        ax_b.set_ylim(0, y_max_b * 1.2 if y_max_b > 0 else 1.0)
        ax_s.set_ylim(0, y_max_s * 1.2 if y_max_s > 0 else 1.0)
        ax_s.yaxis.set_visible(True)
        ax_s.yaxis.tick_right()
        ax_s.tick_params(axis="y", labelsize=6)

        d  = 0.018
        kw = dict(color="k", clip_on=False, linewidth=0.9)
        for ax, xs_pos in [(ax_b, [1 - d, 1 + d]), (ax_s, [-d, +d])]:
            t = ax.transAxes
            ax.plot(xs_pos, [-d, +d], transform=t, **kw)
            ax.plot(xs_pos, [1 - d, 1 + d], transform=t, **kw)

        r_eff = snap.layers[layer_idx].r_eff
        ax_b.set_title(
            f"epoch {snap.epoch}  $r_{{\\mathrm{{eff}}}}={r_eff:.1f}$", fontsize=9
        )
        ax_b.set_xlabel(r"$\lambda_k\,/\,\mathrm{Tr}(\Sigma_*)$", fontsize=8)

    fig.suptitle(
        f"Layer {layer_idx + 1} — eigenvalue distribution at training snapshots",
        fontsize=11, y=1.01,
    )
    fname = f"eigenvalue_kde_layer{layer_idx + 1}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eigenvalue_kde(snapshots: List[SpectralSnapshot], output_dir: str):
    """One KDE figure per hidden layer."""
    depth = len(snapshots[0].layers)
    for l in range(depth):
        _plot_kde_one_layer(snapshots, l, output_dir)


def plot_velocity_overview(vel_snapshots: List[VelocitySnapshot], output_dir: str):
    """
    One row per hidden layer, three columns:
      Col 0: mean ± std of v^(u), v^(r), v^(a) over epochs.
      Col 1: lambda_max of C_u (dominant direction in velocity space) over epochs.
      Col 2: effective rank of C_u (dimensionality of velocity space) over epochs.
    """
    if not vel_snapshots:
        return

    depth  = len(vel_snapshots[0].layers)
    epochs = [v.epoch_to for v in vel_snapshots]

    fig, axes = plt.subplots(depth, 3, figsize=(18, 5 * depth), squeeze=False)

    for l in range(depth):
        ax_vel  = axes[l, 0]
        ax_lmax = axes[l, 1]
        ax_reff = axes[l, 2]
        label   = f"Layer {l + 1}"

        # ---- Mean ± std of per-neuron velocities ----------------------------
        for key, color, ls, marker, tex in [
            ("direction_velocity", "steelblue",  "-",  "o", r"$v^{(u)}$  (angular)"),
            ("distance_velocity",  "darkorange", "--", "s", r"$v^{(r)}$  (translation)"),
            ("anchor_velocity",    "seagreen",   ":",  "^", r"$v^{(a)}$  (anchor)"),
        ]:
            means = [getattr(v.layers[l], key).mean().item() for v in vel_snapshots]
            stds  = [getattr(v.layers[l], key).std().item()  for v in vel_snapshots]
            means = np.array(means)
            stds  = np.array(stds)
            ax_vel.plot(epochs, means, color=color, linestyle=ls, marker=marker,
                        markersize=4, linewidth=1.5, label=tex)
            ax_vel.fill_between(epochs, means - stds, means + stds,
                                color=color, alpha=0.15)

        ax_vel.set_xlabel("Epoch")
        ax_vel.set_ylabel("Velocity (mean ± std)")
        ax_vel.set_title(f"{label} — hyperplane velocities")
        ax_vel.legend(fontsize=8)

        # ---- lambda_max(C_u) -------------------------------------------------
        lmax_vals = [v.layers[l].Cu_lambda_max for v in vel_snapshots]
        ax_lmax.plot(epochs, lmax_vals, marker="o", markersize=4,
                     linewidth=1.5, color="mediumvioletred")
        ax_lmax.set_xlabel("Epoch")
        ax_lmax.set_ylabel(r"$\lambda_{\max}(C_u)$")
        ax_lmax.set_title(f"{label} — dominant direction velocity")

        # ---- r_eff(C_u) ------------------------------------------------------
        reff_vals = [v.layers[l].Cu_r_eff for v in vel_snapshots]
        ax_reff.plot(epochs, reff_vals, marker="s", markersize=4,
                     linewidth=1.5, color="saddlebrown")
        ax_reff.set_xlabel("Epoch")
        ax_reff.set_ylabel(r"$r_{\mathrm{eff}}(C_u)$")
        ax_reff.set_title(f"{label} — effective rank of direction velocity space")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "velocity_overview.png"), dpi=150)
    plt.close(fig)


def _plot_velocity_kde_one_layer(
    vel_snapshots: List[VelocitySnapshot],
    layer_idx: int,
    output_dir: str,
):
    """KDE of per-neuron anchor velocity v^(a) for one layer at each snapshot."""
    n     = len(vel_snapshots)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, vs in enumerate(vel_snapshots):
        ax   = axes_flat[idx]
        vals = vs.layers[layer_idx].anchor_velocity.numpy()
        vals = vals[vals > 0]

        if len(vals) >= 2:
            log_v  = np.log10(vals)
            kde    = gaussian_kde(log_v, bw_method="scott")
            log_xs = np.linspace(log_v.min(), log_v.max(), 400)
            xs     = 10.0 ** log_xs
            pdf    = kde(log_xs) / (xs * np.log(10.0))
            ax.plot(xs, pdf, linewidth=1.5, color="seagreen")
            ax.fill_between(xs, pdf, alpha=0.25, color="seagreen")
            ax.set_xscale("log")

        mean_v = vs.layers[layer_idx].anchor_velocity.mean().item()
        ax.set_title(
            f"ep {vs.epoch_from}→{vs.epoch_to}  "
            f"$\\bar{{v}}^{{(a)}}={mean_v:.3g}$",
            fontsize=9,
        )
        ax.set_xlabel(r"$v_i^{(a)}$", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Layer {layer_idx + 1} — anchor velocity distribution",
        fontsize=11, y=1.01,
    )
    fname = f"velocity_kde_layer{layer_idx + 1}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_kde(vel_snapshots: List[VelocitySnapshot], output_dir: str):
    """One KDE figure per hidden layer showing per-neuron anchor velocity distribution."""
    if not vel_snapshots:
        return
    depth = len(vel_snapshots[0].layers)
    for l in range(depth):
        _plot_velocity_kde_one_layer(vel_snapshots, l, output_dir)


def plot_Cu_spectrum(vel_snapshots: List[VelocitySnapshot], output_dir: str):
    """
    Eigenvalue spectrum of C_u (direction velocity covariance) at each snapshot.
    One figure per hidden layer.
    """
    if not vel_snapshots:
        return

    depth  = len(vel_snapshots[0].layers)
    colors = _cmap_colors(len(vel_snapshots))

    for l in range(depth):
        fig, ax = plt.subplots(figsize=(8, 4))
        for vs, color in zip(vel_snapshots, colors):
            eig = vs.layers[l].Cu_eigenvalues.numpy()
            eig = eig[eig > 1e-14]
            if len(eig) == 0:
                continue
            ax.plot(np.arange(1, len(eig) + 1), eig, color=color,
                    alpha=0.85, linewidth=1.2,
                    label=f"ep {vs.epoch_from}→{vs.epoch_to}")

        ax.set_xlabel("Rank k")
        ax.set_ylabel(r"$\lambda_k(C_u)$")
        ax.set_title(f"Layer {l + 1} — direction velocity covariance spectrum $C_u$")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"Cu_spectrum_layer{l + 1}.png"), dpi=150)
        plt.close(fig)


def save_all_plots(
    snapshots: List[SpectralSnapshot],
    vel_snapshots: List[VelocitySnapshot],
    output_dir: str,
    n_classes: int = None,
    data_eigenvalues=None,
    data_r_eff: float = None,
):
    os.makedirs(output_dir, exist_ok=True)
    plot_spectrum_and_rank(snapshots, output_dir, n_classes=n_classes,
                           data_eigenvalues=data_eigenvalues, data_r_eff=data_r_eff)
    plot_eigenvalue_kde(snapshots, output_dir)
    plot_velocity_overview(vel_snapshots, output_dir)
    plot_velocity_kde(vel_snapshots, output_dir)
    plot_Cu_spectrum(vel_snapshots, output_dir)
    print(f"Plots saved to {output_dir}/")
