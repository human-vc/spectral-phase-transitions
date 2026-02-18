"""
Publication-quality plots for softplus phase transition experiments.
Generates Figures 4-6 replacements with clean, visible phase transitions.

Author: Claw (deep work session, Feb 17 2026)
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def gamma_star_approx(delta):
    return 4.0 / (2.0 + 3.0 * delta)


# ============================================================================
# Plot 1: Phase Boundary Heatmap with Success Rate
# ============================================================================

def plot_phase_boundary(results_path, output_dir):
    """
    Two-panel figure:
      Left: log-loss heatmap in (delta, gamma) plane with theoretical boundary
      Right: success rate heatmap (fraction that reach global minimum)
    """
    with open(results_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    deltas = sorted([float(k) for k in data.keys()])
    gammas = [r["gamma"] for r in data[str(deltas[0])]["results"]]

    # Build matrices
    loss_matrix = np.full((len(deltas), len(gammas)), np.nan)
    success_matrix = np.full((len(deltas), len(gammas)), np.nan)

    for i, delta in enumerate(deltas):
        for j, entry in enumerate(data[str(delta)]["results"]):
            loss_matrix[i, j] = np.log10(entry["median_loss"] + 1e-15)
            success_matrix[i, j] = entry["success_rate"]

    D, G = np.meshgrid(deltas, gammas, indexing="ij")

    # Left: Loss heatmap
    c1 = ax1.pcolormesh(D, G, loss_matrix, cmap="RdYlBu_r", shading="gouraud")
    fig.colorbar(c1, ax=ax1, label=r"$\log_{10}(\mathrm{loss})$", shrink=0.85)

    # Right: Success rate heatmap
    c2 = ax2.pcolormesh(D, G, success_matrix, cmap="RdYlGn", shading="gouraud",
                        vmin=0, vmax=1)
    fig.colorbar(c2, ax=ax2, label="Success rate (global minimum)", shrink=0.85)

    # Theoretical boundary on both
    d_theory = np.linspace(0.05, max(deltas) + 0.1, 200)
    g_theory = [gamma_star_approx(d) for d in d_theory]

    for ax, title in [(ax1, "Training Loss"), (ax2, "Convergence to Global Min")]:
        ax.plot(d_theory, g_theory, "k--", linewidth=2.5,
                label=r"$\gamma^\star \approx 4/(2+3\delta)$")
        ax.set_xlabel(r"$\delta = d/n$")
        ax.set_ylabel(r"$\gamma = m/n$")
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(deltas[0], deltas[-1])
        ax.set_ylim(gammas[0], gammas[-1])

    fig.suptitle(r"Phase Transition: Softplus($\beta=1$) + Vanilla GD", fontsize=14, y=1.02)
    fig.tight_layout()

    outpath = os.path.join(output_dir, "fig_phase_boundary_softplus.pdf")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved {outpath}")


# ============================================================================
# Plot 2: Spectral Gap vs Gamma Offset
# ============================================================================

def plot_spectral_gap(results_path, output_dir):
    """
    Scatter + mean line of min Hessian eigenvalue vs (gamma - gamma*).
    Should show: negative eigenvalues below gamma*, positive above.
    """
    with open(results_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    offsets = sorted(set(r["gamma_offset"] for r in data))

    # Left: min eigenvalue
    for off in offsets:
        subset = [r for r in data if r["gamma_offset"] == off and r["min_hessian_eigenvalue"] is not None]
        if not subset:
            continue
        eigs = [r["min_hessian_eigenvalue"] for r in subset]
        ax1.scatter([off] * len(eigs), eigs, alpha=0.4, s=25, c="steelblue", edgecolors="none")

    # Mean line
    offset_means = []
    mean_eigs = []
    for off in offsets:
        subset = [r for r in data if r["gamma_offset"] == off and r["min_hessian_eigenvalue"] is not None]
        if subset:
            offset_means.append(off)
            mean_eigs.append(np.mean([r["min_hessian_eigenvalue"] for r in subset]))

    ax1.plot(offset_means, mean_eigs, "k-o", linewidth=2, markersize=5, zorder=10)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
    ax1.set_xlabel(r"$\gamma - \gamma^\star$")
    ax1.set_ylabel("Min Hessian Eigenvalue")
    ax1.set_title("Spectral Gap at Convergence")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: success rate vs offset
    for off in offsets:
        subset = [r for r in data if r["gamma_offset"] == off]
        if not subset:
            continue
        rate = sum(1 for r in subset if r["converged_global"]) / len(subset)
        ax2.bar(off, rate, width=0.08, color="steelblue", alpha=0.7)

    ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
    ax2.set_xlabel(r"$\gamma - \gamma^\star$")
    ax2.set_ylabel("Fraction Converging to Global Min")
    ax2.set_title("Success Rate Near Phase Boundary")
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    outpath = os.path.join(output_dir, "fig_spectral_gap_softplus.pdf")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved {outpath}")


# ============================================================================
# Plot 3: Spectral Density Histograms
# ============================================================================

def plot_spectral_density(results_path, output_dir):
    """
    Panel of histograms: Hessian eigenvalue distributions at different gamma/gamma* ratios.
    """
    with open(results_path) as f:
        data = json.load(f)

    gammas = sorted(data.keys(), key=lambda k: float(k))
    n_panels = len(gammas)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, n_panels))

    for idx, gamma_key in enumerate(gammas):
        ax = axes[idx]
        info = data[gamma_key]
        ratio = info["gamma_ratio"]

        # Pool all eigenvalues across seeds
        all_eigs = []
        for eigset in info["eigenvalue_sets"]:
            all_eigs.extend(eigset)

        if not all_eigs:
            ax.set_title(f"$\\gamma/\\gamma^\\star = {ratio:.1f}$\n(no data)")
            continue

        all_eigs = np.array(all_eigs)
        ax.hist(all_eigs, bins=50, density=True, color=colors[idx], alpha=0.7, edgecolor="none")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Eigenvalue")
        if idx == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"$\\gamma/\\gamma^\\star = {ratio:.1f}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Hessian Spectral Density at Critical Points", fontsize=13, y=1.02)
    fig.tight_layout()

    outpath = os.path.join(output_dir, "fig_spectral_density_softplus.pdf")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved {outpath}")


# ============================================================================
# Plot 4: Activation Comparison
# ============================================================================

def plot_activation_comparison(results_path, output_dir):
    """
    Overlay success-rate curves for different softplus betas.
    Shows that smoother activations make the transition sharper.
    """
    with open(results_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"beta_1.0": "C0", "beta_5.0": "C1", "beta_20.0": "C2"}
    labels = {"beta_1.0": r"softplus($\beta=1$)", "beta_5.0": r"softplus($\beta=5$)", "beta_20.0": r"softplus($\beta=20$)"}

    gs = None
    for key, info in data.items():
        gammas = [r["gamma"] for r in info["results"]]
        success_rates = [r["success_rate"] for r in info["results"]]
        median_losses = [r["median_loss"] for r in info["results"]]
        gs = info["gamma_star"]

        ax1.plot(gammas, success_rates, "o-", color=colors.get(key, "gray"),
                 label=labels.get(key, key), linewidth=2, markersize=4)
        ax2.semilogy(gammas, [max(l, 1e-15) for l in median_losses], "o-",
                     color=colors.get(key, "gray"), label=labels.get(key, key),
                     linewidth=2, markersize=4)

    for ax in [ax1, ax2]:
        if gs:
            ax.axvline(x=gs, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                       label=r"$\gamma^\star$")
        ax.set_xlabel(r"$\gamma = m/n$")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_ylabel("Success Rate")
    ax1.set_title("Convergence to Global Minimum")
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_ylabel("Median Loss")
    ax2.set_title("Training Loss")

    fig.suptitle("Activation Smoothness and Phase Transition Visibility", fontsize=13, y=1.02)
    fig.tight_layout()

    outpath = os.path.join(output_dir, "fig_activation_comparison.pdf")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved {outpath}")


# ============================================================================
# Plot 5: Convergence Time (Critical Slowing Down)
# ============================================================================

def plot_convergence_time(results_path, output_dir):
    """
    Convergence time vs gamma offset -- should diverge near gamma*.
    """
    with open(results_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))

    offsets = [r["gamma_offset"] for r in data]
    med_times = [r["median_time"] for r in data]
    success = [r["success_rate"] for r in data]

    # Plot only where we have data
    valid = [(o, t, s) for o, t, s in zip(offsets, med_times, success) if t is not None]
    if valid:
        offs, times, _ = zip(*valid)
        ax.plot(offs, times, "ko-", linewidth=2, markersize=6)

        # Color by success rate
        for o, t, s in valid:
            color = plt.cm.RdYlGn(s)
            ax.scatter([o], [t], c=[color], s=80, edgecolors="black", linewidths=0.5, zorder=10)

    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
               label=r"$\gamma^\star$")
    ax.set_xlabel(r"$\gamma - \gamma^\star$")
    ax.set_ylabel("Steps to Converge (median)")
    ax.set_title("Critical Slowing Down Near Phase Boundary")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    outpath = os.path.join(output_dir, "fig_convergence_time_softplus.pdf")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved {outpath}")


# ============================================================================
# CLI: Plot all available results
# ============================================================================

def main():
    result_dir = "experiments/results/softplus"
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]

    os.makedirs(result_dir, exist_ok=True)

    plotters = {
        "phase_boundary_softplus.json": plot_phase_boundary,
        "spectral_gap_softplus.json": plot_spectral_gap,
        "spectral_density_softplus.json": plot_spectral_density,
        "activation_comparison.json": plot_activation_comparison,
        "convergence_time_softplus.json": plot_convergence_time,
    }

    for filename, plotter in plotters.items():
        path = os.path.join(result_dir, filename)
        if os.path.exists(path):
            print(f"\nPlotting {filename}...")
            try:
                plotter(path, result_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            print(f"Skipping {filename} (not found)")


if __name__ == "__main__":
    main()
