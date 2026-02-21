import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "blue": "#2166ac",
    "red": "#b2182b",
    "green": "#1b7837",
    "orange": "#e08214",
    "purple": "#542788",
    "gray": "#636363",
}

PALETTE = ["#2166ac", "#762a83", "#e08214", "#1b7837", "#b2182b"]


def compute_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_phase_boundary(data, output_path):
    results = data["results"]
    convergence = defaultdict(list)

    for r in results:
        key = (r["delta"], r["gamma"])
        convergence[key].append(r["converged"])

    deltas = sorted(set(r["delta"] for r in results))
    gammas = sorted(set(r["gamma"] for r in results))

    rate_matrix = np.zeros((len(gammas), len(deltas)))
    for i, gamma in enumerate(gammas):
        for j, delta in enumerate(deltas):
            vals = convergence.get((delta, gamma), [])
            rate_matrix[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))

    im = ax.imshow(
        rate_matrix,
        origin="lower",
        aspect="auto",
        extent=[min(deltas), max(deltas), min(gammas), max(gammas)],
        cmap="RdYlBu",
        vmin=0, vmax=1,
        interpolation="bilinear",
    )

    delta_theory = np.linspace(min(deltas), max(deltas), 200)
    gamma_theory = [compute_gamma_star(d) for d in delta_theory]
    ax.plot(delta_theory, gamma_theory, "k-", linewidth=1.8,
            label=r"$\gamma^\star(\delta) = \frac{4}{2 + 3\delta}$")

    ax.set_xlabel(r"$\delta = d/n$", fontsize=11)
    ax.set_ylabel(r"$\gamma = m/n$", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", frameon=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=25)
    cbar.set_label("Convergence rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.6)

    ax.tick_params(labelsize=9)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spectral_gap(data, output_path):
    results = data["results"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    deltas = sorted(set(r["delta"] for r in results))

    ax = axes[0]
    for i, delta in enumerate(deltas):
        subset = [r for r in results if abs(r["delta"] - delta) < 1e-6]
        offsets = sorted(set(r["gamma_offset"] for r in subset))

        mean_gaps = []
        std_gaps = []
        for off in offsets:
            vals = [abs(r["min_eigenvalue"]) for r in subset if abs(r["gamma_offset"] - off) < 1e-6]
            mean_gaps.append(np.mean(vals))
            std_gaps.append(np.std(vals))

        offsets_arr = np.array(offsets)
        mean_arr = np.array(mean_gaps)
        std_arr = np.array(std_gaps)

        ax.errorbar(offsets_arr, mean_arr, yerr=std_arr,
                     fmt="o-", color=PALETTE[i % len(PALETTE)], capsize=2,
                     markersize=3.5, linewidth=1.2, elinewidth=0.8,
                     label=rf"$\delta={delta:.1f}$")

    ax.axvline(x=0, color=COLORS["gray"], linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\gamma - \gamma^\star$", fontsize=11)
    ax.set_ylabel(r"$|\lambda_{\min}|$", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_yscale("log")
    ax.tick_params(labelsize=9)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

    ax2 = axes[1]
    saddle_offsets = []
    saddle_gaps = []
    local_min_offsets = []
    local_min_gaps = []

    for r in results:
        off = abs(r["gamma_offset"])
        if off < 1e-6:
            continue
        gap = abs(r["min_eigenvalue"])
        if gap < 1e-10:
            continue
        if r["is_saddle"]:
            saddle_offsets.append(off)
            saddle_gaps.append(gap)
        elif r["is_local_min"]:
            local_min_offsets.append(off)
            local_min_gaps.append(gap)

    if saddle_offsets:
        ax2.scatter(saddle_offsets, saddle_gaps, c=COLORS["red"], alpha=0.4,
                    s=15, linewidths=0, label="Saddle points")
        if len(saddle_offsets) > 2:
            coeffs = np.polyfit(np.log(saddle_offsets), np.log(saddle_gaps), 1)
            x_fit = np.linspace(min(saddle_offsets), max(saddle_offsets), 100)
            ax2.plot(x_fit, np.exp(coeffs[1]) * x_fit ** coeffs[0],
                     "-", color=COLORS["red"], linewidth=1.5,
                     label=rf"slope = {coeffs[0]:.2f}")

    if local_min_offsets:
        ax2.scatter(local_min_offsets, local_min_gaps, c=COLORS["blue"], alpha=0.4,
                    s=15, linewidths=0, label="Local minima")
        if len(local_min_offsets) > 2:
            coeffs2 = np.polyfit(np.log(local_min_offsets), np.log(local_min_gaps), 1)
            x_fit2 = np.linspace(min(local_min_offsets), max(local_min_offsets), 100)
            ax2.plot(x_fit2, np.exp(coeffs2[1]) * x_fit2 ** coeffs2[0],
                     "-", color=COLORS["blue"], linewidth=1.5,
                     label=rf"slope = {coeffs2[0]:.2f}")

    ax2.set_xlabel(r"$|\gamma - \gamma^\star|$", fontsize=11)
    ax2.set_ylabel(r"$|\lambda_{\min}|$", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, frameon=False)
    ax2.tick_params(labelsize=9)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=11,
             fontweight="bold", va="top", ha="left")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spurious_minima(data, output_path):
    results = data["results"]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    deltas = sorted(set(r["delta"] for r in results))

    for i, delta in enumerate(deltas):
        subset = [r for r in results if abs(r["delta"] - delta) < 1e-6]
        gammas = sorted(set(r["gamma"] for r in subset))
        gs = compute_gamma_star(delta)

        spurious_counts = []
        gamma_vals = []
        for gamma in gammas:
            runs = [r for r in subset if abs(r["gamma"] - gamma) < 1e-6]
            spurious = sum(1 for r in runs if r["is_local_min"] and r["final_loss"] > 1e-3)
            gamma_vals.append(gamma)
            spurious_counts.append(spurious)

        ax.plot(gamma_vals, spurious_counts, "o-", color=PALETTE[i % len(PALETTE)],
                markersize=4, linewidth=1.2,
                label=rf"$\delta={delta:.1f}$, $\gamma^\star={gs:.2f}$")
        ax.axvline(x=gs, color=PALETTE[i % len(PALETTE)], linestyle=":", alpha=0.4, linewidth=0.8)

    ax.set_xlabel(r"$\gamma = m/n$", fontsize=11)
    ax.set_ylabel("Spurious local minima", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=9)
    ax.set_ylim(bottom=-0.5)
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-boundary", type=str, default="results/phase_boundary.json")
    parser.add_argument("--spectral-gap", type=str, default="results/spectral_gap.json")
    parser.add_argument("--output-dir", type=str, default="../paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating figures...")

    if os.path.exists(args.phase_boundary):
        phase_data = load_json(args.phase_boundary)
        plot_phase_boundary(phase_data, os.path.join(args.output_dir, "fig1_phase_boundary.pdf"))
    else:
        print(f"  Skipping Fig 1: {args.phase_boundary} not found")

    if os.path.exists(args.spectral_gap):
        spectral_data = load_json(args.spectral_gap)
        plot_spectral_gap(spectral_data, os.path.join(args.output_dir, "fig2_spectral_gap.pdf"))
        plot_spurious_minima(spectral_data, os.path.join(args.output_dir, "fig3_spurious_minima.pdf"))
    else:
        print(f"  Skipping Figs 2-3: {args.spectral_gap} not found")

    print("All figures generated.")


if __name__ == "__main__":
    main()
