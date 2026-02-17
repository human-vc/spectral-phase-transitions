import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict


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

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    cmap = LinearSegmentedColormap.from_list("phase", ["#d73027", "#fee08b", "#1a9850"])
    im = ax.imshow(
        rate_matrix,
        origin="lower",
        aspect="auto",
        extent=[min(deltas), max(deltas), min(gammas), max(gammas)],
        cmap=cmap,
        vmin=0, vmax=1,
        interpolation="bilinear",
    )

    delta_theory = np.linspace(min(deltas), max(deltas), 200)
    gamma_theory = [compute_gamma_star(d) for d in delta_theory]
    ax.plot(delta_theory, gamma_theory, "k--", linewidth=2.5, label=r"$\gamma^* = \frac{4}{2 + 3\delta}$")

    for key, vals in convergence.items():
        delta, gamma = key
        rate = np.mean(vals)
        marker = "o" if rate > 0.5 else "x"
        color = "white" if rate > 0.5 else "black"
        ax.plot(delta, gamma, marker, color=color, markersize=5, markeredgewidth=1.5, alpha=0.7)

    ax.set_xlabel(r"$\delta = d/n$", fontsize=14)
    ax.set_ylabel(r"$\gamma = m/n$", fontsize=14)
    ax.set_title("Phase Boundary: Convergence to Global Minimum", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Convergence Rate", fontsize=12)

    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spectral_gap(data, output_path):
    results = data["results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    deltas = sorted(set(r["delta"] for r in results))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(deltas)))

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
                     fmt="o-", color=colors[i], capsize=3, markersize=5,
                     label=rf"$\delta={delta:.2f}$")

    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$\gamma - \gamma^*$", fontsize=13)
    ax.set_ylabel(r"$|\lambda_{\min}|$", fontsize=13)
    ax.set_title("Spectral Gap Near Phase Boundary", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.tick_params(labelsize=11)

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
        ax2.scatter(saddle_offsets, saddle_gaps, c="#d73027", alpha=0.5, s=25, label="Saddle points")
        x_fit = np.linspace(min(saddle_offsets), max(saddle_offsets), 100)
        if len(saddle_offsets) > 2:
            coeffs = np.polyfit(np.log(saddle_offsets), np.log(saddle_gaps), 1)
            ax2.plot(x_fit, np.exp(coeffs[1]) * x_fit ** coeffs[0],
                     "--", color="#d73027", linewidth=2,
                     label=rf"Slope: {coeffs[0]:.2f} (theory: 0.5)")

    if local_min_offsets:
        ax2.scatter(local_min_offsets, local_min_gaps, c="#1a9850", alpha=0.5, s=25, label="Local minima")
        if len(local_min_offsets) > 2:
            coeffs2 = np.polyfit(np.log(local_min_offsets), np.log(local_min_gaps), 1)
            x_fit2 = np.linspace(min(local_min_offsets), max(local_min_offsets), 100)
            ax2.plot(x_fit2, np.exp(coeffs2[1]) * x_fit2 ** coeffs2[0],
                     "--", color="#1a9850", linewidth=2,
                     label=rf"Slope: {coeffs2[0]:.2f} (theory: 1.0)")

    ax2.set_xlabel(r"$|\gamma - \gamma^*|$", fontsize=13)
    ax2.set_ylabel(r"$|\lambda_{\min}|$", fontsize=13)
    ax2.set_title("Scaling Law Verification", fontsize=14)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spurious_minima(data, output_path):
    results = data["results"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    deltas = sorted(set(r["delta"] for r in results))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(deltas)))

    for i, delta in enumerate(deltas):
        subset = [r for r in results if abs(r["delta"] - delta) < 1e-6]
        gammas = sorted(set(r["gamma"] for r in subset))
        gs = compute_gamma_star(delta)

        spurious_counts = []
        gamma_vals = []
        for gamma in gammas:
            runs = [r for r in subset if abs(r["gamma"] - gamma) < 1e-6]
            spurious = sum(1 for r in runs if r["is_local_min"] and r["final_loss"] > 1e-3)
            total = len(runs)
            gamma_vals.append(gamma)
            spurious_counts.append(spurious)

        ax.plot(gamma_vals, spurious_counts, "o-", color=colors[i], markersize=6,
                label=rf"$\delta={delta:.2f}$, $\gamma^*={gs:.2f}$")
        ax.axvline(x=gs, color=colors[i], linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xlabel(r"$\gamma = m/n$", fontsize=13)
    ax.set_ylabel("Number of Spurious Local Minima", fontsize=13)
    ax.set_title("Spurious Minima vs. Network Width", fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=-0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
