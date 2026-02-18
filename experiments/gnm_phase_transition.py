"""
Gradient Norm Minimization (GNM) Phase Transition Experiments
=============================================================
Instead of hoping GD gets stuck in spurious minima (it doesn't),
we ACTIVELY SEARCH for critical points by minimizing ||grad L||^2.

This is the correct experimental approach because:
  - GD always converges to global min (even below gamma*) -- paper's key finding
  - But spurious critical points DO exist below gamma*
  - GNM finds them by treating gradient-norm-squared as the objective
  - We can then classify found critical points by their loss value

Experimental pipeline:
  1. Generate teacher-student data
  2. For each (delta, gamma): run many GNM optimizations from random inits
  3. At each found critical point: record loss, Hessian min eigenvalue
  4. Plot: fraction of found critical points that are spurious (loss > threshold)
  5. This fraction should jump at gamma* -- the phase transition becomes visible

Author: Claw (deep work session, Feb 17 2026)
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Model
# ============================================================================

class TwoLayerSoftplus(nn.Module):
    def __init__(self, d: int, m: int, beta: float = 5.0):
        super().__init__()
        self.d = d
        self.m = m
        self.beta = beta
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(F.softplus(self.fc1(x), beta=self.beta))


def gamma_star_approx(delta: float) -> float:
    return 4.0 / (2.0 + 3.0 * delta)


# ============================================================================
# GNM: Minimize ||grad L(theta)||^2
# ============================================================================

@dataclass
class GNMResult:
    seed: int
    loss_at_critical: float
    grad_norm_at_critical: float
    is_critical: bool           # grad norm < threshold
    is_spurious: bool           # critical AND loss > global_threshold
    is_global: bool             # critical AND loss < global_threshold
    min_hessian_eig: Optional[float]
    n_negative_eigs: int
    is_local_min: bool          # all Hessian eigenvalues >= -tolerance
    steps: int


def run_gnm(
    X: torch.Tensor,
    y: torch.Tensor,
    m: int,
    beta: float,
    seed: int,
    lr: float = 1e-3,
    max_steps: int = 15000,
    grad_threshold: float = 1e-4,
    global_loss_threshold: float = 1e-4,
    compute_hessian: bool = True,
) -> GNMResult:
    """
    Gradient Norm Minimization: find critical points by minimizing ||grad L||^2.
    Uses Adam on the squared gradient norm objective.
    """
    n, d = X.shape
    torch.manual_seed(seed)

    model = TwoLayerSoftplus(d, m, beta=beta)
    nn.init.normal_(model.fc1.weight, std=1.0 / np.sqrt(d))
    nn.init.normal_(model.fc2.weight, std=1.0 / np.sqrt(m))

    # Adam on ||grad L||^2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(max_steps):
        # Forward pass for L
        pred = model(X)
        loss = F.mse_loss(pred, y)

        # Compute grad L
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_norm_sq = sum((g ** 2).sum() for g in grads)

        # Backward pass on ||grad L||^2
        optimizer.zero_grad()
        grad_norm_sq.backward()
        optimizer.step()

        if grad_norm_sq.item() < grad_threshold ** 2:
            break

    # Evaluate at found point
    with torch.no_grad():
        pred = model(X)
        final_loss = F.mse_loss(pred, y).item()

    # Recompute grad norm (no graph)
    pred = model(X)
    loss = F.mse_loss(pred, y)
    grads = torch.autograd.grad(loss, model.parameters())
    final_grad_norm = sum((g ** 2).sum().item() for g in grads) ** 0.5

    is_critical = final_grad_norm < grad_threshold
    is_global = is_critical and final_loss < global_loss_threshold
    is_spurious = is_critical and final_loss >= global_loss_threshold

    # Hessian analysis at critical points
    min_eig = None
    n_negative = 0
    is_local_min = False

    if compute_hessian and is_critical:
        try:
            min_eig, n_negative, is_local_min = _hessian_analysis(model, X, y)
        except Exception:
            pass

    return GNMResult(
        seed=seed,
        loss_at_critical=final_loss,
        grad_norm_at_critical=final_grad_norm,
        is_critical=is_critical,
        is_spurious=is_spurious,
        is_global=is_global,
        min_hessian_eig=min_eig,
        n_negative_eigs=n_negative,
        is_local_min=is_local_min,
        steps=step + 1,
    )


def _hessian_analysis(model, X, y, neg_tolerance=-1e-5):
    """Compute Hessian eigenvalues at current parameters."""
    params = list(model.parameters())
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    total = sum(numels)

    if total > 3000:
        return None, 0, False

    def loss_fn(flat_params):
        idx = 0
        w1 = flat_params[idx:idx + numels[0]].view(shapes[0])
        idx += numels[0]
        w2 = flat_params[idx:idx + numels[1]].view(shapes[1])
        h = F.softplus(X @ w1.t(), beta=model.beta)
        pred = h @ w2.t()
        return F.mse_loss(pred, y)

    flat = torch.cat([p.detach().clone().view(-1) for p in params])
    H = torch.autograd.functional.hessian(loss_fn, flat).detach()
    H = 0.5 * (H + H.t())

    eigs = torch.linalg.eigvalsh(H)
    min_eig = eigs[0].item()
    n_negative = int((eigs < neg_tolerance).sum().item())
    is_local_min = n_negative == 0

    return min_eig, n_negative, is_local_min


# ============================================================================
# Experiment 1: GNM Phase Boundary
# ============================================================================

def experiment_gnm_boundary(
    n: int = 50,
    deltas: List[float] = None,
    gamma_range: Tuple[float, float] = (0.2, 2.5),
    n_gammas: int = 12,
    num_seeds: int = 30,
    beta: float = 5.0,
    lr: float = 1e-3,
    max_steps: int = 15000,
    output_dir: str = "experiments/results/gnm",
):
    """
    For each (delta, gamma): run GNM from many random inits.
    Measure what fraction of found critical points are spurious vs global.
    """
    if deltas is None:
        deltas = [0.25, 0.5, 0.75, 1.0]

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gammas).tolist()
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    total = len(deltas) * len(gammas)
    done = 0

    for delta in deltas:
        d = max(1, int(delta * n))
        gs = gamma_star_approx(delta)
        m_teacher = max(2, n // 4)

        # Generate data
        torch.manual_seed(42)
        X = torch.randn(n, d)
        teacher = TwoLayerSoftplus(d, m_teacher, beta=beta)
        nn.init.normal_(teacher.fc1.weight, std=2.0 / np.sqrt(d))
        nn.init.normal_(teacher.fc2.weight, std=2.0 / np.sqrt(m_teacher))
        with torch.no_grad():
            y = teacher(X)

        delta_results = []
        for gamma in gammas:
            m = max(1, int(gamma * n))
            done += 1

            n_critical = 0
            n_spurious = 0
            n_global = 0
            n_spurious_minima = 0  # spurious AND local minimum
            losses = []
            min_eigs = []

            for seed in range(num_seeds):
                res = run_gnm(
                    X, y, m, beta=beta, seed=seed,
                    lr=lr, max_steps=max_steps,
                    compute_hessian=(d * m + m <= 3000),
                )
                if res.is_critical:
                    n_critical += 1
                    losses.append(res.loss_at_critical)
                    if res.min_hessian_eig is not None:
                        min_eigs.append(res.min_hessian_eig)
                if res.is_spurious:
                    n_spurious += 1
                    if res.is_local_min:
                        n_spurious_minima += 1
                if res.is_global:
                    n_global += 1

            frac_spurious = n_spurious / max(n_critical, 1)
            frac_critical = n_critical / num_seeds

            entry = {
                "gamma": gamma,
                "m": m,
                "n_critical": n_critical,
                "n_spurious": n_spurious,
                "n_global": n_global,
                "n_spurious_minima": n_spurious_minima,
                "frac_spurious": frac_spurious,
                "frac_critical": frac_critical,
                "median_loss": float(np.median(losses)) if losses else None,
                "losses_at_critical": losses,
                "min_eigs": min_eigs,
            }
            delta_results.append(entry)

            status = "BELOW" if gamma < gs else "ABOVE"
            print(
                f"[{done}/{total}] delta={delta:.2f}, gamma={gamma:.2f} ({status} gs={gs:.2f}), "
                f"critical={n_critical}/{num_seeds}, spurious={n_spurious}, "
                f"spurious_minima={n_spurious_minima}, frac_spurious={frac_spurious:.0%}",
                flush=True,
            )

        results[str(delta)] = {
            "delta": delta,
            "d": d,
            "gamma_star": gs,
            "results": delta_results,
        }

    outpath = os.path.join(output_dir, "gnm_boundary.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


# ============================================================================
# Experiment 2: GNM Spectral Analysis Near Boundary
# ============================================================================

def experiment_gnm_spectral(
    n: int = 40,
    delta: float = 0.5,
    gamma_offsets: List[float] = None,
    num_seeds: int = 40,
    beta: float = 5.0,
    lr: float = 1e-3,
    max_steps: int = 20000,
    output_dir: str = "experiments/results/gnm",
):
    """
    Zoom into the phase boundary: many GNM runs near gamma*.
    For each found critical point: compute full Hessian spectrum.
    """
    if gamma_offsets is None:
        gamma_offsets = [-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5]

    gs = gamma_star_approx(delta)
    d = max(1, int(delta * n))
    m_teacher = max(2, n // 4)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(42)
    X = torch.randn(n, d)
    teacher = TwoLayerSoftplus(d, m_teacher, beta=beta)
    nn.init.normal_(teacher.fc1.weight, std=2.0 / np.sqrt(d))
    nn.init.normal_(teacher.fc2.weight, std=2.0 / np.sqrt(m_teacher))
    with torch.no_grad():
        y = teacher(X)

    results = []
    for offset in gamma_offsets:
        gamma = gs + offset
        if gamma <= 0:
            continue
        m = max(1, int(gamma * n))

        print(f"\ngamma={gamma:.3f} (offset={offset:+.2f}), m={m}:", flush=True)

        for seed in range(num_seeds):
            res = run_gnm(
                X, y, m, beta=beta, seed=seed,
                lr=lr, max_steps=max_steps,
                compute_hessian=True,
            )

            entry = {
                "delta": delta,
                "gamma": float(gamma),
                "gamma_star": gs,
                "gamma_offset": float(offset),
                "seed": seed,
                **asdict(res),
            }
            results.append(entry)

            if res.is_critical:
                tag = "GLOBAL" if res.is_global else ("SPUR-MIN" if res.is_local_min else "SPUR-SADDLE")
                eig_str = f"min_eig={res.min_hessian_eig:.4e}" if res.min_hessian_eig is not None else ""
                print(f"  seed={seed}: [{tag}] loss={res.loss_at_critical:.2e} {eig_str}", flush=True)

    outpath = os.path.join(output_dir, "gnm_spectral.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


# ============================================================================
# Experiment 3: Loss Distribution at Critical Points
# ============================================================================

def experiment_loss_distribution(
    n: int = 50,
    delta: float = 0.5,
    gammas: List[float] = None,
    num_seeds: int = 50,
    beta: float = 5.0,
    lr: float = 1e-3,
    max_steps: int = 15000,
    output_dir: str = "experiments/results/gnm",
):
    """
    For a few gamma values: run many GNM and plot the DISTRIBUTION of losses
    at found critical points. Below gamma*, should see bimodal (zero + nonzero).
    Above gamma*, should see unimodal (all near zero).
    """
    gs = gamma_star_approx(delta)
    if gammas is None:
        gammas = [0.3 * gs, 0.5 * gs, 0.8 * gs, gs, 1.2 * gs, 1.5 * gs, 2.0 * gs]

    d = max(1, int(delta * n))
    m_teacher = max(2, n // 4)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(42)
    X = torch.randn(n, d)
    teacher = TwoLayerSoftplus(d, m_teacher, beta=beta)
    nn.init.normal_(teacher.fc1.weight, std=2.0 / np.sqrt(d))
    nn.init.normal_(teacher.fc2.weight, std=2.0 / np.sqrt(m_teacher))
    with torch.no_grad():
        y = teacher(X)

    results = {}
    for gamma in gammas:
        m = max(1, int(gamma * n))
        ratio = gamma / gs
        print(f"\ngamma={gamma:.3f} ({ratio:.1f}x gamma*), m={m}:", flush=True)

        losses = []
        grad_norms = []
        types = []  # 'global', 'spurious_min', 'spurious_saddle', 'not_critical'

        for seed in range(num_seeds):
            res = run_gnm(
                X, y, m, beta=beta, seed=seed,
                lr=lr, max_steps=max_steps,
                compute_hessian=True,
            )

            losses.append(res.loss_at_critical)
            grad_norms.append(res.grad_norm_at_critical)

            if res.is_global:
                types.append("global")
            elif res.is_spurious and res.is_local_min:
                types.append("spurious_min")
            elif res.is_spurious:
                types.append("spurious_saddle")
            else:
                types.append("not_critical")

        n_global = types.count("global")
        n_spur_min = types.count("spurious_min")
        n_spur_sad = types.count("spurious_saddle")
        n_not = types.count("not_critical")
        print(f"  global={n_global}, spur_min={n_spur_min}, spur_saddle={n_spur_sad}, not_crit={n_not}", flush=True)

        results[f"{gamma:.4f}"] = {
            "gamma": float(gamma),
            "gamma_ratio": ratio,
            "m": m,
            "losses": losses,
            "grad_norms": grad_norms,
            "types": types,
            "n_global": n_global,
            "n_spurious_min": n_spur_min,
            "n_spurious_saddle": n_spur_sad,
            "n_not_critical": n_not,
        }

    outpath = os.path.join(output_dir, "gnm_loss_distribution.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_gnm_boundary(results_path, output_dir):
    """Plot fraction of spurious critical points vs gamma for each delta."""
    with open(results_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))

    for idx, (delta_key, info) in enumerate(sorted(data.items(), key=lambda x: float(x[0]))):
        delta = info["delta"]
        gs = info["gamma_star"]
        gammas = [r["gamma"] for r in info["results"]]
        frac_spurious = [r["frac_spurious"] for r in info["results"]]
        frac_critical = [r["frac_critical"] for r in info["results"]]

        # Normalize gamma by gamma*
        gamma_ratios = [g / gs for g in gammas]

        ax1.plot(gamma_ratios, frac_spurious, "o-", color=colors[idx],
                 label=f"$\\delta={delta}$", linewidth=2, markersize=4)
        ax2.plot(gamma_ratios, frac_critical, "o-", color=colors[idx],
                 label=f"$\\delta={delta}$", linewidth=2, markersize=4)

    ax1.axvline(x=1.0, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                label=r"$\gamma = \gamma^\star$")
    ax1.set_xlabel(r"$\gamma / \gamma^\star$")
    ax1.set_ylabel("Fraction of Critical Points That Are Spurious")
    ax1.set_title("Spurious Critical Points (GNM)")
    ax1.legend(frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.axvline(x=1.0, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax2.set_xlabel(r"$\gamma / \gamma^\star$")
    ax2.set_ylabel("Fraction of Runs Finding Critical Points")
    ax2.set_title("GNM Convergence Rate")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Phase Transition via Gradient Norm Minimization", fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(output_dir, "fig_gnm_boundary.pdf")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved {outpath}")


def plot_loss_distribution(results_path, output_dir):
    """Histogram panels of loss at critical points for different gamma/gamma* ratios."""
    with open(results_path) as f:
        data = json.load(f)

    keys = sorted(data.keys(), key=lambda k: float(k))
    n_panels = len(keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for idx, key in enumerate(keys):
        ax = axes[idx]
        info = data[key]
        ratio = info["gamma_ratio"]
        losses = info["losses"]
        types = info["types"]

        # Color by type
        global_losses = [l for l, t in zip(losses, types) if t == "global"]
        spur_min_losses = [l for l, t in zip(losses, types) if t == "spurious_min"]
        spur_sad_losses = [l for l, t in zip(losses, types) if t == "spurious_saddle"]
        not_crit = [l for l, t in zip(losses, types) if t == "not_critical"]

        bins = np.linspace(0, max(max(losses), 0.01), 25)

        if global_losses:
            ax.hist(global_losses, bins=bins, alpha=0.7, color="green", label="Global min")
        if spur_min_losses:
            ax.hist(spur_min_losses, bins=bins, alpha=0.7, color="red", label="Spur. min")
        if spur_sad_losses:
            ax.hist(spur_sad_losses, bins=bins, alpha=0.7, color="orange", label="Spur. saddle")
        if not_crit:
            ax.hist(not_crit, bins=bins, alpha=0.4, color="gray", label="Not crit.")

        ax.set_xlabel("Loss")
        if idx == 0:
            ax.set_ylabel("Count")
        ax.set_title(f"$\\gamma/\\gamma^\\star = {ratio:.1f}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Loss Distribution at GNM-Found Critical Points", fontsize=13, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(output_dir, "fig_gnm_loss_distribution.pdf")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved {outpath}")


def plot_all(output_dir="experiments/results/gnm"):
    plotters = {
        "gnm_boundary.json": plot_gnm_boundary,
        "gnm_loss_distribution.json": plot_loss_distribution,
    }
    for filename, plotter in plotters.items():
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            print(f"\nPlotting {filename}...")
            plotter(path, output_dir)
        else:
            print(f"Skipping {filename}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GNM Phase Transition Experiments")
    parser.add_argument("experiment", choices=[
        "boundary", "spectral", "loss_dist", "quick", "full", "plot",
    ])
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--output", type=str, default="experiments/results/gnm")
    args = parser.parse_args()

    torch.set_num_threads(4)
    t0 = time.time()

    if args.experiment == "boundary":
        experiment_gnm_boundary(
            n=args.n or 50, beta=args.beta, lr=args.lr, max_steps=args.steps,
            num_seeds=args.seeds or 30, output_dir=args.output,
        )

    elif args.experiment == "spectral":
        experiment_gnm_spectral(
            n=args.n or 40, delta=args.delta or 0.5,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            num_seeds=args.seeds or 40, output_dir=args.output,
        )

    elif args.experiment == "loss_dist":
        experiment_loss_distribution(
            n=args.n or 50, delta=args.delta or 0.5,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            num_seeds=args.seeds or 50, output_dir=args.output,
        )

    elif args.experiment == "quick":
        print("=" * 60)
        print("QUICK GNM SUITE")
        print("=" * 60)
        experiment_gnm_boundary(
            n=30, deltas=[0.5, 1.0], n_gammas=8, num_seeds=15,
            beta=args.beta, lr=args.lr, max_steps=8000,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_loss_distribution(
            n=30, delta=0.5, num_seeds=20,
            gammas=None,  # auto: fractions of gamma*
            beta=args.beta, lr=args.lr, max_steps=8000,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        plot_all(args.output)

    elif args.experiment == "full":
        print("=" * 60)
        print("FULL GNM SUITE")
        print("=" * 60)
        experiment_gnm_boundary(
            n=50, deltas=[0.25, 0.5, 0.75, 1.0], n_gammas=15, num_seeds=40,
            beta=args.beta, lr=args.lr, max_steps=20000,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_loss_distribution(
            n=50, delta=0.5, num_seeds=60,
            beta=args.beta, lr=args.lr, max_steps=20000,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_gnm_spectral(
            n=40, delta=0.5, num_seeds=50,
            beta=args.beta, lr=args.lr, max_steps=20000,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        plot_all(args.output)

    elif args.experiment == "plot":
        plot_all(args.output)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
