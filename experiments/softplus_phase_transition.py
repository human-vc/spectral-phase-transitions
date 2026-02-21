"""
Softplus Phase Transition Experiments
=====================================
Designed to make the topological phase transition VISIBLE experimentally.

Key design choices vs. the existing code:
  1. softplus(beta=1) -- genuinely smooth, well-conditioned Hessians
  2. Vanilla gradient descent -- no momentum, no adaptation, no escaping minima
  3. Small learning rate + fixed steps -- ensures GD stays near its initialization basin
  4. Multiple random restarts -- measures failure rate across the boundary
  5. Full Hessian eigenvalue computation -- spectral gap analysis at convergence
  6. Convergence threshold tracking -- distinguishes global vs spurious minima

Theory: gamma* = 4/(2+3*delta) for approximate boundary.
        Below gamma*, spurious local minima exist exponentially.
        Above gamma*, all local minima are global.

If we use a weak enough optimizer (vanilla GD), it SHOULD get stuck below gamma*
at least some fraction of the time.

Author: Claw (deep work session, Feb 17 2026)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model
# ============================================================================

class TwoLayerSoftplus(nn.Module):
    """Two-layer network with softplus activation (smooth ReLU)."""

    def __init__(self, d: int, m: int, beta: float = 1.0):
        super().__init__()
        self.d = d
        self.m = m
        self.beta = beta
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(F.softplus(self.fc1(x), beta=self.beta))


# ============================================================================
# Data Generation (Teacher-Student)
# ============================================================================

def generate_teacher_student_data(
    n: int, d: int, m_teacher: int, beta: float, noise_std: float = 0.0, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data from a teacher network (realizable setting).
    The theory assumes this setup.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    X = torch.randn(n, d, generator=rng)

    teacher = TwoLayerSoftplus(d, m_teacher, beta=beta)
    with torch.no_grad():
        nn.init.normal_(teacher.fc1.weight, std=2.0 / np.sqrt(d), generator=rng)
        nn.init.normal_(teacher.fc2.weight, std=2.0 / np.sqrt(m_teacher), generator=rng)
        y = teacher(X)
        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y)

    return X, y


# ============================================================================
# Training: Vanilla Gradient Descent
# ============================================================================

@dataclass
class TrainResult:
    """Result of a single training run."""
    seed: int
    init_loss: float
    final_loss: float
    converged_global: bool      # loss < global_threshold
    converged_spurious: bool    # grad small but loss > global_threshold
    min_hessian_eigenvalue: Optional[float] = None
    hessian_spectral_gap: Optional[float] = None
    hessian_eigenvalues: Optional[List[float]] = None
    steps_to_converge: Optional[int] = None
    final_grad_norm: float = 0.0
    loss_trajectory: Optional[List[float]] = None


def train_vanilla_gd(
    X: torch.Tensor,
    y: torch.Tensor,
    m: int,
    beta: float,
    seed: int,
    lr: float = 1e-3,
    max_steps: int = 30000,
    global_threshold: float = 1e-6,
    grad_threshold: float = 1e-5,
    record_trajectory: bool = False,
    compute_hessian: bool = True,
) -> TrainResult:
    """
    Train with vanilla gradient descent (no momentum, no adaptation).
    This is critical -- Adam would escape spurious minima.
    """
    n, d = X.shape
    torch.manual_seed(seed)

    model = TwoLayerSoftplus(d, m, beta=beta)
    with torch.no_grad():
        # Xavier-style init, scaled down to start in a non-trivial basin
        nn.init.normal_(model.fc1.weight, std=0.5 / np.sqrt(d))
        nn.init.normal_(model.fc2.weight, std=0.5 / np.sqrt(m))

    trajectory = [] if record_trajectory else None
    init_loss = None
    steps_to_converge = None

    for step in range(max_steps):
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss_val = loss.item()

        if step == 0:
            init_loss = loss_val

        if record_trajectory and step % 100 == 0:
            trajectory.append(loss_val)

        # Check convergence
        if loss_val < global_threshold and steps_to_converge is None:
            steps_to_converge = step

        # Vanilla GD step
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

    # Final state
    with torch.no_grad():
        pred = model(X)
        final_loss = F.mse_loss(pred, y).item()

    # Compute gradient norm
    pred = model(X)
    loss = F.mse_loss(pred, y)
    model.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5

    converged_global = final_loss < global_threshold
    converged_spurious = (not converged_global) and (grad_norm < grad_threshold)

    # Hessian analysis
    min_eig = None
    spectral_gap = None
    eigenvalues = None

    if compute_hessian and (converged_global or converged_spurious or final_loss < 1e-2):
        try:
            min_eig, spectral_gap, eigenvalues = compute_hessian_spectrum(model, X, y)
        except Exception as e:
            print(f"  Hessian computation failed: {e}", flush=True)

    return TrainResult(
        seed=seed,
        init_loss=init_loss,
        final_loss=final_loss,
        converged_global=converged_global,
        converged_spurious=converged_spurious,
        min_hessian_eigenvalue=min_eig,
        hessian_spectral_gap=spectral_gap,
        hessian_eigenvalues=eigenvalues,
        steps_to_converge=steps_to_converge,
        final_grad_norm=grad_norm,
        loss_trajectory=trajectory,
    )


# ============================================================================
# Hessian Computation
# ============================================================================

def compute_hessian_spectrum(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, top_k: int = 20
) -> Tuple[float, float, List[float]]:
    """
    Compute full Hessian eigenvalues at current parameters.
    Returns (min_eigenvalue, spectral_gap, list_of_eigenvalues).
    """
    params = list(model.parameters())
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    total_params = sum(numels)

    # For small models, compute full Hessian
    if total_params > 5000:
        # Use Lanczos / power iteration for large models
        return _hessian_lanczos(model, X, y, top_k)

    def loss_fn(flat_params):
        idx = 0
        reconstructed = []
        for shape, numel in zip(shapes, numels):
            reconstructed.append(flat_params[idx:idx + numel].view(shape))
            idx += numel

        # Manually compute forward pass
        w1, w2 = reconstructed[0], reconstructed[1]
        h = F.softplus(X @ w1.t(), beta=model.beta)
        pred = h @ w2.t()
        return F.mse_loss(pred, y)

    flat_params = torch.cat([p.detach().clone().view(-1) for p in params])
    flat_params.requires_grad_(True)

    H = torch.autograd.functional.hessian(loss_fn, flat_params)
    H = H.detach()

    # Symmetrize (numerical stability)
    H = 0.5 * (H + H.t())

    eigenvalues = torch.linalg.eigvalsh(H)
    eigenvalues_list = eigenvalues.tolist()

    min_eig = eigenvalues[0].item()
    # Spectral gap = smallest positive eigenvalue (gap above zero)
    positive_eigs = eigenvalues[eigenvalues > 1e-10]
    if len(positive_eigs) > 0:
        spectral_gap = positive_eigs[0].item()
    else:
        spectral_gap = 0.0

    return min_eig, spectral_gap, eigenvalues_list


def _hessian_lanczos(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, num_eigs: int = 20
) -> Tuple[float, float, List[float]]:
    """
    Approximate Hessian extremal eigenvalues via Lanczos iteration.
    Uses Hessian-vector products (no full Hessian materialization).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)

    def hvp(v_flat):
        """Hessian-vector product via double backprop."""
        # Reshape v into parameter shapes
        idx = 0
        vs = []
        for p in params:
            vs.append(v_flat[idx:idx + p.numel()].view_as(p))
            idx += p.numel()

        pred = model(X)
        loss = F.mse_loss(pred, y)

        grads = torch.autograd.grad(loss, params, create_graph=True)

        # grad . v
        gv = sum((g * v).sum() for g, v in zip(grads, vs))

        hvps = torch.autograd.grad(gv, params, retain_graph=False)
        return torch.cat([h.detach().view(-1) for h in hvps])

    # Lanczos iteration
    k = min(num_eigs * 3, total_params)  # oversampling
    Q = torch.zeros(total_params, k)
    alphas = []
    betas = []

    q = torch.randn(total_params)
    q = q / q.norm()
    Q[:, 0] = q

    for j in range(k):
        w = hvp(Q[:, j])
        alpha = (Q[:, j] @ w).item()
        alphas.append(alpha)

        if j > 0:
            w = w - betas[-1] * Q[:, j - 1]
        w = w - alpha * Q[:, j]

        # Re-orthogonalize
        for i in range(j + 1):
            w = w - (Q[:, i] @ w) * Q[:, i]

        beta = w.norm().item()
        if beta < 1e-12:
            break
        betas.append(beta)

        if j + 1 < k:
            Q[:, j + 1] = w / beta

    # Build tridiagonal matrix and solve
    size = len(alphas)
    T = torch.zeros(size, size)
    for i in range(size):
        T[i, i] = alphas[i]
    for i in range(len(betas)):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]

    eigenvalues = torch.linalg.eigvalsh(T)
    eigenvalues_list = sorted(eigenvalues.tolist())

    min_eig = eigenvalues_list[0]
    positive_eigs = [e for e in eigenvalues_list if e > 1e-10]
    spectral_gap = positive_eigs[0] if positive_eigs else 0.0

    return min_eig, spectral_gap, eigenvalues_list


# ============================================================================
# Experiment Configurations
# ============================================================================

def gamma_star_approx(delta: float) -> float:
    """Approximate critical ratio: 4 / (2 + 3*delta)."""
    return 4.0 / (2.0 + 3.0 * delta)


def gamma_star_exact(delta: float) -> float:
    """Exact critical ratio for delta < 0.5: 2(1-2d)/(1-d-d^2)."""
    if delta >= 0.5:
        return gamma_star_approx(delta)
    return 2.0 * (1.0 - 2.0 * delta) / (1.0 - delta - delta ** 2)


# ============================================================================
# Experiment 1: Phase Boundary Sweep
# ============================================================================

def experiment_phase_boundary(
    n: int = 100,
    deltas: List[float] = None,
    gamma_range: Tuple[float, float] = (0.2, 2.5),
    n_gammas: int = 15,
    num_seeds: int = 20,
    beta: float = 1.0,
    lr: float = 5e-4,
    max_steps: int = 30000,
    output_dir: str = "experiments/results/softplus",
):
    """
    Sweep (delta, gamma) and measure success rate of vanilla GD.
    This should show: above gamma*, high success rate; below, lower.
    """
    if deltas is None:
        deltas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gammas).tolist()
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    total = len(deltas) * len(gammas)
    done = 0

    for delta in deltas:
        d = max(1, int(delta * n))
        gs = gamma_star_approx(delta)
        m_teacher = max(2, n // 4)

        # Generate ONE dataset per delta (all seeds train on same data, different init)
        X, y = generate_teacher_student_data(n, d, m_teacher, beta=beta, seed=42)

        delta_results = []
        for gamma in gammas:
            m = max(1, int(gamma * n))
            done += 1

            seed_results = []
            n_global = 0
            n_spurious = 0

            for seed in range(num_seeds):
                res = train_vanilla_gd(
                    X, y, m, beta=beta, seed=seed,
                    lr=lr, max_steps=max_steps,
                    compute_hessian=False,  # too slow for sweep
                    record_trajectory=False,
                )
                seed_results.append(asdict(res))
                if res.converged_global:
                    n_global += 1
                elif res.converged_spurious:
                    n_spurious += 1

            losses = [r["final_loss"] for r in seed_results]
            median_loss = float(np.median(losses))
            success_rate = n_global / num_seeds

            delta_results.append({
                "gamma": gamma,
                "m": m,
                "median_loss": median_loss,
                "mean_loss": float(np.mean(losses)),
                "min_loss": float(np.min(losses)),
                "max_loss": float(np.max(losses)),
                "success_rate": success_rate,
                "n_global": n_global,
                "n_spurious": n_spurious,
                "losses": losses,
            })

            status = "BELOW" if gamma < gs else "ABOVE"
            print(
                f"[{done}/{total}] delta={delta:.2f}, gamma={gamma:.2f} ({status} gs={gs:.2f}), "
                f"m={m}, success={success_rate:.0%}, median_loss={median_loss:.2e}",
                flush=True,
            )

        results[str(delta)] = {
            "delta": delta,
            "d": d,
            "gamma_star": gs,
            "results": delta_results,
        }

    # Save
    outpath = os.path.join(output_dir, "phase_boundary_softplus.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    return results


# ============================================================================
# Experiment 2: Spectral Gap Near Transition
# ============================================================================

def experiment_spectral_gap(
    n: int = 60,
    delta: float = 0.5,
    gamma_offsets: List[float] = None,
    num_seeds: int = 10,
    beta: float = 1.0,
    lr: float = 5e-4,
    max_steps: int = 30000,
    output_dir: str = "experiments/results/softplus",
):
    """
    Fix delta, sweep gamma near gamma*, compute Hessian eigenvalues.
    Should show: spectral gap opens linearly as gamma crosses gamma*.
    
    Uses smaller n to make full Hessian computation tractable.
    """
    if gamma_offsets is None:
        gamma_offsets = [-0.6, -0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4, 0.6]

    gs = gamma_star_approx(delta)
    d = max(1, int(delta * n))
    m_teacher = max(2, n // 4)
    os.makedirs(output_dir, exist_ok=True)

    X, y = generate_teacher_student_data(n, d, m_teacher, beta=beta, seed=42)

    results = []
    for offset in gamma_offsets:
        gamma = gs + offset
        if gamma <= 0:
            continue
        m = max(1, int(gamma * n))

        # Check if full Hessian is tractable
        total_params = d * m + m  # fc1 + fc2
        use_full = total_params <= 5000

        for seed in range(num_seeds):
            print(f"delta={delta}, offset={offset:+.2f}, gamma={gamma:.3f}, m={m}, seed={seed}...", end=" ", flush=True)

            res = train_vanilla_gd(
                X, y, m, beta=beta, seed=seed,
                lr=lr, max_steps=max_steps,
                compute_hessian=True,
                record_trajectory=True,
            )

            entry = {
                "delta": delta,
                "gamma": float(gamma),
                "gamma_star": gs,
                "gamma_offset": float(offset),
                "seed": seed,
                "final_loss": res.final_loss,
                "init_loss": res.init_loss,
                "converged_global": res.converged_global,
                "converged_spurious": res.converged_spurious,
                "min_hessian_eigenvalue": res.min_hessian_eigenvalue,
                "hessian_spectral_gap": res.hessian_spectral_gap,
                "final_grad_norm": res.final_grad_norm,
                "steps_to_converge": res.steps_to_converge,
            }
            results.append(entry)

            status = f"loss={res.final_loss:.2e}"
            if res.min_hessian_eigenvalue is not None:
                status += f", min_eig={res.min_hessian_eigenvalue:.4e}, gap={res.hessian_spectral_gap:.4e}"
            if res.converged_global:
                status += " [GLOBAL]"
            elif res.converged_spurious:
                status += " [SPURIOUS]"
            else:
                status += " [NOT CONVERGED]"
            print(status, flush=True)

    outpath = os.path.join(output_dir, "spectral_gap_softplus.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    return results


# ============================================================================
# Experiment 3: Hessian Spectral Density
# ============================================================================

def experiment_spectral_density(
    n: int = 60,
    delta: float = 1.0,
    gammas: List[float] = None,
    num_seeds: int = 5,
    beta: float = 1.0,
    lr: float = 5e-4,
    max_steps: int = 30000,
    output_dir: str = "experiments/results/softplus",
):
    """
    Compute full Hessian eigenvalue distributions at several gamma values.
    Used for spectral density plots (Fig 5 in the paper).
    """
    gs = gamma_star_approx(delta)
    if gammas is None:
        gammas = [0.5 * gs, 0.8 * gs, gs, 1.2 * gs, 1.5 * gs, 2.0 * gs]

    d = max(1, int(delta * n))
    m_teacher = max(2, n // 4)
    os.makedirs(output_dir, exist_ok=True)

    X, y = generate_teacher_student_data(n, d, m_teacher, beta=beta, seed=42)

    results = {}
    for gamma in gammas:
        m = max(1, int(gamma * n))
        total_params = d * m + m
        print(f"\ngamma={gamma:.3f} (ratio={gamma/gs:.2f}x gamma*), m={m}, params={total_params}", flush=True)

        all_eigenvalues = []
        for seed in range(num_seeds):
            print(f"  seed={seed}...", end=" ", flush=True)

            res = train_vanilla_gd(
                X, y, m, beta=beta, seed=seed,
                lr=lr, max_steps=max_steps,
                compute_hessian=True,
            )

            if res.hessian_eigenvalues is not None:
                all_eigenvalues.append(res.hessian_eigenvalues)
                print(f"loss={res.final_loss:.2e}, min_eig={res.min_hessian_eigenvalue:.4e}", flush=True)
            else:
                print(f"loss={res.final_loss:.2e}, Hessian failed", flush=True)

        results[f"{gamma:.4f}"] = {
            "gamma": float(gamma),
            "gamma_ratio": float(gamma / gs),
            "m": m,
            "eigenvalue_sets": all_eigenvalues,
        }

    outpath = os.path.join(output_dir, "spectral_density_softplus.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    return results


# ============================================================================
# Experiment 4: Activation Comparison (ReLU vs Softplus vs GELU)
# ============================================================================

def experiment_activation_comparison(
    n: int = 80,
    delta: float = 0.5,
    n_gammas: int = 12,
    gamma_range: Tuple[float, float] = (0.2, 2.5),
    num_seeds: int = 15,
    lr: float = 5e-4,
    max_steps: int = 30000,
    output_dir: str = "experiments/results/softplus",
):
    """
    Compare phase transition sharpness across activations:
    - softplus(beta=1): very smooth
    - softplus(beta=5): moderate
    - softplus(beta=20): nearly ReLU
    This tests whether smoother activations make the transition more visible.
    """
    betas = [1.0, 5.0, 20.0]
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gammas).tolist()
    gs = gamma_star_approx(delta)
    d = max(1, int(delta * n))
    m_teacher = max(2, n // 4)
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for beta in betas:
        X, y = generate_teacher_student_data(n, d, m_teacher, beta=beta, seed=42)
        beta_results = []

        for gamma in gammas:
            m = max(1, int(gamma * n))
            losses = []
            n_success = 0

            for seed in range(num_seeds):
                res = train_vanilla_gd(
                    X, y, m, beta=beta, seed=seed,
                    lr=lr, max_steps=max_steps,
                    compute_hessian=False,
                )
                losses.append(res.final_loss)
                if res.converged_global:
                    n_success += 1

            rate = n_success / num_seeds
            med = float(np.median(losses))
            beta_results.append({
                "gamma": gamma,
                "m": m,
                "success_rate": rate,
                "median_loss": med,
                "losses": losses,
            })
            status = "BELOW" if gamma < gs else "ABOVE"
            print(f"beta={beta}, gamma={gamma:.2f} ({status}), success={rate:.0%}, med_loss={med:.2e}", flush=True)

        results[f"beta_{beta}"] = {
            "beta": beta,
            "gamma_star": gs,
            "results": beta_results,
        }

    outpath = os.path.join(output_dir, "activation_comparison.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    return results


# ============================================================================
# Experiment 5: Convergence Time Near Boundary
# ============================================================================

def experiment_convergence_time(
    n: int = 100,
    delta: float = 0.5,
    gamma_offsets: List[float] = None,
    num_seeds: int = 20,
    beta: float = 1.0,
    lr: float = 5e-4,
    max_steps: int = 50000,
    output_dir: str = "experiments/results/softplus",
):
    """
    Measure how convergence time scales near the phase boundary.
    Theory predicts critical slowing down: diverging time at gamma*.
    """
    gs = gamma_star_approx(delta)
    if gamma_offsets is None:
        gamma_offsets = [-0.8, -0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5, 0.8]

    d = max(1, int(delta * n))
    m_teacher = max(2, n // 4)
    os.makedirs(output_dir, exist_ok=True)

    X, y = generate_teacher_student_data(n, d, m_teacher, beta=beta, seed=42)

    results = []
    for offset in gamma_offsets:
        gamma = gs + offset
        if gamma <= 0:
            continue
        m = max(1, int(gamma * n))

        times = []
        successes = 0
        for seed in range(num_seeds):
            res = train_vanilla_gd(
                X, y, m, beta=beta, seed=seed,
                lr=lr, max_steps=max_steps,
                compute_hessian=False,
                record_trajectory=True,
            )
            if res.steps_to_converge is not None:
                times.append(res.steps_to_converge)
                successes += 1

        entry = {
            "gamma": float(gamma),
            "gamma_offset": float(offset),
            "gamma_star": gs,
            "m": m,
            "success_rate": successes / num_seeds,
            "median_time": float(np.median(times)) if times else None,
            "mean_time": float(np.mean(times)) if times else None,
            "times": times,
        }
        results.append(entry)
        time_str = f"median={entry['median_time']:.0f}" if entry['median_time'] else "N/A"
        print(f"offset={offset:+.2f}, gamma={gamma:.2f}, success={entry['success_rate']:.0%}, time={time_str}", flush=True)

    outpath = os.path.join(output_dir, "convergence_time_softplus.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Softplus Phase Transition Experiments")
    parser.add_argument("experiment", choices=[
        "phase_boundary", "spectral_gap", "spectral_density",
        "activation_comparison", "convergence_time", "all_quick", "all_full",
    ])
    parser.add_argument("--n", type=int, default=None, help="Number of samples")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--beta", type=float, default=1.0, help="Softplus beta (1=smooth, 20=near-ReLU)")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--output", type=str, default="experiments/results/softplus")
    args = parser.parse_args()

    torch.set_num_threads(4)
    print(f"PyTorch {torch.__version__}, device=cpu, threads={torch.get_num_threads()}")
    print(f"Softplus beta={args.beta}, lr={args.lr}, max_steps={args.steps}")
    print()

    t0 = time.time()

    if args.experiment == "phase_boundary":
        experiment_phase_boundary(
            n=args.n or 100,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            num_seeds=args.seeds or 20,
            output_dir=args.output,
        )

    elif args.experiment == "spectral_gap":
        experiment_spectral_gap(
            n=args.n or 60,
            delta=args.delta or 0.5,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            num_seeds=args.seeds or 10,
            output_dir=args.output,
        )

    elif args.experiment == "spectral_density":
        experiment_spectral_density(
            n=args.n or 60,
            delta=args.delta or 1.0,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            num_seeds=args.seeds or 5,
            output_dir=args.output,
        )

    elif args.experiment == "activation_comparison":
        experiment_activation_comparison(
            n=args.n or 80,
            delta=args.delta or 0.5,
            num_seeds=args.seeds or 15,
            lr=args.lr, max_steps=args.steps,
            output_dir=args.output,
        )

    elif args.experiment == "convergence_time":
        experiment_convergence_time(
            n=args.n or 100,
            delta=args.delta or 0.5,
            num_seeds=args.seeds or 20,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            output_dir=args.output,
        )

    elif args.experiment == "all_quick":
        print("=" * 60)
        print("QUICK SUITE: small n, few seeds, testing pipeline")
        print("=" * 60)
        experiment_phase_boundary(
            n=50, deltas=[0.25, 0.5, 1.0], n_gammas=8, num_seeds=5,
            beta=args.beta, lr=args.lr, max_steps=10000,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_spectral_gap(
            n=30, delta=0.5, num_seeds=3,
            gamma_offsets=[-0.4, -0.1, 0.0, 0.1, 0.4],
            beta=args.beta, lr=args.lr, max_steps=10000,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_convergence_time(
            n=50, delta=0.5, num_seeds=5,
            gamma_offsets=[-0.4, -0.1, 0.0, 0.1, 0.4],
            beta=args.beta, lr=args.lr, max_steps=15000,
            output_dir=args.output,
        )

    elif args.experiment == "all_full":
        print("=" * 60)
        print("FULL SUITE: publication-quality runs")
        print("=" * 60)
        experiment_phase_boundary(
            n=100, num_seeds=30, n_gammas=20,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_spectral_gap(
            n=60, delta=0.5, num_seeds=15,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_spectral_density(
            n=60, delta=1.0, num_seeds=8,
            beta=args.beta, lr=args.lr, max_steps=args.steps,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_activation_comparison(
            n=80, delta=0.5, num_seeds=20,
            lr=args.lr, max_steps=args.steps,
            output_dir=args.output,
        )
        print("\n" + "=" * 60)
        experiment_convergence_time(
            n=100, delta=0.5, num_seeds=30,
            beta=args.beta, lr=args.lr, max_steps=50000,
            output_dir=args.output,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
