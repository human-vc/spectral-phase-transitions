import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def compute_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)


class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.d = d
        self.m = m
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def compute_full_hessian(model, X, y):
    params = list(model.parameters())
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]

    def loss_fn(flat_params):
        idx = 0
        w1 = flat_params[idx:idx + numels[0]].view(shapes[0])
        idx += numels[0]
        w2 = flat_params[idx:idx + numels[1]].view(shapes[1])
        h = torch.relu(X @ w1.t())
        pred = h @ w2.t()
        return nn.functional.mse_loss(pred, y)

    flat_current_params = torch.cat([p.detach().view(-1) for p in params])
    H = torch.autograd.functional.hessian(loss_fn, flat_current_params)
    return H


def train_to_minimum(model, X, y, lr=0.001, epochs=20000):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    final_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if final_loss < 1e-6:
            break

    return final_loss


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/results/spectral_gap.json")
    args = parser.parse_args()

    n = 50
    deltas = [0.5, 1.0]
    gamma_offsets = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    num_seeds = 5

    if args.quick:
        deltas = [1.0]
        gamma_offsets = [-0.3, 0.0, 0.3]
        num_seeds = 1
        n = 30

    results = []

    for delta in deltas:
        d = int(delta * n)
        gamma_star = compute_gamma_star(delta)

        for offset in gamma_offsets:
            gamma = gamma_star + offset
            if gamma <= 0:
                continue

            m = int(gamma * n)
            if m < 1:
                m = 1

            for seed in range(num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                X = torch.randn(n, d)

                m_teacher = max(1, n // 4)
                teacher = TwoLayerReLU(d, m_teacher)
                with torch.no_grad():
                    teacher.fc1.weight.normal_(0, 1 / np.sqrt(d))
                    teacher.fc2.weight.normal_(0, 1 / np.sqrt(m_teacher))
                    y = teacher(X)

                model = TwoLayerReLU(d, m)
                with torch.no_grad():
                    model.fc1.weight.normal_(0, 1 / np.sqrt(d))
                    model.fc2.weight.normal_(0, 1 / np.sqrt(m))

                final_loss = train_to_minimum(model, X, y)

                min_eig = None
                if final_loss < 1e-3:
                    H = compute_full_hessian(model, X, y)
                    eigvals = torch.linalg.eigvalsh(H)
                    min_eig = eigvals[0].item()

                entry = {
                    "delta": float(delta),
                    "gamma": float(gamma),
                    "gamma_star": float(gamma_star),
                    "gamma_offset": float(offset),
                    "min_eig": min_eig,
                    "final_loss": final_loss,
                    "seed": seed,
                    "converged": final_loss < 1e-3,
                }
                results.append(entry)
                status = f"loss={final_loss:.2e}"
                if min_eig is not None:
                    status += f", min_eig={min_eig:.4e}"
                print(f"delta={delta}, offset={offset:+.1f}, gamma={gamma:.2f}, seed={seed}, {status}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)

    plot_results(results)


def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, delta in enumerate(sorted(set(r["delta"] for r in results))):
        if ax_idx >= 2:
            break
        ax = axes[ax_idx]
        subset = [r for r in results if r["delta"] == delta and r["min_eig"] is not None]

        if not subset:
            ax.set_title(f"delta={delta} (no converged runs)")
            continue

        offsets = sorted(set(r["gamma_offset"] for r in subset))
        for off in offsets:
            eigs = [r["min_eig"] for r in subset if r["gamma_offset"] == off]
            ax.scatter([off] * len(eigs), eigs, alpha=0.6, s=30)

        offset_means = []
        mean_eigs = []
        for off in offsets:
            eigs = [r["min_eig"] for r in subset if r["gamma_offset"] == off]
            if eigs:
                offset_means.append(off)
                mean_eigs.append(np.mean(eigs))

        ax.plot(offset_means, mean_eigs, "k-o", linewidth=2, markersize=6, label="Mean min eigenvalue")
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.7)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(r"$\gamma - \gamma^*$", fontsize=12)
        ax.set_ylabel("Min Hessian Eigenvalue", fontsize=12)
        ax.set_title(f"Spectral Gap: delta={delta}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/results/spectral_gap.pdf", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_experiment()
